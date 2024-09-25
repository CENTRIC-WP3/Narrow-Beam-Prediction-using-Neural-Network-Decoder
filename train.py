import os
import shutil
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

# from tensorboardX import SummaryWriter
from trainer import SpatialBPTrainer,TemporalBPTrainer
from utils.logger import get_logger
from utils.tools import validation_data, save_model, MaskGeneration
from utils.dataSet import BeamDataset



def main():

    ######################################
    ######Initialization##################
    ######################################
    if torch.cuda.is_available():
        cuda0 = torch.device('cuda:0')
        # cuda1 = torch.device('cuda:1')
        print('Devices:', cuda0.index)
        # print('Devices:', cuda1.index)
        device = [cuda0]
        # device = [cuda0,cuda1]
        torch.cuda.synchronize()
        print('Running on GPU')
    else:
        device = [torch.device('cpu')]
        print('Running on CPU')
    #######################################
    #read the configuration
    current_pwd = os.getcwd()
    with open(
            current_pwd +'/configs/config.yaml'
    ) as f:
        config = yaml.load(f, Loader=yaml.loader.SafeLoader)
    print(config)

    # Configure checkpoint save path
    if config['s_f']: #for spatial BP
        checkpoint_path = os.path.join('checkpoints',config['ModelDirectory_spatial'])
    elif config['t_f']:#for temporal BP
        checkpoint_path = os.path.join('checkpoints',config['ModelDirectory_temporal'])

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call

    try:  # for unexpected error logging

        ######################################
        ######Data Preparation################
        ######################################
        # Define the trainer
        if config['s_f']:  # for spatial BP
            trainer = SpatialBPTrainer(config,device)

        logger.info("\n{}".format(trainer.net))
        trainer_module = trainer
        # Get the resume iteration to restart training
        ####################################
        # load the data
        RSRPDataSet = BeamDataset(config) #pytorch dataset
        data_size = RSRPDataSet.__len__()
        ######################################
        #spliting the data
        if config['s_f']: #spatial BP
            batch_size = config['Spatial_batch_size']
            TrainData, ValiData, TestData = random_split(RSRPDataSet, [int(0.8 * data_size), int(0.1 * data_size),
                                                                       data_size - int(0.8 * data_size) - int(
                                                                           0.1 * data_size)],
                                                         generator=torch.Generator().manual_seed(config['random_seed']))

       

        np.random.seed(config['random_seed'])

        #generate data loader
        TrainDataLoader = DataLoader(TrainData, batch_size=batch_size,shuffle=True)


        ########################################################################
        #preparing the validation data
        ValiDataLoader = DataLoader(ValiData, batch_size=int(ValiData.__len__()*0.1),shuffle=True)
        if config['s_f']:#spatial BP
            x_vali,x_GT_vali,beamMask_vali = validation_data(ValiDataLoader,config,device)
            # validation loss function to cuda
            if torch.cuda.is_available():
                bce_loss = nn.BCELoss(reduction='sum').cuda()
                softmax_Torch = nn.Softmax(dim=1).cuda()
                # beamMask_vali = beamMask_vali.cuda()
                # x_GT_vali = x_GT_vali.cuda()
                # x_vali = x_vali.cuda()
            else:
                bce_loss = nn.BCELoss(reduction='sum')
                softmax_Torch = nn.Softmax(dim=1)

            x_GT_vali = x_GT_vali.flatten(start_dim=1, end_dim=-1)
            ground_truth_bb_vec3 = softmax_Torch(x_GT_vali)
        ########################################################

        ########################################################

            if torch.cuda.is_available():
                l1 = nn.L1Loss().cuda()
                mse = nn.MSELoss().cuda()

            else:
                l1 = nn.L1Loss()
                mse = nn.MSELoss()



        ######################################
        ######Training #######################
        ######################################
        itter = 0
        for iteration in range(config['epochIter'] + 1):
            batch_Num = TrainDataLoader.__len__()
            for i, (data1, data2, Mask_true, beamMask, Adj) in enumerate(TrainDataLoader):
                #for temporal BP, only use RSRP AND Mask_true
                itter = itter + 1
                time_count = time.time()
                ##############################################
                #############Spatial BP######################
                #############################################
                if config['s_f']:
                    if config['TrainBeamcodebookName_narrow'] == 'Narrower':
                        MatrixSize = config['SetC_image_shape_GT']
                    else:
                        MatrixSize = config['image_shape_GT']

                    #if running the subset measured beam for prediction
                    #then it needs to generate the measured beam mask
                    if not config['W_N']:
                        x_train, x_GT, Mask_TRAIN, Mask_TRAIN_GT \
                            = MaskGeneration(data1, data2, Mask_true, config, 1, MatrixSize)

                    else:
                        x_train = data2 # wide beam measurement
                        x_GT = data1  # wide beam measurement

                    # training data to cuda
                    if  torch.cuda.is_available():
                            x_train = x_train.cuda()
                            x_GT = x_GT.cuda()
                            beamMask = beamMask.cuda()

                    #return the loss from trainer
                    losses = trainer(x_train, x_GT, beamMask)
                    losses['total'] = losses['bce']

                ###############################################
                ###############Temporal BP####################
                ##############################################
                elif config['t_f']:

                    # training data to cuda
                    RSRPFrame = data1
                    RSRPFrame_label = beamMask
                    # if len(device) > 1 and torch.cuda.is_available():
                    if torch.cuda.is_available():
                        RSRPFrame = RSRPFrame.cuda()
                        RSRPFrame_label = RSRPFrame_label.cuda()

                    #forward loss
                    losses = trainer(RSRPFrame,RSRPFrame_label)
                    losses['total'] = losses['l1_total1'] + losses['mse_total1']
                    # losses['total'] =  losses['mse_total1']


                ##############################################
                ##############################################
                ###### Backward pass ######
                trainer_module.optimizer.zero_grad()
                losses['total'].backward()
                trainer_module.optimizer.step()
                trainer_module.scheduler.step()

                #log the losses
                if config['s_f']:
                    log_losses = ['bce']
                elif config['t_f']:
                    log_losses = ['l1_total1']

                ##################################
                #validation#######################
                ###################################
                # for spatial domain BP
                if config['s_f']:
                    if np.mod(itter,config['validation_itter']) ==0:
                        trainer.net.eval()
                        x1 = trainer.net(x_vali)
                        x1_class = softmax_Torch(x1)
                        # loss_vali = bce_loss(x1_class, ground_truth_bb_vec3)
                        #beamMask_vali
                        loss_vali = bce_loss(x1_class*beamMask_vali, ground_truth_bb_vec3*beamMask_vali)
                        loss_vali = loss_vali / x1.shape[0] * config['Spatial_batch_size']

                #############################################
                #printing out ###############################
                #############################################
                if itter % config['print_iter'] == 0:
                    time_count = time.time() - time_count
                    speed = config['print_iter'] / time_count
                    speed_msg = 'speed: %.2f batches/s ' % speed

                    message = 'Iter: [%d/%d] ' % (itter, int(config['epochIter']*batch_Num))
                    for k in log_losses:
                        v = losses.get(k, 0.)
                        message += '%s: %.6f ' % (k, v)
                    if np.mod(itter, config['validation_itter']) == 0:
                        message += 'validation: %.6f ' % loss_vali
                    message += speed_msg
                    logger.info(message)

                if config['s_f']:
                    if itter % config['spatial_snapshot_save_iter'] == 0:
                        save_model(checkpoint_path, itter,config,trainer.net)
                elif config['t_f']:
                    if itter % config['temporal_snapshot_save_iter'] == 0:
                        save_model(checkpoint_path, itter,config,trainer.net)


    except Exception as e:  # for unexpected error logging
        logger.error("{}".format(e))
        raise e


if __name__ == '__main__':
    main()
