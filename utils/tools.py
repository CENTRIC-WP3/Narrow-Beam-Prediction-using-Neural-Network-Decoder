import os
import torch
import yaml
import numpy as np
from PIL import Image
import time
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import norm
import captum

#get model parameter number
def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

# get configs.yaml to a library
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

# read data directory name
def get_data_name(config):

    channel_type = config['channelType']
    currentpath = os.getcwd()
    if config['s_f']:
        RX_BEAM = '_RXbeam' + str(config['TrainRX'])

        if config['RX_BeamNum'] >1:
            sampleRate = '_TX-RXsampleRate' + str(int(config['testSampleRate_rxtx']/(config['RX_BeamNum']/config['RX_DS']))) + 'cirRX' + str(config['CircularOverRx'])

        else:
            sampleRate = '_TXsampleRate' + str(config['testSampleRate'])

        save_data_name =  currentpath + '/data/spatial/'+ channel_type + sampleRate \
                         + RX_BEAM + '_UE#' + str(config['UE_sample_num']) \
                         + '_' + config['TrainBeamcodebookName']\

        if config['W_N']:
            save_data_name = save_data_name + '_'+ config['TrainBeamcodebookName_wide']


        save_data_name = save_data_name + '.npz'

    elif config['t_f']:
       save_data_name =currentpath + '/data/temporal/' + channel_type + '_UEspeed' + str(config['UE_SPEED'])\
                       +'_TimeFrame'+ str(config['data_Total_time'])\
                       +'_channel' + str(config['training_channel']) + '.npz'

    return save_data_name

# construct the validation dataSet
def validation_data(ValiDataLoader,config,device):

    #spatial domain
    if config['s_f']:
        if config['TrainBeamcodebookName_narrow'] == 'Narrower':
            MatrixSize = config['SetC_image_shape_GT']
        else:
            MatrixSize = config['image_shape_GT']

        if config['W_N']:

            rsrp_vali, rsrp_wide, dummy, dummy, dummy = next(iter(ValiDataLoader))
            if torch.cuda.is_available():
                rsrp_vali = rsrp_vali.cuda()
                rsrp_wide = rsrp_wide.cuda()

            return rsrp_wide, rsrp_vali, dummy

        else:
            rsrp_vali, Mask_pre_vali, Mask_true_vali, beamMask_vali, Adj = next(iter(ValiDataLoader))
            x_vali, x_GT_vali, Mask_vali, Mask_vali_GT \
                = MaskGeneration(rsrp_vali, Mask_pre_vali, Mask_true_vali, config, 0, MatrixSize)
            if config['RX_BeamNum']>1:
                x_vali = x_vali.reshape((x_vali.shape[0], int(config['RX_BeamNum']/config['RX_DS']), MatrixSize[2], MatrixSize[1]))
            else:
                x_vali = x_vali.reshape((x_vali.shape[0], config['RX_BeamNum'], MatrixSize[2], MatrixSize[1]))

            # x_vali = x_vali.reshape((x_vali.shape[0], config['RX_BeamNum'], MatrixSize[2], MatrixSize[1]))

            if torch.cuda.is_available():
                x_vali = x_vali.cuda()
                x_GT_vali = x_GT_vali.cuda()
                beamMask_vali = beamMask_vali.cuda()

            return x_vali,x_GT_vali,beamMask_vali



#save model to the directory
def save_model(checkpoint_dir, iteration,config,net):
    # Save generators, discriminators, and optimizers
    model_name = get_model_name(config)
    NET_name = os.path.join(checkpoint_dir, model_name+'_%08d.pt' % iteration)
    torch.save(net.state_dict(), NET_name)

#control the model saving name
def get_model_name(config):

    if config['RX_BeamNum'] > 1:
        rx_beam = '_RXbeam' + str(config['TrainTXRX'])
    else:
        rx_beam = '_RXbeam' + str(config['TrainRX'])

    codebook = config['TrainBeamcodebookName_narrow']
    if config['W_N']:
        codebook = codebook+ '_wideBeamCodebook_' + config['TrainBeamcodebookName_wide']

    model_name = config['channelType'] + rx_beam + codebook

    if config['s_f']:
        model_name = model_name + '_SpatialDomainBP'
        if config['RX_BeamNum'] > 1:
            model_name = model_name + '_RXTX_' + str(int(config['testSampleRate_rxtx'])) + '_trainMask' + config[
                'TrainingMask'] + '_sampleRate' + str(config['TrainingSampleRate'])
        else:
            model_name = model_name + '_TX_' + str(int(config['testSampleRate_rxtx'])) + '_trainMask' + config[
                'TrainingMask'] + '_sampleRate' + str(config['TrainingSampleRate'])
    else:
        model_name = model_name + '_totalFrame' + str(config['Total_frame']) + '_sampleRate' + str(config['Time_sampling_rate'])\
                     + '_TTW' + str(config['Training_total_frame']) + '_LSTM' + str(config['num_layers'])


    return model_name


# Get model list for resume
def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name

#plot the RSRP error cdf
def cal_cdf(data):

    fig, ax = plt.subplots(constrained_layout=True)

    # getting data of the histogram
    count, bins_count = np.histogram(data, bins=50)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    ax.plot(bins_count[1:], cdf, label='ML prediction')

    # plotting PDF and CDF
    # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    ax.set_title("CDF of RSRP prediction error")
    ax.set_xlabel("RSRP prediction error (dB)")
    # ax.set_xticks(np.arange(0,10,6))
    ax.title.set_size(16)
    ax.xaxis.label.set_size(16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# generate the measured beam pattern
def MaskGeneration(x,Mask_Pred,Mask_GT,config,trainFlag, MatrixSize):

    if trainFlag:
        MaskType = config['TrainingMask']
        ds = config['TrainingSampleRate']
    else:
        MaskType = config['testMsk']
        ds = config['testSampleRate']

    RPN = config['randomPatternNum']

    X_TRAIN = torch.zeros(x.shape)
    Mask_TRAIN = Mask_Pred
    Mask_TRAIN_GT = Mask_GT
    X_GT = x

   # loop over batch
    if MaskType == 'uniform':
        # the uniform measured beam pattern (Mask_TRAIN) is generated from dataprocessing.py
        X_TRAIN_temp = x * Mask_TRAIN
        #no normalization#############
        X_TRAIN = X_TRAIN_temp
        ##############################

    # generate random measured beams
    if MaskType == 'random':
        #sample the input
        SizeRPN = int(x.shape[0]/RPN)
        for l in range(RPN):
            totel_TXbeam = x.shape[-1]
            if ds == 'random': #sample random number of beams with random pattern
                #determine the minimum beam number
                if config['RX_BeamNum'] > 1:#maximum of measurement
                    min_beam = int(x.shape[-1]*x.shape[-2]/config['testSampleRate_rxtx'])
                else:
                    min_beam = int(x.shape[-1] / config['testSampleRate'])

                # determine the maximum beam number
                max_beam = x.shape[-1]
                #take a random beam number between (min_beam, max_beam)
                ds_list = range(min_beam-1,max_beam-1)
                #
                ds_true = np.array(random.choices(ds_list,k=1))
                sample_rate = ds_true[0]
                InputIndex = random.choices(range(0,totel_TXbeam - 1), k=sample_rate)

            else: #sample fix number of beams with random pattern
                sample_rate = int(x.shape[1]*x.shape[-1]/ds)
                InputIndex = random.choices(range(0,totel_TXbeam - 1),k=sample_rate)

            #creating mask
            temp_mask = torch.zeros(x.shape[1::])

            if config['RX_BeamNum'] > 1:
                for r in range(sample_rate):
                    #generating the measured beam mask
                    temp_mask[r%x.shape[1],InputIndex[r]] = 1
                temp_mask = temp_mask.reshape(int(config['RX_BeamNum']/config['RX_DS']), MatrixSize[2]*MatrixSize[1])
            else:
                temp_mask[:,InputIndex] = 1

            #generate the measured beams
            temp = x[l*SizeRPN:(l+1)*SizeRPN,:]*temp_mask

            #concaternate the generated samples
            X_TRAIN[l*SizeRPN:(l+1)*SizeRPN,:] = temp

    return X_TRAIN,X_GT,Mask_TRAIN,Mask_TRAIN_GT

