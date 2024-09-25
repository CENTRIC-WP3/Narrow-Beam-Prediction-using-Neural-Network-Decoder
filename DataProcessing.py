import numpy as np
import scipy.io as sio
from argparse import ArgumentParser
import torch
from utils.tools import get_config, get_data_name
import matplotlib.pyplot as plt


def DataPrepossing(config):
    # number of data sample related parameters
    UE_sample = config['UE_sample_num']
    UE_index = config['UE_index_num']
    drop = config['drop']
    each_drop = UE_sample * UE_index



    #wide/narrow beam prediction
    if config['W_N']:
        wide_inputD = config['wide_flatD']
        rsrp_wide_input = torch.empty((each_drop * drop, wide_inputD))

    # best N beam masks for classification
    rsrp_mask1 = torch.zeros((each_drop * drop, RX_BeamNum*outputD))
    rsrp_mask2 = torch.zeros((each_drop * drop, RX_BeamNum*outputD))
    rsrp_mask3 = torch.zeros((each_drop * drop, RX_BeamNum*outputD))
    rsrp_mask4 = torch.zeros((each_drop * drop, RX_BeamNum*outputD))



        #define the MaskSet_pred
        MaskSet_Pred[MaskSet_GT>0] = MaskforMaskSet_Pred.flatten()

    #change mask shape
    MaskSet_Pred = MaskSet_Pred.transpose(dim0 = -2, dim1 = -1)
    MaskSet_GT = MaskSet_GT.transpose(dim0 = -2, dim1 = -1)

    #flatten the measured beam mask
    MaskSet_Pred = MaskSet_Pred.flatten()
    MaskSet_GT  =MaskSet_GT.flatten()

    #for multiple Rx
    #extend the mask to rx domain
    MaskSet_Pred = MaskSet_Pred[None,:]
    MaskSet_GT = MaskSet_GT[None, :]
    MaskSet_Pred = MaskSet_Pred.repeat(RX_BeamNum, 1)
    MaskSet_GT = MaskSet_GT.repeat(RX_BeamNum, 1)



    candidateBeamNum = config['narrowBeamMaskNum'] #define N for the best N beams

    for d in range(drop):
        each_drop = UE_sample * UE_index

        for kk in range(UE_sample):
            # read the narrow beam rsrp
            rsrpdata_narrow = data_path + str((kk + 1) * UE_index) + '_RXBeam' + str(config['TrainRX']) + '.mat'
            temp_narrow = sio.loadmat(rsrpdata_narrow)['beam_map_Time']

            # read the wide beam rsrp
            if config['W_N']:
                rsrpdata_wide= data_path + str((kk + 1) * UE_index) + '_RXBeam' + str(config['TrainRX']) + '_' + config[
                    'TrainBeamcodebookName_wide'] + '.mat'
                temp_wide= sio.loadmat(rsrpdata_wide)['beam_map_Time']

            for k in range(UE_index):
                #reshape the narrow beam rsrp
                temp = temp_narrow[0][k]
                temp = temp[:outputD,::config['RX_DS']]
                temp = temp.T
                rsrp_output[d * each_drop + kk * UE_index + k, :] = torch.tensor(temp)

                # identify the best N beams ground truth
                sort = np.argsort(temp.flatten()) #over alll tx-rx beam pairs
                rsrp_mask1[d * each_drop + kk * UE_index + k, sort[-candidateBeamNum:]] = 1
                rsrp_mask2[d * each_drop + kk * UE_index + k, sort[-int(candidateBeamNum / 2):]] = 1
                rsrp_mask3[d * each_drop + kk * UE_index + k, sort[-int(candidateBeamNum / 4):]] = 1
                rsrp_mask4[d * each_drop + kk * UE_index + k, sort[-1:]] = 1

                # process the wide beam rsrp
                if config['W_N']:
                    temp_wide_DS = temp_wide[0][k].flatten()
                    rsrp_wide_input[d * each_drop + kk * UE_index + k, :] = torch.tensor(temp_wide_DS)

    beamcodebook = config['BeamCodebookPath'] + config['TrainBeamcodebookName'] + '.mat' #not used
    Adj = sio.loadmat(beamcodebook)['R'] #not used

    if config['W_N']:
        return rsrp_wide_input, rsrp_output, rsrp_mask1, rsrp_mask2, rsrp_mask3, rsrp_mask4, MaskSet_Pred, MaskSet_GT, Adj
    else:
        return rsrp_output, rsrp_mask1, rsrp_mask2, rsrp_mask3, rsrp_mask4, MaskSet_Pred, MaskSet_GT, Adj

if __name__ == '__main__':
    #read the config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help="training configuration")
    parser.add_argument('--seed', type=int, help='manual seed')
    args = parser.parse_args()
    config = get_config(args.config)

    if config['W_N']:
        rsrp_wide_input, rsrp_output, rsrp_mask1, rsrp_mask2, rsrp_mask3, rsrp_mask4, MaskSet_Pred, MaskSet_GT, Adj = DataPrepossing(config)
    else:
        rsrp_output, rsrp_mask1, rsrp_mask2, rsrp_mask3, rsrp_mask4, MaskSet_Pred, MaskSet_GT, Adj = DataPrepossing(config)


    ## getting the save data name
    save_fileName = get_data_name(config)

    # save data
    if config['W_N']:
        np.savez(save_fileName, rsrp_wide_input = rsrp_wide_input, rsrp_output=rsrp_output, rsrp_mask1=rsrp_mask1, rsrp_mask2=rsrp_mask2,
                 rsrp_mask3=rsrp_mask3,
                 rsrp_mask4=rsrp_mask4, MaskSet_Pred=MaskSet_Pred, MaskSet_GT=MaskSet_GT, Adj=Adj)
    else:
        np.savez(save_fileName, rsrp_output=rsrp_output, rsrp_mask1=rsrp_mask1, rsrp_mask2=rsrp_mask2, rsrp_mask3=rsrp_mask3,
                 rsrp_mask4=rsrp_mask4, MaskSet_Pred=MaskSet_Pred, MaskSet_GT=MaskSet_GT, Adj=Adj)