import numpy as np
import scipy.io as sio
import os
import yaml
import torch
from utils.dataSet import BeamDataset
from torch.utils.data import DataLoader,random_split


# read the configuration
current_pwd = os.getcwd()
with open(
        current_pwd + '/configs/config.yaml'
) as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)


####################################
# load the data
RSRPDataSet = BeamDataset(config) #pytorch dataset
data_size = RSRPDataSet.__len__()
TrainData,ValiData,TestData = random_split(RSRPDataSet, [int(0.8*data_size),int(0.1*data_size), data_size - int(0.8*data_size)- int(0.1*data_size)], generator=torch.Generator().manual_seed(config['random_seed']))
TestDataLoader = DataLoader(TestData, batch_size=ValiData.__len__())
data1, data2, Mask_true, beamMask, Adj = next(iter(TestDataLoader))

dim = data1.shape[0]

rsrp_wide_input = torch.empty((dim, 1))
rsrp_mask = torch.zeros((dim, 1))
rsrp_diff = torch.zeros((dim, 1))

#reference path
reference_path = '/projectsnbl/AHsim/qipzhu/AHsim/AH5Gsim/QIPING/beam_prediction/Gob_cbooks/powerSequence_db.mat'

if config['TrainBeamcodebookName_wide'] == 'wideBeam_linear_shift_8beams':
    reference = sio.loadmat(reference_path)['PS']
elif config['TrainBeamcodebookName_wide'] == 'wideBeam_linear_8beams':
    reference = sio.loadmat(reference_path)['PL']
elif config['TrainBeamcodebookName_wide'] == 'wideBeam_hybrid_8beams':
    reference = sio.loadmat(reference_path)['PH']


temp_narrow = data1.flatten(start_dim=1, end_dim=-1)
temp_wide = data2

BP_A = 0

for d in range(dim):
    tempN = temp_narrow[d]
    sort = np.argsort(tempN)
    rsrp_mask[d, :] = sort[-1:]

    temp_wide_DS = temp_wide[d]
    temp_wide_power = torch.pow(10,temp_wide_DS/10)
    temp_wide_power = temp_wide_power/torch.sqrt(torch.sum(torch.pow(temp_wide_power,2)))

    #change to log domain
    temp_wide_power = 10 * torch.log10(temp_wide_power)

    temp = torch.zeros(reference.shape[0])
    for l in range(reference.shape[0]):
        temp[l] = torch.linalg.norm(temp_wide_power - reference[l,:], ord=2)

    rsrp_wide_input[d, :] = torch.argmin(temp)

    rsrp_diff[d,:] = tempN[sort[-1:]] - tempN[torch.argmin(temp)]

    if rsrp_diff[d,:]<=1:
        BP_A = BP_A + 1


prob_dd = BP_A / data1.shape[0] * 100
print('NN best beam prediction probability is  with 1dB ', prob_dd, '\n')

np.savez('data/spatial/'+config['TrainBeamcodebookName_wide']+'_comparenew_'+'RX' + str(config['TrainRX'])+'.npz', rsrp_diff=rsrp_diff)