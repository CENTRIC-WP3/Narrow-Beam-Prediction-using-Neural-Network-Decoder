import numpy as np
import torch
from utils.tools import get_data_name
from torch.utils.data import Dataset
from scipy.special import softmax
import time


class BeamDataset(Dataset):
    def __init__(self, config):

        self.config = config
        self.datapath = get_data_name(self.config)

        self.data = np.load(self.datapath)

        if self.config['s_f']:  # spatial domain prediction
            if self.config['W_N']:
                self.rsrp_wide = torch.from_numpy(self.data["rsrp_wide_input"])

            self.rsrp = torch.from_numpy(self.data["rsrp_output"])
            self.beamMask = torch.from_numpy(self.data["rsrp_mask" + str(int(config["beamMask_load"]))])
            self.Mask_Pred = torch.from_numpy(self.data["MaskSet_Pred"])
            self.Mask_GT = torch.from_numpy(self.data["MaskSet_GT"])
            adj = self.data["Adj"]  # adjacency matrix
            self.Adj = torch.from_numpy(softmax(adj, axis=1))

        elif self.config['t_f']:  # temporal domain prediction
            self.RSRP_Frame = torch.from_numpy(self.data["RSRPdataFrame"])
            self.RSRP_Frame = self.RSRP_Frame[:, 0::self.config['Time_sampling_rate']]

            # cut the data
            self.RSRP_Frame = self.RSRP_Frame[:, 0:self.config['Training_total_frame'], :]
            # self.RSRP_Frame_label = torch.zeros(self.data["RSRPdataFrame_label"])
            self.RSRP_Frame_label = torch.zeros(self.RSRP_Frame.shape[1:])
            self.RSRP_Frame_label[:self.config['window_observation']] = 1

    def __len__(self):

        if self.config['s_f']:
            return self.rsrp.shape[0]
        elif self.config['t_f']:
            return self.RSRP_Frame.shape[0]

    def __getitem__(self, idx):
        if self.config['s_f']:
            if self.config['W_N']:
                rsrp = self.rsrp[idx]
                rsrp_wide = self.rsrp_wide[idx]
                dummy1 = 1
                return rsrp, rsrp_wide, dummy1, dummy1, dummy1

            else:
                rsrp = self.rsrp[idx]
                Mask_pre = self.Mask_Pred
                Mask_true = self.Mask_GT
                bm = self.beamMask[idx]
                Adj = self.Adj

                return rsrp, Mask_pre, Mask_true, bm, Adj

        elif self.config['t_f']:
            RSRP_FRAME = self.RSRP_Frame[idx].float()
            RSRP_Frame_label = self.RSRP_Frame_label.float()
            dummy1 = 1


            return RSRP_FRAME, dummy1, dummy1, RSRP_Frame_label, dummy1  # make it symmetric to spatial BP
