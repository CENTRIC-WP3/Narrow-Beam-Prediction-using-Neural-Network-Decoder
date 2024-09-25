import os
import torch
import torch.nn as nn
import numpy as np
from model.networks_classification import FNN
from model.networks_UEsideSpatial import CNNPlusFNN
from model.network_Temporal import BP_Temporal
from utils.tools import count_parameters
from utils.logger import get_logger
from torch.optim.lr_scheduler import StepLR

logger = get_logger()


class SpatialBPTrainer(nn.Module):
    def __init__(self, config,device):
        super(SpatialBPTrainer, self).__init__()
        self.config = config
        self.device= device

        if self.config['RX_BeamNum']>1:
            self.Rx = int(self.config['RX_BeamNum']/config['RX_DS'])
        else:
            self.Rx = self.config['RX_BeamNum']

        if self.config['TrainBeamcodebookName_narrow'] == 'Narrower':
            self.MatrixSize = self.config['SetC_image_shape_GT']
        else:
            self.MatrixSize = self.config['image_shape_GT']


        if self.config['W_N']:
            self.net = FNN(self.config)
        else:
            self.net = CNNPlusFNN(self.config)
            # self.net = FNN(self.config)

        #to cuda
        if len(device)>1:
            self.net = torch.nn.DataParallel(self.net)
            print('Devices:', self.net.device_ids)
            self.net.cuda()
        else:
            self.net.to(device[0])

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['spatial_lr'],weight_decay=1e-5)
        self.scheduler = StepLR(self.optimizer, step_size=self.config['spatial_lr_change'], gamma= self.config['spatial_lr_gamma'])


        if torch.cuda.is_available():
            self.bce_loss = nn.BCELoss(reduction = 'sum').cuda()
            self.softmax = nn.Softmax(dim=1).cuda()
        else:
            self.bce_loss = nn.BCELoss(reduction='sum')
            self.softmax = nn.Softmax(dim=1)


    def forward(self, x,x_GT,beamMask):

        losses = {}
        self.train()

        if not self.config['W_N']:
            #CNN/FNN selection
            x_input = x.reshape((x.shape[0], self.Rx, self.MatrixSize[2], self.MatrixSize[1]))
            # x_input = x.flatten(start_dim=1, end_dim=-1)
        else:
            x_input = x

        Pre = self.net(x_input)
        ##########################################################
        ## BEAM PREDICTION loss
        Pre = self.softmax(Pre)
        x_GT =  x_GT.flatten(start_dim = 1,end_dim=-1)
        GT = self.softmax(x_GT)
        # losses['bce'] = self.bce_loss(Pre, GT)
        losses['bce'] = self.bce_loss(Pre*beamMask, GT*beamMask)


        return losses







    def forward(self, RSRPFrame,RSRPFrame_label):
        losses = {}

        self.train()
        #cut the frame sequence
        if self.config['Total_frame'] > self.config['Training_total_frame']:
            RSRPFrame = RSRPFrame[:,:self.config['Training_total_frame']]
            RSRPFrame_label = RSRPFrame_label[:, :self.config['Training_total_frame']]

        #modify the data if both normalized RSRP and raw RSRP are used
        if self.config['training_channel'] > 1:
            if self.config['training_channel']:
                RSRPFrame = RSRPFrame[:,:,1]
                RSRPFrame_label = RSRPFrame_label[:,:,1]
                RSRPFrame = RSRPFrame.reshape((RSRPFrame.shape[0],RSRPFrame.shape[1],1,RSRPFrame.shape[2],RSRPFrame.shape[3]))
                RSRPFrame_label = RSRPFrame_label.reshape((RSRPFrame_label.shape[0],RSRPFrame_label.shape[1],1,RSRPFrame_label.shape[2],RSRPFrame_label.shape[3]))

        preFrame = self.net(RSRPFrame, RSRPFrame_label)
        MaskTrueFrame = RSRPFrame[:,1:]
        MaskPreFrame = preFrame

        # if self.config['training_channel']>1 and :
        #     MaskTrueFrame = MaskTrueFrame[:,:,0]
        #     MaskPreFrame = MaskPreFrame[:,:,0]
        # # L1 loss
        losses['l1_total1'] = self.l1(MaskPreFrame, MaskTrueFrame)
        # mse loss
        losses['mse_total1'] = self.mse(MaskPreFrame, MaskTrueFrame)

        return losses

