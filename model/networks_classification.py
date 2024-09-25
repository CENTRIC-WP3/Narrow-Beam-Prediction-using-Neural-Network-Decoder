import torch
import numpy as np
import torch.nn as nn

# the model can be used for beam ID prediction for
#1.  Set B is different to Set A
#2.  Set B is a subset of Set A

class FNN(nn.Module):
    def __init__(self,config):
        super(FNN, self).__init__()
        # global average pooling: feature --> point
        self.neuronNum = config['layerNum']
        if config['W_N']:
            self.inputD = config['wide_flatD']
            self.outputD = config['GT_flatD']
        else:
            if config['RX_BeamNum']>1:
                self.inputD = int(config['GT_flatD'] * config['RX_BeamNum'] /config['RX_DS'])
                self.outputD = int(config['GT_flatD'] * config['RX_BeamNum'] /config['RX_DS'])
        #block 1
        self.linear1 = nn.Linear(self.inputD,self.outputD)
        self.linear2 = nn.Linear(self.outputD, self.outputD)
        self.linear3 = nn.Linear(self.outputD, self.outputD*self.neuronNum)
        self.linear4 = nn.Linear(self.outputD*self.neuronNum, self.outputD)
        self.linear5 = nn.Linear(self.outputD, self.outputD)
        self.linear6 = nn.Linear(self.outputD, self.outputD * self.neuronNum)
        self.linear7 = nn.Linear(self.outputD * self.neuronNum, self.outputD)
        self.linear8 = nn.Linear(self.outputD, self.outputD)

        self.normL_1 = nn.LayerNorm(self.inputD, elementwise_affine=False)
        self.normL_2 = nn.LayerNorm(self.outputD, elementwise_affine=False)
        self.normL_3 = nn.LayerNorm(self.outputD * self.neuronNum, elementwise_affine=False)

        self.normL_out = nn.LayerNorm(self.outputD, elementwise_affine=False)
        self.OUT = nn.Linear(self.outputD, self.outputD)
        self.dropout = nn.Dropout(0.5)
        self.act = nn.Sigmoid()
        self.act2 = nn.ReLU()

    def forward(self, x):
        #stage 1
        # x_norm = self.normL_1(x)
        y_core = self.act2(self.linear1(x))
        y_core = self.normL_2(y_core)
        y_core = self.act2(self.linear2(y_core))
        y_core_1 = self.normL_2(y_core)
        #output size
        ########################################
        #stage2 to higher dimension
        y_core = self.act2(self.linear3(y_core))
        y_core = self.normL_3(y_core)
        y_core = self.dropout(y_core)
        y_core = self.act2(self.linear4(y_core))
        y_core = self.normL_2(y_core)
        y_core = self.act2(self.linear5(y_core))
        y_core_2 = self.normL_2(y_core)
        ########################################
        #output layer
        y_out =  y_core_1  + y_core_2
        y_out = self.normL_out(y_out)
        y_out= self.OUT(y_out)

        return y_out
