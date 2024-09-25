import torch
import numpy as np
import torch.nn as nn
import math

class CNNPlusFNN(nn.Module):
    def __init__(self,config):
        super(CNNPlusFNN, self).__init__()

        self.config = config
        #initialize the CNN
        self.cnn = CNN(self.config)
        self.cnn_H = CNN_H_CHANNEL(self.config)
        self.cnn_W = CNN_W_CHANNEL(self.config)
        #############################
        #initialize the FNN
        if self.config['RX_BeamNum']>1:
            self.outputDim = self.config['GT_flatD']*int(self.config['RX_BeamNum']/self.config['RX_DS'])
        else:
            self.outputDim = self.config['GT_flatD']*self.config['RX_BeamNum']

        if self.config['SetC_GT_flatD'] > 64:# for set C

            if self.config['RX_BeamNum']>1:
                # self.inputDim = 512*3
                self.inputDim = 1216
            else:
                self.inputDim = 256 * 3
        else:# for set A
            self.inputDim = self.outputDim*3


        self.lnum = self.config['FNNHiddenLayerNum']

        self.Multi_linear = nn.ModuleList()
        for l in range(self.lnum):
            self.Multi_linear.append(nn.Linear(self.outputDim,self.outputDim))

        #linear hidden transform
        self.linear_conv = nn.Linear(1024, self.outputDim)
        self.linear_convH = nn.Linear(144, self.outputDim)
        self.linear_convW = nn.Linear(48, self.outputDim)

        self.norm_conv = nn.LayerNorm(self.outputDim, elementwise_affine=False)
        # self.norm_convH = nn.LayerNorm(128, elementwise_affine=False)
        # self.norm_convW = nn.LayerNorm(216, elementwise_affine=False)


        self.linear_out1 = nn.Linear(self.inputDim, self.outputDim)
        self.linear_out2 = nn.Linear(self.outputDim, self.outputDim)

        self.norm_in = nn.LayerNorm([1,4,16], elementwise_affine=False)
        self.norm_out = nn.LayerNorm(self.outputDim, elementwise_affine=False)
        self.batchNorm = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout(0.5)

        #activation function
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_norm = self.norm_in(x)
        # x_norm = self.batchNorm(x)
        #overall beams
        x_conv = self.cnn(x_norm)
        # elevation beam
        x_convH = self.cnn_H(x_norm)
        # azimuth beam
        x_convW = self.cnn_W(x_norm)

        # flatten out
        x_flat = x_conv.flatten(start_dim=1,end_dim=-1)
        x_flatH = x_convH.flatten(start_dim=1, end_dim=-1)
        x_flatW = x_convW.flatten(start_dim=1, end_dim=-1)

        #linear transform
        x_flat = self.norm_conv(self.act(self.linear_conv(x_flat)))
        x_flatH = self.norm_conv(self.act(self.linear_convH(x_flatH)))
        x_flatW = self.norm_conv(self.act(self.linear_convW(x_flatW)))

        #concaternation
        x_com = torch.cat((x_flat,x_flatH,x_flatW),dim=1)
        # for l in range(self.lnum):
        #     x = self.norm_out(self.act(self.Multi_linear[l](x_flat)))

        # x_out = self.norm_out(self.act(self.linear_out1(x_com)))
        x_out = self.norm_out(self.act(self.linear_out1(x_com)))
        x_out = self.dropout(x_out)
        x_out = self.linear_out2(x_out)

        return x_out

class CNN(nn.Module):
    def __init__(self,config):
        super(CNN, self).__init__()

        self.config = config
        if self.config['RX_BeamNum']>1:
            self.inputChannel = int(self.config['RX_BeamNum']/self.config['RX_DS'])
        else:
            self.inputChannel = self.config['RX_BeamNum']
        # self.inputChannel = self.config['RX_BeamNum']
        self.cnum = self.config['ChannelNum']
        self.HCnum = self.config['CNNHiddenlayerNum']
        self.shape = self.config['SetC_image_shape_GT']
        self.shape[0] = self.inputChannel

        self.output = self.cnum


        #activation
        self.act = nn.LeakyReLU(0.2, inplace=True)

        if self.config['SetC_GT_flatD'] >64: #Set C configuration

            k1, s1, p1, d1 = self.config['convParam1']
            k2, s2, p2, d2 = self.config['convParam2']
            k3, s3, p3, d3 = self.config['convParam3']

            #encoder conv layer
            self.conv1 = nn.Conv2d(self.inputChannel, self.cnum, k1,s1,p1,d1,bias = False)
            self.conv2 = nn.Conv2d(self.cnum, self.cnum*2, k2, s2, p2, d2,bias = False)
            self.conv3 = nn.Conv2d(self.cnum*2, self.cnum * 2, k3, s3, p3, d3,bias = False)

            #residule block
            self.conv4 = nn.Conv2d(self.cnum * 2, self.cnum * 2, k3, s3, p3, d3,bias = False)
            self.conv5 = nn.Conv2d(self.cnum * 2, self.cnum * 2, k3, s3, p3, d3,bias = False)
            self.conv6 = nn.Conv2d(self.cnum * 2, self.output , k3, s3, p3, d3,bias = False)

            # # layer normalize
            self.norm1 = nn.LayerNorm((self.cnum, 7, 31), elementwise_affine=False)
            self.norm2 = nn.LayerNorm((self.cnum*2, 4, 16), elementwise_affine=False)
            self.norm3 = nn.LayerNorm((self.output ,4, 16),elementwise_affine=False)

        else: #Set A configuration
            k1, s1, p1, d1 = self.config['A_convParam1']


            # encoder conv layer
            self.conv1 = nn.Conv2d(self.inputChannel, self.cnum, k1, s1, p1, d1, bias=False)
            self.conv2 = nn.Conv2d(self.cnum, self.cnum, k1, s1, p1, d1, bias=False)
            self.conv3 = nn.Conv2d(self.cnum, self.cnum, k1, s1, p1, d1, bias=False)

            # # layer normalize
            self.norm1 = nn.LayerNorm((self.cnum, 4, 16), elementwise_affine=False)


    def forward(self, x):

        if self.config['SetC_GT_flatD'] >64: #for set C
            #encoder
            x1 = self.norm1(self.act(self.conv1(x)))
            #################################
            x2 = self.norm2(self.act(self.conv2(x1)))
            ################################
            x3 = self.norm2(self.act(self.conv3(x2)))
            ################################
            #residule block#1
            x4 = self.norm2(self.act(self.conv4(x3)))
            x4_r = x4 + x3

            # residule block#2
            x5 = self.norm2(self.act(self.conv5(x4_r)))
            x5_r = x4_r + x5
            ################################
            #output
            y_out = self.norm3(self.act(self.conv6(x5_r)))

        else: #for set A
            x1 = self.norm1(self.act(self.conv1(x)))
            x2 = self.norm1(self.act(self.conv2(x1)))
            x3 = x1+x2
            y_out = self.norm1(self.act(self.conv2(x3)))


        return y_out

class CNN_H_CHANNEL(nn.Module):
    def __init__(self, config):
        super(CNN_H_CHANNEL, self).__init__()

        self.config = config
        if self.config['RX_BeamNum'] > 1:
            self.inputChannel = int(self.config['RX_BeamNum'] / self.config['RX_DS'])
        else:
            self.inputChannel = self.config['RX_BeamNum']
        # self.inputChannel = self.config['RX_BeamNum']
        self.cnum = self.config['ChannelNum']


        self.output = self.cnum


        # activation
        self.act = nn.LeakyReLU(0.2, inplace=True)

        if self.config['SetC_GT_flatD'] >64: #for Set C
            k1, s1, p1, d1 = self.config['convParamH_1']
            k2, s2, p2, d2 = self.config['convParamH_2']
            # encoder conv layer
            # # # layer normalize
            self.norm1 = nn.LayerNorm((self.output, 4, 16), elementwise_affine=False)

        else:#for Set A
            k1, s1, p1, d1 = self.config['A_convParamH_1']
            k2, s2, p2, d2 = self.config['A_convParamH_2']
            self.norm1 = nn.LayerNorm((self.output, 1, 9), elementwise_affine=False)

        self.conv1 = nn.Conv2d(self.inputChannel, self.output, (k1, k2), (s1, s2), (p1, p2), (d1, d2), bias=False)

    def forward(self, x):
        # encoder
        x1 = self.norm1(self.act(self.conv1(x)))

        return x1


class CNN_W_CHANNEL(nn.Module):
    def __init__(self, config):
        super(CNN_W_CHANNEL, self).__init__()

        self.config = config
        if self.config['RX_BeamNum'] > 1:
            self.inputChannel = int(self.config['RX_BeamNum'] / self.config['RX_DS'])
        else:
            self.inputChannel = self.config['RX_BeamNum']
        # self.inputChannel = self.config['RX_BeamNum']

        self.cnum = self.config['ChannelNum']


        self.output = self.cnum


        # activation
        self.act = nn.LeakyReLU(0.2, inplace=True)
        if self.config['SetC_GT_flatD'] > 64:  # for Set C
            k1, s1, p1, d1 = self.config['convParamW_1']
            k2, s2, p2, d2 = self.config['convParamW_2']

            # # # layer normalize
            self.norm1 = nn.LayerNorm((self.output, 16, 4), elementwise_affine=False)
        else:#for Set A
            k1, s1, p1, d1 = self.config['A_convParamW_1']
            k2, s2, p2, d2 = self.config['A_convParamW_2']
            # encoder conv layer
            # # # layer normalize
            self.norm1 = nn.LayerNorm((self.output, 3, 1), elementwise_affine=False)

            # encoder conv layer
            self.conv1 = nn.Conv2d(self.inputChannel, self.output, (k1, k2), (s1, s2), (p1, p2), (d1, d2),
                                   bias=False)

    def forward(self, x):
        # encoder
        x1 = self.norm1(self.act(self.conv1(x)))

        return x1