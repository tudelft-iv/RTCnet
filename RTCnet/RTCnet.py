
import sys 
import os 
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "lib"))
import os.path as osp 
import argparse
from copy import deepcopy

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib.pyplot as plt

class conv_module_para_2D(nn.Module):
    """
    The building block for constructing RTC net. This is one convolutional layer for spatial feature extraction
    """
    
    def __init__(   self, 
                    conv_kernel_size = (3, 3, 3), 
                    pool_kernel_size = 3,
                    max_pool_stride = 2,
                    input_channels = 1,
                    output_channels = 1,
                    ):
        """
        
        Parameters
        ----------
        conv_kernal_size: int
                          The kernel size of the convolutional layer
        pool_kernel_size: int
                          The kernel size of the maxpooling layer
        max_pool_stride:  int
                          The stride for maxpooling operation
        input_channels: int
                        The number of input channels of each layer. The channel means how many 3D feature map is given as input.
        output_channels: int
                         The number of output channels of each layer. The channel means how many 3D feature map is given as output.
        """
        super(conv_module_para_2D, self).__init__()

        self.pool_kernel_size = (1, pool_kernel_size, pool_kernel_size)
        self.conv_kernel_size = (conv_kernel_size[0], conv_kernel_size[1], conv_kernel_size[2])
        self.conv_padding = (int((conv_kernel_size[0]-1)/2), int((conv_kernel_size[1]-1)/2), int((conv_kernel_size[2]-1)/2))
        self.pool_padding = int(np.round((pool_kernel_size-1)/2))
        self.max_pool_stride = [1, max_pool_stride, max_pool_stride]
        
        self.conv_layer = nn.Conv3d(
            input_channels, output_channels, self.conv_kernel_size, stride=1, padding=self.conv_padding, dilation=1, groups=1, bias=False, padding_mode='zeros'
        )
        self.max_pool = nn.MaxPool3d(
            self.pool_kernel_size, stride=self.max_pool_stride, padding=self.pool_padding, dilation=1, return_indices=False, ceil_mode=False
        )
        self.relu_layer = nn.ReLU()
        self.BN = nn.BatchNorm3d(num_features=output_channels)


    def forward(self, features):
        """
        Define forward pass with maxpooling, ReLU and Batch Normalization
        """
        features = self.conv_layer(features)
        features = self.max_pool(features)
        features = self.relu_layer(features)
        features = self.BN(features)
        return features


class conv_module_1D(nn.Module):
    """
    The building block for constructing RTC net. This is one convolutional layer for speed feature extraction
    """

    def __init__(   self, 
                    input_channels=1, 
                    conv_kernel_size=3, 
                    pool_kernel_size=3,
                    output_channels=16,
                    max_pool_stride=2,
                    ):
        """
        
        Parameters
        ----------
        input_channels: int
                       The number of input channels for the 1D convolution operation
        conv_kernel_size: int
                        The size of the convolutional kernal 
        pool_kernel_size: int
                          The size of the maxpooling kernal
        output_channels: int
                        The number of output channels 
        max_pool_stride: int
                         The number of stride for maxpooling
        """
        super(conv_module_1D, self).__init__()
        self.input_channels = input_channels
        self.pool_kernel_size = pool_kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.padding = int(np.round((conv_kernel_size-1)/2))
        self.padding_pool = int(np.round((pool_kernel_size-1)/2))
        self.output_channels = output_channels
        self.max_pool_stride = max_pool_stride
        
        self.conv_layer = nn.Conv1d(
            self.input_channels, self.output_channels, self.conv_kernel_size, stride=1, padding=self.padding, dilation=1, groups=1, bias=False, padding_mode='zeros'
        )
        self.max_pool = nn.MaxPool1d(
            self.pool_kernel_size, stride=self.max_pool_stride, padding=self.padding_pool, dilation=1, return_indices=False, ceil_mode=False
        )
        self.relu_layer = nn.ReLU()
        self.BN = nn.BatchNorm1d(num_features=self.output_channels)


    def forward(self, features):
        """
        Define forward pass with maxpooling, ReLU and Batch Normalization
        """
        features = self.conv_layer(features)
        features = self.max_pool(features)
        features = self.relu_layer(features)
        features = self.BN(features)
        return features



class RTCnet(nn.Module):
    """
    The construction of entire RTCnet 
    """
    def __init__(self, 
                num_classes=4, 
                Doppler_dims=32, 
                high_dims = 4, 
                dropout= True,
                input_size = 5):
        """  
        The default dimension of low-level cropping is (window_size, window_size, 32) 
        The default high-level features are  (x, y, vel, rcs)  
        The default classes are (others, pedestrian, biker, car)

        Parameters
        ----------
        num_classes: int 
                     The number of classes
        Doppler_dims: int
                     The number of grids in Doppler (speed) dimension
        high_dims: int
                   The number of high-level features
        drop_out: bool
                  Whether drop-out is applied during training
        input_size: int
                    The size in space when cropping the low-level radar cube
        """

        super(RTCnet, self).__init__()
        self.num_classes = num_classes
        self.Doppler_dims = Doppler_dims
        self.high_dims = high_dims
        self.input_size = input_size
        self.dropout = dropout

        ##### Define kernal in space (x, y) dimension  #####
        self.stride_pool_para_2D = [2, 2] # This stride is in the space dimension, i.e. in H and W dimension of 3D input (N, C, D, H, W)
        self.conv_kernels_para_2D = [np.array([3, 3, 3]), np.array([3, 3, 3])] # This kernel size is in the space dimension, i.e. in H and W dimension of 3D input (N, C, D, H, W)
        self.pool_kernels_para_2D = [2, 2] # This kernel size is in the space dimension, i.e. in H and W dimension of 3D input (N, C, D, H, W)
        self.channels_para_2D = [1, 6, 25]

        ##### Define kernel in speed (v) dimension #####
        self.channels_1D = [self.channels_para_2D[-1], 16, 32, 32] # Contains 1 input channel
        self.stride_pool_1D = [2, 2, 2]
        self.conv_kernels_1D = [7, 7, 7]
        self.pool_kernels_1D = [3, 3, 3]


        ##### Define nodes for Fully-connected layer #####
        self.FCN_nodes = [self.channels_1D[-1]*4 + self.high_dims, 128, 128, num_classes]

        self.conv_modules_para_2D = nn.ModuleList()

        ##### Construct CNN for space dimension feature extraction #####
        for i, conv_kernel_para_2D in enumerate(self.conv_kernels_para_2D):
            self.conv_modules_para_2D.append(
                conv_module_para_2D(
                    conv_kernel_size = conv_kernel_para_2D, 
                    pool_kernel_size = self.pool_kernels_para_2D[i],
                    max_pool_stride = self.stride_pool_para_2D[i],
                    input_channels = self.channels_para_2D[i],
                    output_channels = self.channels_para_2D[i+1]
                )
            )


        self.pool_para_2D_end = nn.MaxPool2d(kernel_size = (2,2), stride=1, padding=0)

        ##### Construct CNN for speed dimension feature extraction #####
        self.conv_modules_1D = nn.ModuleList()
        for i, conv_kernel_1D in enumerate(self.conv_kernels_1D):
            self.conv_modules_1D.append(
                conv_module_1D(
                    input_channels = self.channels_1D[i],
                    conv_kernel_size = conv_kernel_1D, 
                    pool_kernel_size = self.pool_kernels_1D[i],
                    output_channels = self.channels_1D[i+1],
                    max_pool_stride = self.stride_pool_1D[i],
                )
            )
        
        self.pool_conv_1D_end = nn.MaxPool1d(
            kernel_size = int(4), stride=1, padding=0
        )
        
        ##### Construct Fully-connected layers #####
        self.FCN_modules = nn.ModuleList()
        for i in range(len(self.FCN_nodes)-1):
            if self.dropout and i !=0:
                self.FCN_modules.append(
                    nn.Dropout(p=0.5)
                )
        
            self.FCN_modules.append(
                nn.Conv1d(
                    in_channels = self.FCN_nodes[i],
                    out_channels = self.FCN_nodes[i+1],
                    kernel_size = 1,
                    stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode = "zeros"
                )
            )
            self.FCN_modules.append(
                nn.ReLU()
            )
        


    def _break_up_feat(self, feat):
        """ 
        The features is a tensor with [high-level-vector, low-level-vector]. This function returns high-level feature and low-level feature separately
        """
        high_feat = feat[..., :self.high_dims].contiguous()
        low_feat = feat[..., self.high_dims:].contiguous()
        return high_feat, low_feat

    def forward(self, features):
        """ 
        Define forward pass 
        """
        high_feat, low_feat = self._break_up_feat(features)
        high_feat = high_feat.view(-1, 1, self.high_dims)
        # Reshape the low-level feature into a multi-channel image
        low_feat = low_feat.view(-1, 1, self.input_size, self.input_size, self.Doppler_dims)
        low_feat = low_feat.permute(0, 1, 4, 2, 3).contiguous()
        # The multi-scale 2D convolutional layers
        for i in range(len(self.conv_modules_para_2D)):
            low_feat = self.conv_modules_para_2D[i](low_feat)
        low_feat = low_feat.view(-1,  self.channels_para_2D[-1], self.Doppler_dims)

        for i in range(len(self.conv_modules_1D)):
            low_feat = self.conv_modules_1D[i](low_feat)
        low_feat = low_feat.view(-1, self.channels_1D[-1]*4, 1).contiguous()
        high_feat = high_feat.transpose(1,2).contiguous()
        full_feat = torch.cat((high_feat, low_feat), 1).contiguous()

        for i in range(len(self.FCN_modules)):
            full_feat = self.FCN_modules[i](full_feat)    
        return full_feat


        

        
