
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
    
    def __init__(   self, 
                    conv_kernel_size = 3, 
                    pool_kernel_size = 3,
                    max_pool_stride = 2,
                    input_channels = 1,
                    output_channels = 1,
                    ):
        super(conv_module_para_2D, self).__init__()

        self.pool_kernel_size = (1, pool_kernel_size, pool_kernel_size)
        self.conv_kernel_size = (conv_kernel_size[0], conv_kernel_size[1], conv_kernel_size[2])
        # self.conv_kernel_size = conv_kernel_size
        # self.conv_padding = np.round((conv_kernel_size-1)/2).astype(np.int)
        self.conv_padding = (int((conv_kernel_size[0]-1)/2), int((conv_kernel_size[1]-1)/2), int((conv_kernel_size[2]-1)/2))
        self.pool_padding = int(np.round((pool_kernel_size-1)/2))
        # print(self.conv_padding)
        # print(self.pool_padding)
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
        features = self.conv_layer(features)
        features = self.max_pool(features)
        features = self.relu_layer(features)
        features = self.BN(features)
        return features


class conv_module_1D(nn.Module):
    
    def __init__(   self, 
                    input_channels=1, 
                    conv_kernel_size=3, 
                    pool_kernel_size=3,
                    output_channels=16,
                    max_pool_stride=2,
                    ):
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
        features = self.conv_layer(features)
        features = self.max_pool(features)
        features = self.relu_layer(features)
        features = self.BN(features)
        return features



class RTCnetV4(nn.Module):

    def __init__(self, 
                num_classes=4, 
                Doppler_dims=32, 
                high_dims = 4, 
                dropout= True,
                input_size = 5):
        # The default low-level is window_size * window_size * 32 image cropped from radar cube
        # The default high-level is  0:x  1:y  2:vel 3:rcs  
        # The default num_classes is four: others, pedestrian, biker, car 
        super(RTCnetV4, self).__init__()
        self.num_classes = num_classes
        self.Doppler_dims = Doppler_dims
        self.high_dims = high_dims
        self.input_size = input_size
        self.dropout = dropout

        self.stride_pool_para_2D = [2, 2] # This stride is in the space dimension, i.e. in H and W dimension of 3D input (N, C, D, H, W)
        self.conv_kernels_para_2D = [np.array([3, 3, 3]), np.array([3, 3, 3])] # This kernel size is in the space dimension, i.e. in H and W dimension of 3D input (N, C, D, H, W)
        self.pool_kernels_para_2D = [2, 2] # This kernel size is in the space dimension, i.e. in H and W dimension of 3D input (N, C, D, H, W)
        self.channels_para_2D = [1, 6, 25]

        self.channels_1D = [self.channels_para_2D[-1], 16, 32, 32] # Contains 1 input channel
        self.stride_pool_1D = [2, 2, 2]
        self.conv_kernels_1D = [7, 7, 7]
        self.pool_kernels_1D = [3, 3, 3]



        self.FCN_nodes = [self.channels_1D[-1]*4 + self.high_dims, 128, 128, num_classes]

        self.conv_modules_para_2D = nn.ModuleList()

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
        # The features is a tensor with [high-level-vector, low-level-vector]
        high_feat = feat[..., :self.high_dims].contiguous()
        low_feat = feat[..., self.high_dims:].contiguous()
        return high_feat, low_feat

    def forward(self, features):
        # print(features.size())
        high_feat, low_feat = self._break_up_feat(features)
        high_feat = high_feat.view(-1, 1, self.high_dims)
        # Reshape the low-level feature into a multi-channel image
        low_feat = low_feat.view(-1, 1, self.input_size, self.input_size, self.Doppler_dims)
        low_feat = low_feat.permute(0, 1, 4, 2, 3).contiguous()
        # The multi-scale 2D convolutional layers
        for i in range(len(self.conv_modules_para_2D)):
            low_feat = self.conv_modules_para_2D[i](low_feat)
            # print("after {}th para conv 2D:{}".format(i, low_feat.size()))
        low_feat = low_feat.view(-1,  self.channels_para_2D[-1], self.Doppler_dims)
        # low_feat = self.pool_para_2D_end(low_feat)
        # print("after pool :{}".format(low_feat.size()))
        # low_feat = low_feat.view(-1, 1, self.Doppler_dims)
        # The 1D convolutional along the Doppler dimension
        # print("input conv 1D:{}".format(low_feat.size()))

        for i in range(len(self.conv_modules_1D)):
            low_feat = self.conv_modules_1D[i](low_feat)
            # print("after {}th conv 1D-ext:{}".format(i, low_feat.size()))
            # print(low_feat)
        # low_feat = self.pool_conv_1D_end(low_feat)
        # print("after 1D_end pool:", low_feat.size())
        # print(low_feat)
        # Concatenate the high-level with low-level
        low_feat = low_feat.view(-1, self.channels_1D[-1]*4, 1).contiguous()
        high_feat = high_feat.transpose(1,2).contiguous()
        # print(high_feat)
        # print(high_feat.size())
        full_feat = torch.cat((high_feat, low_feat), 1).contiguous()
        # print(full_feat)
        # print(full_feat.size())

        for i in range(len(self.FCN_modules)):
            full_feat = self.FCN_modules[i](full_feat)    
        # print(full_feat)
            # print(full_feat.size())
        return full_feat


        

if __name__ == "__main__":
    from TargetLoader import TargetModeDataset, RemoveFeatures, ToTensor
    parser = argparse.ArgumentParser()
    home = osp.expanduser("~")
    parser.add_argument("--data", default = osp.join(home, "ros", "conti_ws", "src", "radarcube","src","data","train","low"))
    parser.add_argument("--gpu", type=bool, default = True)
    args = parser.parse_args()
    data_path = args.data

    batch_size = 512
    num_batches = 1000
    lr = 5e-2
    lr_decay_step = 200

    rm_features = RemoveFeatures([])
    # The default high-level is  0:x  1:y  2:vel  3:range  4:rcs  5:VrelRadVar  6:angle  7:abs_vel
    to_tensor = ToTensor()
    composed_trans = transforms.Compose([rm_features, to_tensor])
    target_dataset = TargetModeDataset(data_path, transform=composed_trans,  mode='train', split_ratio =1)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    
    model = RTCnet(
        num_classes=4,
        low_dims=32,
        high_dims=8
    )
    sftmax = nn.Softmax(dim=1)
    model.double()

    target_iter = iter(target_loader)
    feature = target_iter.next()
    x = feature['data']
    y = feature['label']
    # plt.scatter(x.numpy()[:,0,1], x.numpy()[:,0,0], c = y.numpy()[:,0])

    num_other = (y==0).float().sum()
    num_bikers = (y==2).float().sum()
    num_ped = (y==1).float().sum()
    num_car = (y==3).float().sum()
    print("num_others:{}, num_bikers:{}, num_ped:{}, num_car:{}".format(num_other, num_ped, num_bikers, num_car))
    weights = np.array([1/num_other,1/num_ped,1/num_bikers,1/num_car])
    loss_fn = nn.CrossEntropyLoss(torch.tensor(weights,dtype=torch.double))
    y_pred_list = []
    for i in range(num_batches):
        if i%lr_decay_step==0:
            lr = lr/2
            print("Half the learning rate to :{}".format(lr))
        y_pred = model(x)
        sft_max_score = sftmax(y_pred)
        # print(sftmax(y_pred[1]))
        # y_pred_list.append(np.sum(sft_max_score.detach().numpy(), axis=2))
        loss = loss_fn(y_pred.view(y.numel(),-1), y.view(-1))
        y_pred_list.append(sft_max_score.detach().numpy())
        _, pred_class = torch.max(loss,-1)
        acc = (pred_class == y).float().sum() / y.numel()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad

        # print("finished:{}th batch, loss:{}, acc:{}".format(i, loss, acc))
    y_pred_list = np.array(y_pred_list)
    y_pred_list = np.transpose(y_pred_list,[0,1,3,2])

    ### Visualize the softmax layer 
    fig1 = plt.figure(1, figsize=(30,15))
    ax1 = plt.subplot(221) 
    others_lw = 1
    GT_lw = 2
    GT_marker = '-'
    others_marker = '-'
    for i in range(y_pred_list.shape[1]):
        ax1.plot(np.arange(y_pred_list.shape[0]), y_pred_list[:,i,0, 0 ],'c'+others_marker, linewidth=others_lw)
        if y[i]==0:
            ax1.plot(np.arange(y_pred_list.shape[0]), y_pred_list[:,i,0, 0 ] , 'y'+GT_marker, markersize=GT_lw)
        ax1.set_title("others:{}".format(num_other))
        ax1.set_xlabel("number of batches")
        ax1.set_ylabel("Softmax score")
        ax1.set_ylim([0,1])
    ax2 = plt.subplot(222)
    for i in range(y_pred_list.shape[1]):
        ax2.plot(np.arange(y_pred_list.shape[0]), y_pred_list[:,i,0, 1],'c'+others_marker, linewidth=others_lw)
        if y[i]==1:
            ax2.plot(np.arange(y_pred_list.shape[0]), y_pred_list[:,i,0, 1 ] , 'g'+GT_marker, markersize=GT_lw)
        ax2.set_title("pedestrian:{}".format(num_ped))
        ax2.set_ylabel("Softmax score")
        ax2.set_ylim([0,1])
        ax2.set_xlabel("number of batches")
    ax3 = plt.subplot(223)
    for i in range(y_pred_list.shape[1]):
        ax3.plot(np.arange(y_pred_list.shape[0]), y_pred_list[:,i,0, 2],'c'+others_marker, linewidth=others_lw)
        if y[i]==2:
            ax3.plot(np.arange(y_pred_list.shape[0]), y_pred_list[:,i,0, 2 ] , 'r'+GT_marker, markersize=GT_lw)
        ax3.set_title("biker:{}".format(num_bikers))
        ax3.set_ylim([0,1])
        ax3.set_ylabel("Softmax score")
        ax3.set_xlabel("number of batches")
    ax4 = plt.subplot(224)
    for i in range(y_pred_list.shape[1]):
        ax4.plot(np.arange(y_pred_list.shape[0]), y_pred_list[:,i,0, 3],'c'+others_marker, linewidth=others_lw)
        if y[i]==3:
            ax4.plot(np.arange(y_pred_list.shape[0]), y_pred_list[:,i,0, 3 ] , 'b'+GT_marker, markersize=GT_lw)
        ax4.set_title("car:{}".format(num_car))
        ax4.set_xlabel("number of batches")
        ax4.set_ylabel("Softmax score")
        ax4.set_ylim([0,1])

    fig2 = plt.figure(2)
    plt.imshow(torch.cat((x.squeeze(), y.double()),1), interpolation="nearest", aspect="auto")
    plt.xticks(np.array([0,1,2,3,4,5,6,7,40]), ["x","y","v","r" ,"rcs" ,"Var" ,"ang" ,"V", "label"])
    plt.show()
        