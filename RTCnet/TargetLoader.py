######### This script requires #############
######### Pytorch 1.0 ######################

# Import system related modules
import os.path as osp 
import os 
import sys 
from copy import deepcopy

# Import Pytorch
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Import other third-party modules
import numpy as np 
import matplotlib.pyplot as plt 
import json 

        
class TargetModeDataset(Dataset):
    """ 
    Dataset class to define the dataset of RTCnet, details refer to 
    https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset
    """
    def __init__(self, dataset_folder,  transform, mode='train', high_dims = 4, normalize = True, 
                feature_type = 'low', norms_path ="", speed_limit = 0, dist_near = 0, 
                binary_class = False, dist_far = 100, rm_cars = False, rm_position = False, zero_low=False,
                only_slow = False,
                only_fast = False,
                rm_speed = False,
                rm_rcs = False):
        """  
        Constructor for the TargetModeDataset

        Parameters
        ----------
        dataset_folder: string
                        The folder of the dataset
        transform: torchvision.transform
                   Transforms the input array. The function is defined in the end of this file
        mode: string
              Whether the dataset is used for training or testing
        high_dims: int
                   The number of dimensions for high-level features
        normalize: bool
                   Whether normalization is used
        feature_type: bool
                    If feature_type is "high", only high-level feature will be used
        norms_path: string
                    The path for saving normalization parameters 
        speed_limit: float
                     The limit for highest speed. Data with higher speed will stay
        dist_near: float
                   Due to reflection of bumpers, the data that is too close can be filtered by the distance 
        binary_class: bool
                      Whether the labels will be changed into binary (one over all)
        dist_far: float
                  Due to the reflection from very far away is not informative enough, the data can be filtered by max distance
        rm_cars: bool
                 (This is only used during experimental period) whether cars are ignored
        rm_position: bool
                     (This is only used during experimental period) whether position information is removed 
        zero_low: bool
                  (This is only used during experimental period)  whether low-level features are set as zero
        only_slow: bool
                   (This is only used during experimental period) whether data is filtered by a specific speed upper limit
        only_fast: bool
                   (This is only used during experimental period) whether data is filtered by a specific speed lower limit
        rm_speed: bool
                  (This is only used during experimental period) whether the speed is removed from features
        rm_rcs: bool
                (This is only used during experimental period) whether RCS(reflection) is removed from features
        """
        self.data_folder_path = dataset_folder
        self.mode = mode
        self.speed_limit = speed_limit
        self.dist_near = dist_near
        self.dist_far = dist_far
        self.binary_class = binary_class
        self.transform = transform
        self.rm_cars = rm_cars
        self.features_all = None 
        self.labels_all = None
        self.speed_upper_lim = 2
        self.rm_rcs =rm_rcs
        self.speed_lower_lim = 3
        if mode == "test":
            """ 
            Dataset used for testing
            """
            self.features = np.load(osp.join(self.data_folder_path, "test", "features.npy"))
            print("{} feature shape:{} ".format(mode, self.features.shape))
            self.labels = np.load(osp.join(self.data_folder_path, "test", "labels.npy")).astype(np.uint8)
            if only_slow:
                self.labels = self.labels[np.abs(self.features[:,2])<self.speed_upper_lim]
                self.features = self.features[np.abs(self.features[:,2])<self.speed_upper_lim,:]
            if only_fast:
                self.labels = self.labels[np.abs(self.features[:,2])>self.speed_lower_lim]
                self.features = self.features[np.abs(self.features[:,2])>self.speed_lower_lim,:]
            self.features, self.labels = self.rm_static(self.features, self.labels)
            print("{} features after removing static targets:{}".format(mode, self.features.shape))
            self.features, self.labels = self.rm_close(self.features, self.labels)
            print("{} features after removing neaby and far away targets:{}".format(mode, self.features.shape))
            if zero_low:
                self.features[:, high_dims:] = 0
            if normalize:
                high_mean = np.load(osp.join(norms_path, 'high_mean.npy'))
                low_mean = np.load(osp.join(norms_path, "low_mean.npy"))
                low_std = np.load(osp.join(norms_path, "low_std.npy"))
                high_std = np.load(osp.join(norms_path, "high_std.npy"))
                self.features = self.normalize(self.features, high_dims, high_mean, low_mean, high_std, low_std)
            self.features_chosen = self.features
            self.labels_chosen = self.labels

        elif mode == "train":
            """  
            Dataset used for training
            """
            self.features = np.load(osp.join(self.data_folder_path, "train", "features.npy"))
            print("{} feature shape:{} ".format(mode, self.features.shape))
            self.labels = np.load(osp.join(self.data_folder_path,  "train", "labels.npy")).astype(np.uint8)
            if only_slow:
                self.labels = self.labels[np.abs(self.features[:,2])<self.speed_upper_lim]
                self.features = self.features[np.abs(self.features[:,2])<self.speed_upper_lim,:]
            if only_slow:
                self.labels = self.labels[np.abs(self.features[:,2])>self.speed_lower_lim]
                self.features = self.features[np.abs(self.features[:,2])>self.speed_lower_lim,:]
            self.features, self.labels = self.rm_static(self.features, self.labels)
            print("{} features after removing static targets:{}".format(mode, self.features.shape))
            self.features, self.labels = self.rm_close(self.features, self.labels)
            print("{} features after removing neaby and far away targets:{}".format(mode, self.features.shape))
            if normalize:
                high_mean = np.mean(self.features[:,:high_dims], axis=0) # A vector 
                low_mean = np.mean(self.features[:, high_dims:])     # A number 
                high_std = np.std(self.features[:, :high_dims], axis=0)  # A vector
                low_std = np.std(self.features[:, high_dims:])       # A number 
                np.save(osp.join(norms_path, "high_mean.npy"), high_mean)
                np.save(osp.join(norms_path, "low_mean.npy"), low_mean)
                np.save(osp.join(norms_path, "high_std.npy"), high_std)
                np.save(osp.join(norms_path, "low_std.npy"), low_std)
                
                self.features = self.normalize(self.features, high_dims, high_mean, low_mean, high_std, low_std)
            self.features_chosen = self.features
            self.labels_chosen = self.labels
        elif mode == "val":
            """ 
            Dataset used for validation
            """
            self.features = np.load(osp.join(self.data_folder_path, "val", "features.npy"))
            print("{} feature shape:{} ".format(mode, self.features.shape))
            self.labels = np.load(osp.join(self.data_folder_path, "val", "labels.npy")).astype(np.uint8)
            if only_slow:
                self.labels = self.labels[np.abs(self.features[:,2])<self.speed_upper_lim]
                self.features = self.features[np.abs(self.features[:,2])<self.speed_upper_lim,:]
            if only_slow:
                self.labels = self.labels[np.abs(self.features[:,2])>self.speed_lower_lim]
                self.features = self.features[np.abs(self.features[:,2])>self.speed_lower_lim,:]
            self.features, self.labels = self.rm_static(self.features, self.labels)
            print("{} features after removing static targets:{}".format(mode, self.features.shape))
            self.features, self.labels = self.rm_close(self.features, self.labels)
            print("{} features after removing neaby and far away targets:{}".format(mode, self.features.shape))
            if normalize:
                high_mean = np.load(osp.join(norms_path, 'high_mean.npy'))
                low_mean = np.load(osp.join(norms_path, "low_mean.npy"))
                low_std = np.load(osp.join(norms_path, "low_std.npy"))
                high_std = np.load(osp.join(norms_path, "high_std.npy"))
                self.features = self.normalize(self.features, high_dims, high_mean, low_mean, high_std, low_std)
            self.features_chosen = self.features
            self.labels_chosen = self.labels

        else:
            raise NotImplementedError
        if self.binary_class:
            self.labels[self.labels!=1] = 0 
        if self.rm_cars:
            self.labels[self.labels==3] = 0
        if feature_type == "high":
            self.features = self.features[:, :high_dims]
        if rm_position:
            self.features = self.features[:, 2:]
            print("remove_positions, dimension:{}".format(self.features.shape))
        if rm_speed:
            self.features[:, 2] = 0
        if rm_rcs:
            self.features[:, 3] = 0

        # self.features[:, 2] = np.abs(self.features[:, 2])
        self.indx_valid = np.logical_and(self.indx_valid_close, self.indx_valid_v)

    def __len__(self):
            return self.features.shape[0]
    
    def __getitem__(self, idx):
        feature = np.expand_dims(self.features[idx,:], axis=0)
        label = self.labels[idx]
        label = np.expand_dims(label, axis=0)
        sample = {'data':feature,
                    'label':label}
        if self.transform:
            sample = self.transform(sample)
        return sample  


    def normalize(self, features, high_dims, high_mean, low_mean, high_std, low_std):
        """  
        Normalize the features by subtracting mean and dividing standard deviation
        """
        features[:, :high_dims] = np.subtract(features[:, :high_dims], high_mean)
        features[:, :high_dims] = np.divide(features[:, :high_dims], high_std)
        features[:, high_dims:] = np.subtract(features[:, high_dims:], low_mean)        
        features[:, high_dims:] = np.divide(features[:, high_dims:], low_std)

        return features

    def rm_static(self, features, labels):
        """  
        Used for filtering static targets under certain speed threshold
        """
        self.indx_valid_v = np.abs(features[:, 2]) > self.speed_limit
        features = features[self.indx_valid_v, :]
        labels = labels[self.indx_valid_v]
        return features, labels

    def rm_close(self, features, labels):
        """  
        Used for filtering targets that are closer than a distance threshold
        """
        self.indx_valid_close = np.logical_and(features[:, 0] > self.dist_near, features[:, 0] < self.dist_far) 
        features = features[self.indx_valid_close, :]
        labels = labels[self.indx_valid_close]
        return features, labels

    def to_ovo(self, class_positive, class_negative):
        """  
        Transform the dataset into binary dataset for one-vs-one training and testing
        """
        if self.features_all is None or self.labels_all is None:
            raise ValueError
        valid_indx = np.logical_or(self.labels_all == class_positive, self.labels_all == class_negative)
        self.labels = self.labels_all[valid_indx]
        self.features = self.features_all[valid_indx]
        self.labels = (self.labels == class_positive).astype(np.uint8)

    def to_ova(self, chose_class):
        """  
        Transform the dataset into binary dataset for one-vs-all training and testing
        """
        if self.features_all is None or self.labels_all is None:
            raise ValueError
        valid_indx = (self.labels_all == chose_class).astype(np.uint8)
        self.labels = valid_indx
        

    def save_features_labels(self):
        """  
        Cache labels and features for ova and ovo transform
        """
        self.labels_all = self.labels
        self.features_all = self.features


### Transforms #####


class ToTensor(object):

    def __call__(self, sample):
        feature = sample['data']
        label = sample['label']
        return {"data":torch.tensor(feature), "label": torch.tensor(label,dtype =  torch.int64)}

class Permutation(object):

    def __call__(self, sample):
        feature = sample['data']
        label = sample['label']
        # The permutation should use 0-1, because the data is normalized 
        feature[:, 0] = feature[:, 0] + np.random.random()*0.1-0.1
        feature[:, 2] = feature[:, 2] + np.random.random()*0.1-0.1
        return {"data":feature, "label":label}

