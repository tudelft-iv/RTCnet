######### This script requires #############
######### Pytorch 1.0 ######################

import os.path as osp 
import os 
import sys 

import numpy as np 

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt 

from copy import deepcopy

import json 
def bootstrap_index(ind_before_bs):
    """ 
    Params:
            ind_before_bs: index before boot strapping 
    
    Return:
            ind_after_bs: index after bootstrapping
    """
    ind_after_bs = np.zeros(ind_before_bs.shape)
    size_A = ind_before_bs.shape[0]
    for i in range(size_A):
        random_ind = np.random.uniform(0, size_A)
        ind_after_bs[i] = ind_before_bs[random_ind]
    
    return ind_after_bs
        
class TargetModeDataset(Dataset):

    def __init__(self, dataset_folder,  transform, mode='train', high_dims = 4, normalize = True, 
                feature_type = 'low', norms_path ="", speed_limit = 0, dist_near = 0, 
                binary_class = False, dist_far = 100, rm_cars = False, rm_position = False, zero_low=False,
                only_slow = False,
                only_fast = False,
                rm_speed = False,
                rm_rcs = False,
                do_bootstrapping = False):
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
        self.do_bootstrapping = do_bootstrapping
        if mode == "test":
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

    def resample_data(self):
        print("bootstrapping...")
        ind_after_bs = bootstrap_index(np.arrange(self.features.shape[0]))
        self.features_chosen = self.features[ind_after_bs]
        self.labels_chosen = self.labels[ind_after_bs]


    def normalize(self, features, high_dims, high_mean, low_mean, high_std, low_std):
        features[:, :high_dims] = np.subtract(features[:, :high_dims], high_mean)
        features[:, :high_dims] = np.divide(features[:, :high_dims], high_std)
        features[:, high_dims:] = np.subtract(features[:, high_dims:], low_mean)        
        features[:, high_dims:] = np.divide(features[:, high_dims:], low_std)

        return features

    def rm_static(self, features, labels):
        self.indx_valid_v = np.abs(features[:, 2]) > self.speed_limit
        features = features[self.indx_valid_v, :]
        labels = labels[self.indx_valid_v]
        return features, labels

    def rm_close(self, features, labels):
        self.indx_valid_close = np.logical_and(features[:, 0] > self.dist_near, features[:, 0] < self.dist_far) 
        features = features[self.indx_valid_close, :]
        labels = labels[self.indx_valid_close]
        return features, labels

    def to_ovo(self, class_positive, class_negative):
        if self.features_all is None or self.labels_all is None:
            raise ValueError
        valid_indx = np.logical_or(self.labels_all == class_positive, self.labels_all == class_negative)
        self.labels = self.labels_all[valid_indx]
        self.features = self.features_all[valid_indx]
        self.labels = (self.labels == class_positive).astype(np.uint8)

    def to_ova(self, chose_class):
        if self.features_all is None or self.labels_all is None:
            raise ValueError
        valid_indx = (self.labels_all == chose_class).astype(np.uint8)
        self.labels = valid_indx
        

    def save_features_labels(self):
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


if __name__ == "__main__":
    ## Test for data loader 
    data_path = "/data/jiaaodong/conti_new/dataset1/data/target_data_1558958116/low"
    batch_size = 32
    to_tensor = ToTensor()
    composed_trans = transforms.Compose([to_tensor])
    target_dataset = TargetModeDataset(
                data_path, composed_trans, 
                mode='train', high_dims = 4, 
                normalize = True, feature_type = 'low',
                norms_path=data_path,
                speed_limit=1,
                dist_near= 1,
                binary_class = False)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    train_iter = iter(target_loader)
    sample = train_iter.next()
    feature = sample['data']
    # print(feature.shape)
    label = sample['label']
    # # print(feature.squeeze().size(), label.size())
    # plt.figure(1)
    # plt.scatter(feature[:,0,1], feature[:,0,0],c=label[:,0])
    # plt.figure(2)
    # plt.imshow(torch.cat([label.double(), feature.squeeze()], 1), interpolation="nearest", aspect='auto')
    # plt.title("First Batch")
    # plt.xticks(np.array([0,1,2,3,4,5,6,7,8]), ["label", "x","y","v","r" ,"rcs" ,"Var" ,"ang" ,"V"])
    # plt.colorbar()
    # plt.show()