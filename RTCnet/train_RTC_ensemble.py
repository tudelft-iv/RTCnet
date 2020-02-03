# Python Library
import os
import os.path as osp 
import sys 
from time import time
import json
import argparse
from copy import deepcopy
from datetime import datetime

# Third-party library
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# Pytorch
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

# Network 
from RTCnet import RTCnetV4
from RTCnet_utils import Trainer
from TargetLoader import TargetModeDataset, ToTensor, Permutation

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Base directory of the RTC module


def train_ova(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            chose_class 
            ):
    """ 
        Function to train one-over-all model
    Args:
        train_data (TargetModeDataset): The training dataset
        val_data (TargetModeDataset): The validation dataset
        train_loader (DataLoader): The dataloader for training dataset
        val_loader (DataLoader): The dataloader for validation dataset
        weights_all (NumpyArray): Normalized weights for classes
        model (RTCnetV4): The network modules
        optimizer (Optimizer): The optimizer in torch.optim
        scheduler: The scheduler for training
        eval_frequency (int): The frequency used for evaluation by validation set. 0 means evaluating after each epoch
        use_gpu (bool): Whether using GPU for training and validation
        lr_decay_step (int): the number of steps for learning rate decay
        decay_f (double): the decay factor of learning rate
        lr_clip (double): the clip of learning rate during decay process
        result_folder (string): the folder to save the training result
        n_epochs (int): the number of epochs for training
        chose_class (int): the class number for the chosen class  

    """
    class_list = ['others', 'ped', 'biker', 'car']
    print("start training {} vs All".format(class_list[chose_class]))
    append_str = "{}_vs_All".format(class_list[chose_class])
    weights = torch.tensor(np.array([np.sum(weights_all)-weights_all[chose_class], weights_all[chose_class]]))
    if use_gpu:
        weights = weights.cuda()
    print("weights:{}".format(weights))
    loss_func = nn.CrossEntropyLoss(weight = weights)
    #################### data ##########################
    train_data.to_ova(chose_class = chose_class)
    val_data.to_ova(chose_class = chose_class)
    ################### Trainer ##########################
    trainer = Trainer(
                model, 
                loss_func,
                optimizer,
                lr_scheduler = scheduler,
                eval_frequency = eval_frequency,
                use_gpu = use_gpu,
                lr_decay_step=lr_decay_step,
                lr_decay_f=decay_f,
                lr_clip=lr_clip,
                save_checkerpoint_to=result_folder,
                append_str= append_str
    )
    trainer.train(
            n_epochs,
            train_loader,
            val_loader=val_loader,
            best_loss=1e5,
            start_it=0
    )
    loss_trajectory = trainer.trace_loss
    loss_trajectory_train = trainer.trace_loss_train
    np.save(osp.join(result_folder, "loss_trajectory" + append_str), loss_trajectory)
    np.save(osp.join(result_folder, "train_loss_trajectory" + append_str), loss_trajectory_train)

def train_ovo(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            class_positive,
            class_negative
            ):
    """ 
        Function to train one-over-one model

    Args:
        train_data (TargetModeDataset): The training dataset
        val_data (TargetModeDataset): The validation dataset
        train_loader (DataLoader): The dataloader for training dataset
        val_loader (DataLoader): The dataloader for validation dataset
        weights_all (NumpyArray): Normalized weights for classes
        model (RTCnetV4): The network modules
        optimizer (Optimizer): The optimizer in torch.optim
        scheduler: The scheduler for training
        eval_frequency (int): The frequency used for evaluation by validation set. 0 means evaluating after each epoch
        use_gpu (bool): Whether using GPU for training and validation
        lr_decay_step (int): the number of steps for learning rate decay
        decay_f (double): the decay factor of learning rate
        lr_clip (double): the clip of learning rate during decay process
        result_folder (string): the folder to save the training result
        n_epochs (int): the number of epochs for training
        class_positive(int): the class number for the positive class
        class_negative(int): the class number for the negative class

    """
    class_list = ['others', 'ped', 'biker', 'car']
    print("start training {} vs {}".format(class_list[class_positive], class_list[class_negative]))
    append_str = "{}_vs_{}".format(class_list[class_positive], class_list[class_negative])
    weights = torch.tensor(np.array([weights_all[class_negative], weights_all[class_positive]]))
    if use_gpu:
        weights = weights.cuda()
    print("weights:{}".format(weights))
    loss_func = nn.CrossEntropyLoss(weight = weights)
    #################### data ##########################
    train_data.to_ovo(class_positive, class_negative)
    val_data.to_ovo(class_positive, class_negative)
    ################### Trainer ##########################
    trainer = Trainer(
                model, 
                loss_func,
                optimizer,
                lr_scheduler = scheduler,
                eval_frequency = eval_frequency,
                use_gpu = use_gpu,
                lr_decay_step=lr_decay_step,
                lr_decay_f=decay_f,
                lr_clip=lr_clip,
                save_checkerpoint_to=result_folder,
                append_str= append_str
    )
    trainer.train(
            n_epochs,
            train_loader,
            val_loader=val_loader,
            best_loss=1e5,
            start_it=0
    )
    loss_trajectory = trainer.trace_loss
    np.save(osp.join(result_folder, "loss_trajectory" + append_str), loss_trajectory)
    loss_trajectory_train = trainer.trace_loss_train
    np.save(osp.join(result_folder, "train_loss_trajectory" + append_str), loss_trajectory_train)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default = osp.join(BASE_DIR, osp.pardir, 'dataset','data'), help = "The data path that contains train, val and test folder, this is generated by gen_input_data_and_baseline.py" )
    parser.add_argument("--batch_size", type=int, default = 1024, help = "The version of networks. 1: RTC 2: RTC2 3: Multi-layer perceptron for only high-level" ) 
    parser.add_argument("--rm_speed", type=bool, default = False, help = "Whether to remove the speed during training" )
    parser.add_argument("--rm_rcs", type=bool, default = False, help = "Whether to remove the RCS value during training" ) 
    parser.add_argument("--n_epochs", type=int, default = 1, help = "number of epochs" ) 
    args            = parser.parse_args()
    data_path       = args.data
    rm_speed        = args.rm_speed
    rm_rcs          = args.rm_rcs 
    n_epochs        = args.n_epochs
    model_version   = 4

    # Get the meta information of the dataset
    data_meta_dir   = osp.join(data_path, os.pardir)
    data_meta_path  = osp.join(data_meta_dir, 'meta_data.json')
    with open(data_meta_path) as fp:
        data_meta = json.load(fp)
    data_x_lim      = data_meta['x_lim']                    # The maximum limit for the longitudinal distance

    # Training setup
    lr_start        = 1e-3                                  # The leraning rate starts from lr_start, but it decays according to certain policies
    eval_frequency  = 0                                     # The frequency used for evaluation by validation set. 0 means evaluating after each epoch
    batch_size      = args.batch_size                       # The batch size for stochastic gradient descent
    lr_decay_step   = 2000                                  # The step after which the learning rate is decayed. But if the optimizer is used, this is useless.
    decay_f         = 0.9                                   # The decay factor for the learning rate. After each lr_decay_step iterations, the learning rate is equal to decay_f * last leraning rate
    lr_clip         = 2e-4                                  # The clip of learning rate. The learning rate stops decaying after this value
    use_gpu         = True                                  # Whether use gpu to train
    dropout         = True
    if_shuffle      = True                                  # Whether to shuffle the data when loading them. This should always be true when training
    speed_limit     = 0                                     # The speed limit used when loading data on the fly. But if the data is already filtered when generated, this can be set as 0
    high_dims       = 4                                     # The number of dimensions for high-level features
    dist_near       = 0                                     # The nearest distance threshold. If using polar coordinate, this is range rather than distance
    binary_class    = False                                 # If using binary class
    use_weight      = True                                  # If using weights for loss function
    weights_factor  = np.array([1/0.5, 1/1, 1/0.5, 1/1])    # The scaling factor for the weights, considering unbalanced classes
    rm_cars         = False                                 # Whether to remove cars when training
    input_size      = 5                                     # The input size of RTC2 window
    rm_position     = False                                 # Whether to remove position during training

    t = int(time())
    result_parent_path = osp.join(BASE_DIR, osp.pardir, 'results')

    result_folder = osp.join(result_parent_path,"RTCresults", str(t))
    train_info_path = osp.join(result_parent_path, 'RTCtrain_info')

    if not osp.exists(train_info_path):
        os.makedirs(train_info_path)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    ################### transforms ###################
    to_tensor = ToTensor()
    perm_position = Permutation()
    composed_trans = transforms.Compose([perm_position, to_tensor])

    ################### data loader ###################  
    ## Train dataset 
    train_data = TargetModeDataset(
                data_path, composed_trans, 
                mode='train', high_dims = high_dims, 
                normalize = True, feature_type = 'low',
                norms_path=result_folder,
                speed_limit=speed_limit,
                dist_near= dist_near,
                binary_class = binary_class,
                dist_far=data_x_lim,
                rm_cars=rm_cars,
                rm_position=rm_position,
                rm_speed = rm_speed,
                rm_rcs=rm_rcs)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=if_shuffle, num_workers=2)
    ## Validation dataset
    val_data = TargetModeDataset(
                data_path, composed_trans, 
                mode='val', high_dims=high_dims, 
                normalize=True, feature_type='low',
                norms_path=result_folder,
                speed_limit=speed_limit,
                dist_near= dist_near,
                binary_class = binary_class,
                dist_far=data_x_lim,
                rm_cars=rm_cars,
                rm_position=rm_position,
                rm_speed = rm_speed,
                rm_rcs=rm_rcs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    train_data.save_features_labels()
    val_data.save_features_labels()

    ################### Model ################### 

    model = RTCnetV4(
            num_classes=2, 
            Doppler_dims=32, 
            high_dims = high_dims, 
            dropout= dropout,
            input_size = input_size)

    model.double()


    ################### weight initialization ################### 
    if use_weight:
        num_others = np.sum(train_data.labels==0)
        num_ped = np.sum(train_data.labels == 1)
        num_bikers = np.sum(train_data.labels==2)
        num_car = np.sum(train_data.labels==3)
        print("num_others:{}, num_peds:{}, num_bikers:{}, num_car:{}".format(num_others, num_ped, num_bikers, num_car))

        weights = np.array([1/num_others,1/num_ped,1/num_bikers,1/num_car])
    else:
        weights = np.ones(4)

    weights = np.multiply(weights_factor, weights)
    weights_save = deepcopy(weights.tolist())
    weights_all = np.divide(weights, np.sum(weights))

    ################### Save the training information #############################
    train_info = {
    "data_path": data_path, 
    "lr_start": lr_start,
    "n_epochs": n_epochs,
    "eval_frequency": eval_frequency,
    "batch_size": batch_size,
    "lr_decay_step":lr_decay_step,
    "decay_f":decay_f,
    "lr_clip":lr_clip,
    "use_gpu":use_gpu,
    "t":t,
    "result_folder":result_folder,
    "weights": weights_save,
    "weights_factor": weights_factor.tolist(),
    "if_shuffle":if_shuffle,
    "dropout":dropout,
    "speed_limit": speed_limit,
    "dist_limit": dist_near,
    "binary": binary_class,
    "input_size": input_size,
    "model_version": model_version,
    "data_x_limit": data_x_lim,
    "use_weight":use_weight,
    "rm_cars": rm_cars,
    "rm_position": rm_position,
    "rm_speed": rm_speed,
    "rm_rcs": rm_rcs
    }
    with open(osp.join(train_info_path, '{}.json'.format(t)), 'w') as fp:
        print("save the file at :{}".format(osp.join(train_info_path, '{}.json'.format(t))))
        json.dump(train_info, fp, sort_keys=True, indent=4, separators=(',', ': '))




    ################### Optimizer ########################
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=lr_decay_step,gamma=0.1)
    
    ################### Ped vs ALL ########################
    
    train_ova(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            chose_class = 1 
            )

    ################### Biker VS ALL ######################
    train_ova(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            chose_class = 2
    ) 
    ################### Car VS ALL ######################
    train_ova(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            chose_class = 3
    ) 
    ################### others VS ALL ######################

    train_ova(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            chose_class = 0
    ) 

    ################### Ped VS biker ########################
    train_ovo(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            1,
            2
            )

    #################### Ped VS car #########################
    train_ovo(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            1,
            3
            )
    ################### Biker VS car #######################
    train_ovo(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            2,
            3
            )
    #################### Others VS ped #####################
    train_ovo(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            0,
            1
            )
    #################### Others VS biker #####################
    train_ovo(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            0,
            2
            )
    ##################### Others VS car #######################
    train_ovo(train_data, 
            val_data, 
            train_loader,
            val_loader,
            weights_all, 
            model, 
            optimizer, 
            scheduler, 
            eval_frequency, 
            use_gpu, 
            lr_decay_step, 
            decay_f, 
            lr_clip, 
            result_folder,
            n_epochs,
            0,
            3
            )



if __name__ == "__main__":
    main()
    