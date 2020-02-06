
# Python Libraries
import os 
import sys 
import os.path as osp 
import argparse

# Pytorch
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn 

# Third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import json 

# Modules from current project
from TargetLoader import TargetModeDataset, ToTensor
from RTCnet import RTCnetV4
from RTCnet_utils import Tester, Tester_ensemble
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
result_DIR = osp.join(BASE_DIR, osp.pardir, 'results', 'RTCtrain_info')
config_list = os.listdir(result_DIR)
config_list.sort()

parser = argparse.ArgumentParser()
parser.add_argument("-config", default = osp.join(result_DIR, config_list[-1]), type = str, help=" The default configuration file is the first configuration file in results folder")
parser.add_argument("-test", default = None, type = str)
parser.add_argument("-batch_size", type=int, default = 1024, help = "The version of networks. 1: RTC 2: RTC2 3: Multi-layer perceptron for only high-level" ) 
args = parser.parse_args()



def test_ova(
    chose_class,
    result_folder,
    model,
    test_data,
    weights_all,
    test_loader,
    use_gpu
):

    class_list = ['others', 'ped', 'biker', 'car']
    print("start training {} vs All".format(class_list[chose_class]))
    append_str = "{}_vs_All".format(class_list[chose_class])
    model_path      = osp.join(result_folder,'best_checkerpoint_{}.pth'.format(append_str))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    weights = torch.tensor(np.array([np.sum(weights_all) - weights_all[1], weights_all[1]])) if not use_gpu else \
         torch.tensor(np.array([np.sum(weights_all) - weights_all[1], weights_all[1]])).cuda()
    loss_func = nn.CrossEntropyLoss(weights)

    tester = Tester_ensemble(
        model,
        loss_func = loss_func,
        use_gpu=use_gpu
    )
    scores_ova = tester.test(test_loader)
    return scores_ova

def test_ovo(
    class_positive,
    class_negative,
    result_folder,
    model,
    test_data,
    weights_all,
    test_loader,
    use_gpu,
):

    class_list = ['others', 'ped', 'biker', 'car']
    print("start training {} vs {}".format(class_list[class_positive], class_list[class_negative]))
    append_str = "{}_vs_{}".format(class_list[class_positive], class_list[class_negative])
    weights = torch.tensor(np.array([weights_all[class_negative], weights_all[class_positive]]))
    if use_gpu:
        weights = weights.cuda()
    print("weights:{}".format(weights))
    loss_func = nn.CrossEntropyLoss(weight = weights)

    model_path      = osp.join(result_folder,'best_checkerpoint_{}.pth'.format(append_str))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tester = Tester_ensemble(
        model,
        loss_func = loss_func,
        use_gpu=use_gpu
    )
    scores_ovo = tester.test(test_loader)
    return scores_ovo


    

def main():
    cfg_path        = args.config
    test_data_path  = args.test

    with open(cfg_path) as fp:
        cfg = json.load(fp)
    data_path       = cfg["data_path"]
    use_gpu         = cfg['use_gpu']
    result_folder   = cfg['result_folder']
    weights_all     = cfg['weights']
    use_set         = ["train", "val", "test"]
    weights_factor  = cfg["weights_factor"]
    dropout         = cfg['dropout']
    weights_factor  = np.array(weights_factor)
    weights_all     = np.array(weights_all)
    binary_class    = cfg['binary']
    input_size      = cfg['input_size']
    rm_cars         = cfg["rm_cars"]
    speed_limit     = cfg["speed_limit"]
    dist_near       = cfg["dist_limit"]
    dist_far        = cfg["data_x_limit"]
    if "rm_speed" in cfg:
        rm_speed = cfg["rm_speed"]
    else:
        rm_speed = False
    if "rm_rcs" in cfg:
        rm_rcs = cfg["rm_rcs"]
    else:
        rm_rcs = False
    test_with_nearby = False
    only_slow = False
    only_fast = False
    if "rm_position" in cfg:
        rm_position = cfg['rm_position']
    else:
        rm_position = False
    
    use_nearby = False
    if use_nearby or test_with_nearby:
        dist_far = 60
    if rm_position:
        high_dims = 2
        dist_far = 100
    else:
        high_dims = 4
    if test_data_path is None:
        test_data_path = data_path
    if rm_cars:
        num_classes = 3
    elif binary_class:
        num_classes = 2
    else:
        num_classes = 2
    weights_all = np.multiply(weights_factor, weights_all)
    feature_type = "low"
    cfg['testset'] = test_data_path
    with open(osp.join(result_folder, 'info.json'), 'w') as fp:
        json.dump(cfg, fp, sort_keys=True, indent=4, separators=(',', ': '))
    
    batch_size = args.batch_size

    ################### transforms ###################
    to_tensor = ToTensor()
    composed_trans = transforms.Compose([to_tensor])

    test_data = TargetModeDataset(
                test_data_path, composed_trans, 
                mode='test', high_dims=high_dims, 
                normalize=True, feature_type= feature_type,
                norms_path=result_folder,
                speed_limit=speed_limit,
                dist_near=dist_near,
                binary_class=binary_class,
                dist_far=dist_far,
                rm_cars=rm_cars,
                rm_position=rm_position,
                zero_low=False,
                only_slow=only_slow,
                only_fast=only_fast,
                rm_speed=rm_speed,
                rm_rcs=rm_rcs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)
    np.save(osp.join(result_folder, "valid_indx_test"), test_data.indx_valid)

    ################### Define model ###################
    model = RTCnetV4(
            num_classes=2, 
            Doppler_dims=32, 
            high_dims = high_dims, 
            dropout= dropout,
            input_size = input_size)

    ################### Ped VS ALL ###########
    scores_Ped_vs_All = test_ova(
                        1,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    scores_Biker_vs_All = test_ova(
                        2,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    scores_Car_vs_All = test_ova(
                        3,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    scores_Others_vs_All = test_ova(
                        0,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    scores_Ped_vs_Biker = test_ovo(
                        1,
                        2,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    scores_Ped_vs_Car = test_ovo(
                        1,
                        3,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )
    scores_Biker_vs_Car = test_ovo(
                        2,
                        3,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    scores_Others_vs_Ped = test_ovo(
                        0,
                        1,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    scores_Others_vs_Biker = test_ovo(
                        0,
                        2,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    scores_Others_vs_Car = test_ovo(
                        0,
                        3,
                        result_folder,
                        model,
                        test_data,
                        weights_all,
                        test_loader,
                        use_gpu
                    )

    

    final_score1 = np.reshape(scores_Ped_vs_Biker * (scores_Ped_vs_All + scores_Biker_vs_All) \
                    + scores_Ped_vs_Car * (scores_Ped_vs_All + scores_Car_vs_All) \
                    + (1 - scores_Others_vs_Ped) * (scores_Ped_vs_All + scores_Others_vs_All), (-1, 1))
    final_score2 = np.reshape(( 1 - scores_Ped_vs_Biker) * (scores_Biker_vs_All + scores_Ped_vs_All)\
                    + scores_Biker_vs_Car * (scores_Biker_vs_All + scores_Car_vs_All)\
                    + (1 - scores_Others_vs_Biker) * (scores_Biker_vs_All + scores_Others_vs_All), (-1, 1))
    final_score3 = np.reshape((1 - scores_Ped_vs_Car) * (scores_Car_vs_All + scores_Ped_vs_All)\
                    + (1-scores_Biker_vs_Car) * (scores_Car_vs_All + scores_Biker_vs_All)\
                    + (1-scores_Others_vs_Car) * (scores_Car_vs_All + scores_Others_vs_All), (-1, 1))
    final_score0 = np.reshape(scores_Others_vs_Ped * (scores_Others_vs_All + scores_Ped_vs_All)\
                    + scores_Others_vs_Biker * (scores_Others_vs_All + scores_Biker_vs_All)\
                    + scores_Others_vs_Car * (scores_Others_vs_All + scores_Car_vs_All), (-1, 1))
    final_score = np.concatenate([final_score0, final_score1, final_score2, final_score3], axis=1)
    pred_labels_all = np.argmax(final_score, axis=1)
    true_labels_all = test_data.labels

    np.save(osp.join(result_folder, "final_score.npy"), final_score)
    np.save(osp.join(result_folder, "true_labels_test.npy"), true_labels_all)


    np.save(osp.join(result_folder, "pred_labels_{}".format("test")), pred_labels_all)
    ################### Confusion matrix ######################
    class_names = np.array(['Others', 'Ped.', 'Biker', 'Car'])
    confusion_mat = confusion_matrix(true_labels_all, pred_labels_all)
    f1_score_all = f1_score(true_labels_all, pred_labels_all, average='macro')
    f1_score_individual = f1_score(true_labels_all, 
                            pred_labels_all, average=None)
    with open(osp.join(result_folder, "f1score_{}.txt".format("test")),'a') as f:
        np.savetxt(f,np.reshape(f1_score_all, (1,-1)),fmt='%f')
        np.savetxt(f,np.reshape(f1_score_individual, (1,-1)),fmt='%f')
    np.savetxt(osp.join(result_folder, '{}.txt'.format("test")),confusion_mat,fmt='%f')
    np.save(osp.join('{}.npy'.format("test")),confusion_mat)
if __name__ == "__main__":
    main()