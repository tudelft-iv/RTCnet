import sys 
import os 
import os.path as osp 
import numpy as np 
import json 
from sklearn.cluster import DBSCAN
import tqdm
import matplotlib.pyplot as plt 
from scipy.spatial import distance
from copy import deepcopy
from cylinder_cluster import cylinder_cluster
""" 
Do some work on object level
"""


def cal_precision_recall_single(labels_true, labels_pred, instance_id_true, instance_id_pred, class_label=0, targets_for_debug=None):

    """
    Parameters
    ------------- 
    This function deals with single frame precision and recall calculation
    labels_true: 1-d array
        The true label of the targets of one single frame
    labels_pred: 1-d array
        The predicted label of the targets of one single frame
    instance_id_true: 1-d array
        The instance ID of the targets of one single frame from SSD bounding box
    instance_id_pred: 1-d array
        The instance ID of the post-clustering output
    class_label: an int
        The label of the class to calculation. 0 for background, 1 for pedestrian, 2 for cyclist, 3 for car
    
    Returns
    ------------
    num_TP: an int
        The number of true positives of the class_label
    """

    num_TP_in_pred = 0
    num_TP_in_true = 0
    num_instances_pred = 0
    num_instances_true = 0
    mask_class_label_pred = labels_pred == class_label
    mask_class_label_true = labels_true == class_label
    num_instances_pred = np.unique(instance_id_pred[labels_pred==class_label]).shape[0]
    num_instances_true = np.unique(instance_id_true[labels_true==class_label]).shape[0]
    intersection_sum = 0
    union_sum = 0

    if instance_id_pred.shape[0] >0:
        instance_id_pred_max = instance_id_pred.max()
    else:
        instance_id_pred_max = 0
    if instance_id_true.shape[0] >0:
        instance_id_true_max = instance_id_true.max()
    else:
        instance_id_true_max = 0

    intersection_sum = np.sum(np.logical_and(mask_class_label_pred, mask_class_label_true))
    union_sum = np.sum(np.logical_or(mask_class_label_pred, mask_class_label_true))
    num_TP_single_target = 0
    # calculate num_TP_in_pred (for precision)
    for i in np.arange(instance_id_pred_max + 1):
        indx_pred = np.logical_and(instance_id_pred == i, mask_class_label_pred)
        for j in np.arange(instance_id_true_max + 1):
            indx_true = np.logical_and(instance_id_true == j, mask_class_label_true)
            intersection = np.sum(np.logical_and(indx_pred, indx_true))
            union = np.sum(np.logical_or(indx_pred, indx_true))
            if intersection/union >= 0.5:
                num_TP_in_pred += 1
                break

    # calculate num_TP_in_true (for recall)
    for i in np.arange(instance_id_true_max + 1):
        indx_true = np.logical_and(instance_id_true == i, mask_class_label_true)
        for j in np.arange(instance_id_pred_max + 1):
            indx_pred = np.logical_and(instance_id_pred == j, mask_class_label_pred)
            intersection = np.sum(np.logical_and(indx_pred, indx_true))
            union = np.sum(np.logical_or(indx_pred, indx_true))
            num_true = np.sum(indx_true)
            num_pred = np.sum(indx_pred)

            if intersection/union >= 0.5:
                num_TP_in_true += 1
                if np.sum(indx_true) == 1:
                    num_TP_single_target +=1
                break

                

    return num_TP_in_pred, num_TP_in_true, num_instances_pred, num_instances_true, intersection_sum, union_sum, num_TP_single_target
    
def cal_precision_recall_all(labels_true, labels_pred, instance_id_true, instance_id_pred, frame_id, num_class = 4, targets_for_debug=None):
    """ 
    Parameters
    -------------
    labels_true: 1-d array
        The true labels of the targets of all the frames
    labels_pared: 1-d array
        The predicted labels of the targets of all the frames 
    instance_id_true:
        The instance ID of the targets of all frames from SSD bounding box
    instance_id_pred:
        The instance ID of the post-clustering output
    frame_id: 1-d array
        The frame_id of each target. The size of frame_id should be equal to labels_true and labels_pred 

    Returns
    -------------
    precision: 1-d array 
        the precision of each class
    recall: 1-d array
        the recall of each class

    """

    num_TP_in_pred = np.zeros(num_class)
    num_TP_in_true = np.zeros(num_class)
    intersection_sum = np.zeros(num_class)
    union_sum = np.zeros(num_class)
    num_instance_pred = np.zeros(num_class)
    num_instance_true = np.zeros(num_class)
    precision = np.zeros(num_class)
    recall = np.zeros(num_class)
    num_TP_single_target_total = 0
    pbar = tqdm.tqdm(total = frame_id.max()+1, desc='calculate precision and recall')
    for i in np.arange(0, frame_id.max()+1):
        pbar.update()
        # if i<100:
        #     continue
        labels_true_single = labels_true[frame_id==i]
        labels_pred_single = labels_pred[frame_id==i]
        instance_id_pred_single = instance_id_pred[frame_id==i]
        instance_id_true_single = instance_id_true[frame_id==i]
        for j in np.arange(1, num_class):
            num_TP_in_pred_single, num_TP_in_true_single, num_instance_pred_single, num_instance_true_single, intersection_single, union_single, num_TP_single_target = cal_precision_recall_single(labels_true_single, 
                                                                                                            labels_pred_single, 
                                                                                                            instance_id_true_single, 
                                                                                                            instance_id_pred_single, 
                                                                                                            class_label=j)
                                                                                                            # targets_for_debug=targets_for_debug[frame_id==i,:])
            # print("num_TP_in_pred_single",num_TP_in_pred_single)
            # print("num_TP_in_true_single", num_TP_in_true_single)

            num_TP_in_pred[j] += num_TP_in_pred_single
            num_TP_in_true[j] += num_TP_in_true_single
            num_instance_pred[j] += num_instance_pred_single
            num_instance_true[j] += num_instance_true_single
            intersection_sum[j] += intersection_single
            union_sum[j] += union_single
            num_TP_single_target_total += num_TP_single_target
    precision = num_TP_in_pred / num_instance_pred
    recall    = num_TP_in_true / num_instance_true

    return precision, recall, intersection_sum/union_sum, num_TP_single_target_total

eps_xy_list = [None, 0.5, 2 , 4]
eps_v_list = [None, 2, 1.6, 1]
min_targets_list = [None, 1, 2, 3]
def post_clustering(targets_xyv, labels_pred, frame_id, DBSCAN_eps = 1.1, DBSCAN_min_samples=1, algorithm=1, target_scores = None, filter_objects = False):
    """  
    Parameters:
    ------------------
    targets_xyv: 2-d array
        The x, y coordinates and velocity of targets
    labels_pred: 1-d array
        The predicted labels of the targets of all the frames 
    frame_id: 1-d array 
        The frame_id of each target. The size of frame_id should be equal to labels_pred and targets_rav
    """
    color_LUT = np.array(['c','g','r','b'])
    min_scores_list = []
    post_clst_id = -1 * np.ones(targets_xyv.shape[0])
    pbar = tqdm.tqdm(total = frame_id.max()+1, desc='post_clustering')
    t = 0
    debug_mode = False
    filter_objects = True
    for i in np.arange(0, frame_id.max() + 1):
        pbar.update()
        if debug_mode and i < 3803:
            continue
        targets_xyv_single = targets_xyv[frame_id==i, :]
        labels_pred_single = labels_pred[frame_id==i]
        targets_xyv_ped = targets_xyv_single[labels_pred_single==1, :]
        targets_xyv_biker = targets_xyv_single[labels_pred_single==2, :]
        targets_xyv_car = targets_xyv_single[labels_pred_single==3, :]
        targets_scores_single = target_scores[frame_id == i, :] / np.reshape(np.linalg.norm(target_scores[frame_id == i, :] , ord=2, axis=1), (-1, 1))
        # first time
        if targets_xyv_ped.shape[0] > 0:
            post_clst_id_ped = cylinder_cluster(targets_xyv_ped[:, :3], eps_xy = eps_xy_list[1], eps_v = eps_v_list[1], min_targets=min_targets_list[1])
            max_clst_id_ped = post_clst_id_ped.max()
        else:
            post_clst_id_ped = np.array([])
            max_clst_id_ped = -1
        if targets_xyv_biker.shape[0] > 1:
            post_clst_id_biker = cylinder_cluster(targets_xyv_biker[:, :3], eps_xy = eps_xy_list[2], eps_v = eps_v_list[2], min_targets=min_targets_list[2])
                
            max_clst_id_biker = max_clst_id_ped + 1 + post_clst_id_biker.max()
        else:
            post_clst_id_biker = -1 * np.ones([targets_xyv_biker.shape[0]])
            max_clst_id_biker = max_clst_id_ped 
        if targets_xyv_car.shape[0] > 2:
            post_clst_id_car = cylinder_cluster(targets_xyv_car[:, :3], eps_xy = eps_xy_list[3], eps_v = eps_v_list[3], min_targets=min_targets_list[3])
            
        else:
            post_clst_id_car = -1 * np.ones([targets_xyv_car.shape[0]])

        post_clst_id_biker[post_clst_id_biker>-1] = post_clst_id_biker[post_clst_id_biker>-1] + max_clst_id_ped+1
        post_clst_id_car[post_clst_id_car>-1] = post_clst_id_car[post_clst_id_car>-1] + max_clst_id_biker + 1

        post_clst_id_single = -1 * np.ones(np.sum(frame_id == i))
        post_clst_id_single[labels_pred_single==1] = post_clst_id_ped 
        post_clst_id_single[labels_pred_single==2] = post_clst_id_biker
        post_clst_id_single[labels_pred_single==3] = post_clst_id_car
        labels_pred_single[post_clst_id_single==-1] = 0
        if debug_mode:
            plt.figure(figsize = (16, 16))
            plt.title("Frame:{}".format(i))
            ax1 = plt.subplot(221)
            ax1.set_title("cluster before refinement")
            sc1 = ax1.scatter(targets_xyv_single[:, 1], targets_xyv_single[:, 0], c = post_clst_id_single, s = 10*post_clst_id_single+7)
            plt.colorbar(sc1, ax=ax1)
            ax1.set_xlim([-25, 25])
            ax1.set_ylim([0, 40])
            ax2 = plt.subplot(222)
            ax2.set_title("labels before refinement")
            ax2.scatter(targets_xyv_single[:, 1], targets_xyv_single[:, 0], c = color_LUT[labels_pred_single])
            ax2.set_xlim([-25, 25])
            ax2.set_ylim([0, 40])

        space_threshold = 1
        speed_threshold = [0, 3, 2, 1.2]
        score_threshold = 0.6

        if post_clst_id_single.shape[0] > 0 and filter_objects:
            min_dist_mat = 10 * np.ones([int(post_clst_id_single.max() + 1), int(post_clst_id_single.max() + 1)])
            min_v_diff_mat = 10 * np.ones([int(post_clst_id_single.max() + 1), int(post_clst_id_single.max() + 1)])
            object_label_list = 5 * np.ones(int(post_clst_id_single.max() + 1))
            for k in np.arange(int(post_clst_id_single.max() + 1)):
                for l in np.arange(int(post_clst_id_single.max() + 1)):
                    if k!=l and np.sum(post_clst_id_single == k) == 0 or np.sum(post_clst_id_single == l) == 0:
                        continue
                    object_label_list[k] = labels_pred_single[post_clst_id_single==k][0]
                    min_dist_pair = distance.cdist(targets_xyv_single[post_clst_id_single == k, :2], targets_xyv_single[post_clst_id_single == l, :2]).min()
                    min_dist_mat[k, l] = min_dist_pair 
                    min_v_diff_pair = distance.cdist(np.reshape(targets_xyv_single[post_clst_id_single == k, 2], (-1, 1)), np.reshape(targets_xyv_single[post_clst_id_single == l, 2], (-1, 1))).min()
                    min_v_diff_mat[k, l] = min_v_diff_pair
                    min_score_diff_pair = distance.cdist(targets_scores_single[post_clst_id_single == k, :], targets_scores_single[post_clst_id_single == l, :]).min()
                    if min_dist_pair < space_threshold:
                        label1 = labels_pred_single[post_clst_id_single == k][0] 
                        label2 = labels_pred_single[post_clst_id_single == l][0]
                        if (label1 == 2 and label2 == 3) or (label1 == 3 and label2 == 2):
                            min_scores_list.append(min_score_diff_pair)
                            # print(min_scores_list)
                            if min_v_diff_pair < speed_threshold[3]:
                                num_targets1 = np.sum(post_clst_id_single == k)
                                num_targets2 = np.sum(post_clst_id_single == l)
                                if num_targets1 > num_targets2:
                                    label_refine = label1
                                else:
                                    label_refine = label2 
                                if label_refine == 3 and min_score_diff_pair < score_threshold:
                                    post_clst_id_single[post_clst_id_single == k] = l
                                    labels_pred_single[post_clst_id_single == k] = label_refine
                                    labels_pred_single[post_clst_id_single == l] = label_refine
                        elif (label1 == 1 and label2 == 3) or (label1 == 3 and label2 == 1):
                            if min_v_diff_pair < speed_threshold[3]:
                                num_targets1 = np.sum(post_clst_id_single == k)
                                num_targets2 = np.sum(post_clst_id_single == l)
                                if num_targets1 > num_targets2:
                                    label_refine = label1
                                else:
                                    label_refine = label2 
                                if label_refine == 3  and min_score_diff_pair < score_threshold:
                                    post_clst_id_single[post_clst_id_single == k] = l
                                    labels_pred_single[post_clst_id_single == k] = label_refine
                                    labels_pred_single[post_clst_id_single == l] = label_refine
                        elif (label1 == 1 and label2 == 2) or (label1 == 2 and label2 == 1):
                            if min_v_diff_pair < speed_threshold[2]:
                                num_targets1 = np.sum(post_clst_id_single == k)
                                num_targets2 = np.sum(post_clst_id_single == l)
                                if num_targets1 > num_targets2:
                                    label_refine = label1
                                else:
                                    label_refine = label2 
                                if label_refine == 2 and min_score_diff_pair < score_threshold:
                                    post_clst_id_single[post_clst_id_single == k] = l
                                    labels_pred_single[post_clst_id_single == k] = label_refine
                                    labels_pred_single[post_clst_id_single == l] = label_refine
            num_obj_around = np.sum(np.logical_and(min_dist_mat < space_threshold, min_v_diff_mat < speed_threshold[3]), axis = 1)
            id_list_of_cars_surrounded_by_a_lot_of_bikers = np.nonzero(np.logical_and(num_obj_around>2, object_label_list == 3))
            for id_of_cars_surrounded_by_a_lot_of_bikers in id_list_of_cars_surrounded_by_a_lot_of_bikers:
                labels_pred_single[post_clst_id_single == id_of_cars_surrounded_by_a_lot_of_bikers] = 2

        # second time                     
        targets_xyv_ped = targets_xyv_single[labels_pred_single==1, :]
        targets_xyv_biker = targets_xyv_single[labels_pred_single==2, :]
        targets_xyv_car = targets_xyv_single[labels_pred_single==3, :]
        if targets_xyv_ped.shape[0] > 0:
            post_clst_id_ped = cylinder_cluster(targets_xyv_ped[:, :3], eps_xy = eps_xy_list[1], eps_v = eps_v_list[1], min_targets=min_targets_list[1])
            max_clst_id_ped = post_clst_id_ped.max()
        else:
            post_clst_id_ped = -1 * np.ones([targets_xyv_ped.shape[0]])
            max_clst_id_ped = -1
        if targets_xyv_biker.shape[0] > 1:
            post_clst_id_biker = cylinder_cluster(targets_xyv_biker[:, :3], eps_xy = eps_xy_list[2], eps_v = eps_v_list[2], min_targets=min_targets_list[2] )
            max_clst_id_biker = max_clst_id_ped + 1 + post_clst_id_biker.max()
        else:
            post_clst_id_biker = -1 * np.ones([targets_xyv_biker.shape[0]])
            max_clst_id_biker = max_clst_id_ped 
        if targets_xyv_car.shape[0] > 2:
            post_clst_id_car = cylinder_cluster(targets_xyv_car[:, :3], eps_xy = eps_xy_list[3], eps_v = eps_v_list[3], min_targets=min_targets_list[3])
            
        else:
            post_clst_id_car = -1 * np.ones([targets_xyv_car.shape[0]])

        post_clst_id_biker[post_clst_id_biker>-1] = post_clst_id_biker[post_clst_id_biker>-1] + max_clst_id_ped+1
        post_clst_id_car[post_clst_id_car>-1] = post_clst_id_car[post_clst_id_car>-1] + max_clst_id_biker + 1

        post_clst_id_single = -1 * np.ones(np.sum(frame_id == i))
        post_clst_id_single[labels_pred_single==1] = post_clst_id_ped 
        post_clst_id_single[labels_pred_single==2] = post_clst_id_biker
        post_clst_id_single[labels_pred_single==3] = post_clst_id_car
        labels_pred_single[post_clst_id_single==-1] = 0

        post_clst_id[frame_id==i] = post_clst_id_single
        labels_pred[frame_id==i] = labels_pred_single

    min_scores_list = np.array(min_scores_list)
    return post_clst_id, labels_pred 
def cal_f1(precision, recall):

    return 2*precision*recall/(precision+recall)

if __name__ == "__main__":
    """ Test the post_clustering->precision & recall pipeline """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    result_DIR = osp.join(BASE_DIR, 'results', 'RTCtrain_info')
    config_list = os.listdir(result_DIR)
    config_list.sort()
    RTCnet_result_info_path = osp.join(result_DIR, config_list[-1])
    speed_threshold = 0.3
    speed_threshold_to_change_label = 0
    show_RTC_result = True
    with open(RTCnet_result_info_path, 'r') as f:
        RTCnet_result_info = json.load(f)


    if show_RTC_result:
        result_path = RTCnet_result_info["result_folder"]

        data_path = RTCnet_result_info["data_path"]
        labels_pred = np.load(osp.join(result_path, "pred_labels_test.npy"))
        labels_true = np.load(osp.join(result_path, "true_labels_test.npy"))
        target_scores = np.load(osp.join(result_path, "final_score.npy"))
        frame_id = np.load(osp.join(data_path, "test", "frame_id.npy"))
        instance_id_true = np.load(osp.join(data_path, "instance_id_test.npy"))
        targets_rav = np.load(osp.join(data_path, "test", "features.npy"))[:, :3]


        targets_xyv = np.zeros(targets_rav.shape)
        targets_xyv[:,0] = targets_rav[:,0]*np.cos(targets_rav[:,1])
        targets_xyv[:,1] = targets_rav[:,0]*np.sin(targets_rav[:,1])
        targets_xyv[:,2] = targets_rav[:,2]
        targets_v = np.abs(targets_rav[:, 2])
        labels_pred[targets_v < speed_threshold_to_change_label] = 0
        labels_pred = labels_pred[targets_v > speed_threshold]
        labels_true = labels_true[targets_v > speed_threshold]
        target_scores = target_scores[targets_v > speed_threshold]
        frame_id = frame_id[targets_v > speed_threshold]
        targets_rav = targets_rav[targets_v>speed_threshold, :]
        targets_xyv = targets_xyv[targets_v > speed_threshold, :]
        
        instance_id_true = instance_id_true[targets_v > speed_threshold]
        instance_id_pred, labels_pred = post_clustering(targets_xyv, labels_pred, frame_id, target_scores = target_scores)
        precision_RTC, recall_RTC, IoU_RTC, num_TP_single_target_total = cal_precision_recall_all(labels_true, labels_pred, instance_id_true, instance_id_pred, frame_id, targets_for_debug=targets_xyv)
 
        print("precision_RTC", precision_RTC, "recall_RTC", recall_RTC)
        np.savetxt(osp.join(result_path, "precision_RTC.csv"), precision_RTC, delimiter=",")
        np.savetxt(osp.join(result_path, "recall_RTC.csv"), recall_RTC, delimiter=",")
        np.savetxt(osp.join(result_path, "F1score.csv"), cal_f1(precision_RTC, recall_RTC), delimiter=",")
        np.savetxt(osp.join(result_path, "IoU_RTC.csv"), IoU_RTC, delimiter=",")
        np.save(osp.join(result_path, "pred_labels_refine.npy"), labels_pred)
        print("number of TP object with 1 target:", num_TP_single_target_total)
        print("f1 RTC: ", cal_f1(precision_RTC, recall_RTC))
        print("IoU RTC:", IoU_RTC)
