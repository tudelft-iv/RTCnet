import os
import os.path as osp 
import json

import numpy as np 

def create_path():
    """
    Create the path for the dataset.
    
    Parameters
    ----------
    None
    
    Returnes
    --------
    dataset_DIR: string
                 The path to the dataset folder
    data_DIR: string
              The path to the data parent folder
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_DIR = osp.join(BASE_DIR, 'dataset')
    data_DIR = osp.join(dataset_DIR, 'data')
    train_DIR = osp.join(data_DIR, 'train')
    test_DIR = osp.join(data_DIR, 'test')
    val_DIR = osp.join(data_DIR, 'val')
    if not osp.exists(train_DIR):
        os.makedirs(train_DIR)
    if not osp.exists(test_DIR):
        os.makedirs(test_DIR)
    if not osp.exists(val_DIR):
        os.makedirs(val_DIR)
    return dataset_DIR, data_DIR


def generate_dataset_single(num_features, num_samples):
    
    """
    Generate each single dataset
    
    Parameters
    ----------
    num_features: int
                  The number of features for each sample
    num_samples: int
                 The number of samples for the dataset
    """
    features_x = 120 * np.random.rand(num_samples,1) 
    features_y = 100 * np.random.rand(num_samples,1) - 50
    features_v = 30 * np.random.rand(num_samples,1) - 15
    features_rcs = 5 * np.random.rand(num_samples,1) 
    features_low_level = 3000 * np.random.rand(num_samples, 800)
    features = np.concatenate([features_x, features_y, features_v, features_rcs, features_low_level], axis=1)   
    labels = np.random.randint(4,size=(num_samples)) 
    num_targets_per_frame = 20
    num_frames = num_samples / num_targets_per_frame
    frame_id = np.arange(num_frames + 1)
    frame_id = np.repeat(frame_id, num_targets_per_frame)
    frame_id = frame_id[:num_samples]
    instance_id = labels = np.random.randint(6,size=(num_samples)) 

    return features, labels, frame_id, instance_id


if __name__ == "__main__":
    dataset_DIR, data_DIR = create_path()
    num_features = 804
    num_samples_train = 10000
    num_samples_val = 1000
    num_samples_test = 4000
    features_train, labels_train, frame_id_train, instance_id_train  = generate_dataset_single(num_features, num_samples_train)
    features_val, labels_val, frame_id_val, instance_id_val = generate_dataset_single(num_features, num_samples_val)
    features_test, labels_test, frame_id_test, instance_id_test = generate_dataset_single(num_features, num_samples_test)
    
    """
    The meta data is used for configuration of the project. 
    Most attributes are related to the real data generator that generates data from real files recorded by the vehicle.
    For example, polar_coord means whether the coordinate of radar targets will be transformed to Cartesian coordinate system.
    """
    meta_data = {

    "all_mirroring": True,
    "attribute": "all data with FOV filtering 32 cropping",
    "augmentation": True,
    "crop_window_size": 5,
    "filter_lowlevel": True,
    "mirroring": True,
    "nms": True,
    "polar_coord": True,
    "shuffle": True,
    "test_only_nearby": False,
    "use_manual_annotation": True,
    "view_cluster": True,
    "viz_input": False,
    "x_lim": 1000,
    "yaw_threshold": 100
    }

    with open(osp.join(dataset_DIR, 'meta_data.json'), 'w') as fp:
        json.dump(meta_data, fp, sort_keys=True, indent=4, separators=(',', ': '))

    np.save(osp.join(data_DIR, 'train','features.npy'), features_train)
    np.save(osp.join(data_DIR, 'train','labels.npy'), labels_train)
    np.save(osp.join(data_DIR, 'train','frame_id.npy'), frame_id_train)
    np.save(osp.join(data_DIR, 'val','features.npy'), features_val)
    np.save(osp.join(data_DIR, 'val','labels.npy'), labels_val)
    np.save(osp.join(data_DIR, 'val','frame_id.npy'), frame_id_val)
    np.save(osp.join(data_DIR, 'test','features.npy'), features_test)
    np.save(osp.join(data_DIR, 'test','labels.npy'), labels_test)
    np.save(osp.join(data_DIR, 'test','frame_id.npy'), frame_id_test)
    np.save(osp.join(data_DIR,'instance_id_test.npy'), instance_id_test)