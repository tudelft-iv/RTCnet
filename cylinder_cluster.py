import numpy as np 
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
def cylinder_cluster(targets, eps_xy, eps_v, min_targets):

    distance_xy = distance.cdist(np.reshape(targets[:,:2], (-1, 2)), np.reshape(targets[:, :2], (-1, 2)))
    distance_v = distance.cdist(np.reshape(targets[:,2], (-1, 1)), np.reshape(targets[:, 2], (-1, 1)))
    distance_xy_mask = distance_xy < eps_xy
    if eps_v != 0:
        distance_v_mask = distance_v < eps_v
    else:
        distance_v_mask = np.ones(distance_v.shape).astype(np.bool) 
    distance_mask = np.logical_and(distance_xy_mask, distance_v_mask)
    distance_mask = distance_mask.astype(np.float)
    distance_mask[distance_mask<1] = 1e4
    
    clustering = DBSCAN(eps = 1.1, min_samples=min_targets, metric='precomputed').fit(distance_mask)
    return clustering.labels_