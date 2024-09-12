"""
similarity.py

Test the similirity of two surfaces represented as point clouds sourceing icp and
normal projection of errors.
"""

import numpy as np
from src.icp import icp
from src.normals import get_normals
from src.errors import get_errors
from scipy.spatial import KDTree
from src.utils import path_to_cloud

def get_box_bound(P : np.ndarray, Q : np.ndarray,
                  box_bound : np.ndarray) -> np.ndarray:
    
    """
    Returns a mask for the points in P that are within a cube enclosing the
    points in Q.

    Parameters
    ----------
    P : np.ndarray
        Point cloud of shape (n, 3).
    Q : np.ndarray
        Point cloud of shape (m, 3).
    box_bound : int
        The distance away from the min and max coordinates to define the
        bounding box.

    Returns
    -------
    box_mask : np.ndarray
        Boolean array of shape (n,) containing True for points in P that are
        within the bounding box of Q.
    """

    # get min and max in each dimension of Q
    min_x = Q[:, 0].min() - box_bound
    max_x = Q[:, 0].max() + box_bound
    min_y = Q[:, 1].min() - box_bound
    max_y = Q[:, 1].max() + box_bound
    min_z = Q[:, 2].min() - box_bound
    max_z = Q[:, 2].max() + box_bound

    # get mask for points in P that are within the bounding box of Q
    mask_x = (P[:, 0] > min_x) & (P[:, 0] < max_x)
    mask_y = (P[:, 1] > min_y) & (P[:, 1] < max_y)
    mask_z = (P[:, 2] > min_z) & (P[:, 2] < max_z)
    box_mask = mask_x & mask_y & mask_z

    return box_mask

def shape_tester(cloud_source : np.ndarray, cloud_target : np.ndarray, n_source : int,
                 n_target : int,  box_bound : int = 3, dist_bound : int = 3,
                 max_iter : int = 10, tol : float = 1e-5,
                 k : int = 10) -> dict:
    
    """
    Tests the similarity of two surfaces represented as point clouds sourceing icp
    and normal projection of errors.

    Parameters
    ----------
    cloud_source : np.ndarray
        Point cloud of shape (n, 3).
    cloud_target : np.ndarray
        Point cloud of shape (m, 3).
    n_source : int
        The number of points to sourcee from cloud_source.
    n_target : int
        The number of points to sourcee from cloud_target.
    box_bound : int
        The distance away from the min and max coordinates to define the
        bounding box for cropping.
    dist_bound : int
        The distance away from the closest point in cloud_source to define the
        cropping.
    max_iter : int
        The maximum number of iterations to perform in icp.
    tol : float
        The tolerance for the translation vector norm in icp.
    k : int
        The number of neighbours to use in the normal estimation.
        
    Returns
    -------
    results : dict
        Dictionary containing the results of the shape testing.
        'errors' : np.ndarray
            Array of shape (n,) containing the pointwise L2 errors between the
            two point clouds.
        'normal_errors' : np.ndarray
            Array of shape (n,) containing the L2 errors in the normal direction
            between the two point clouds.
        'tangential_errors' : np.ndarray
            Array of shape (n,) containing the L2 errors in the tangential
            direction between the two point clouds.
        'rotation_matrix' : np.ndarray
            Rotation matrix of shape (3, 3).
        'rotation_angle' : float
            Angle of rotation of the rotation matrix.
        'translation_vector' : np.ndarray
            Translation vector of shape (3,).
        'translation_norm' : float
            Norm of the translation vector.
        'cloud_source' : np.ndarray
            Point cloud of shape (n, 3) containing the points sourceed from
            cloud_source.
        'cloud_source_aligned' : np.ndarray
            Point cloud of shape (n, 3) containing the points sourceed from
            cloud_source after alignment.
        'cloud_target' : np.ndarray
            Point cloud of shape (m, 3) containing the points sourceed from
            cloud_target.
        'cloud_target_closest' : np.ndarray
            Point cloud of shape (n, 3) containing the closest points in
            cloud_target to each point in cloud_source.
        'normals' : np.ndarray
            Array of shape (n, 3) containing the unit normals of each point in
            cloud_source.
    """
    # get mean of both clouds concatenated together
    both_clouds = np.concatenate((cloud_source, cloud_target), axis=0)
    mean = np.mean(both_clouds, axis=0)
    cloud_source = cloud_source - mean
    cloud_target = cloud_target - mean
    
    # get box mask
    box_mask = get_box_bound(cloud_target, cloud_source, box_bound)

    cloud_target_boxed = cloud_target[box_mask, :]

    # find the distance between each target point and the closest source point
    print('Finding distances between each boxed target point and the closest source point...')
    tree = KDTree(cloud_source)
    dists = tree.query(cloud_target_boxed)[0]

    # mask the target cloud to only include points that are within 2 units of the
    # source cloud
    mask_target_close = (dists < dist_bound)

    # mask the target cloud
    cloud_target_cropped = cloud_target_boxed[mask_target_close, :]

    # get normals for cloud_source
    print('Getting normals of closest points in target cloud...')
    normals = get_normals(cloud_target_cropped, k)

    # random mask of target cloud
    if (n_target > cloud_target_cropped.shape[0]):
        n_target = cloud_target_cropped.shape[0]
        print(f'Warning: n_target > cropped target size. Setting n_target to {n_target}.')
              
    mask_target_rand = np.random.choice(cloud_target_cropped.shape[0], size=n_target,
                                     replace=False)
    cloud_target_icp = cloud_target_cropped[mask_target_rand, :]

    normals = normals[mask_target_rand, :]

    # random mask of source cloud
    if (n_source > cloud_source.shape[0]):
        n_source = cloud_source.shape[0]
        print(f'Warning: n_source > source cloud size. Setting n_source to {n_source}.')

    mask_source = np.random.choice(cloud_source.shape[0], size=n_source, replace=False)
    cloud_source_icp = cloud_source[mask_source, :]

    # print cloud_source and cloud_target shapes
    print(f'cloud_source shape: {cloud_source_icp.shape}')
    print(f'cloud_target shape: {cloud_target_icp.shape}')
    
    # run icp on cloud_source and cloud_target_local_surf
    print('Running icp...')
    cloud_source_aligned, closest_k, R, t = icp(cloud_source_icp, cloud_target_icp,
                                            max_iter=max_iter, tol=tol)
    
    # get cloud of closest target points to each source point
    cloud_target_closest = cloud_target_icp[closest_k,:]

    normals = normals[closest_k,:]

    # get errors
    E, E_L2, N, N_L2, N_signs, T, T_L2  = get_errors(cloud_source_aligned,
                                                            cloud_target_closest,
                                                            normals)

    # find norm of t and angle of rotation of R
    t_norm = np.linalg.norm(t)
    R_angle = np.arccos((np.trace(R) - 1) / 2)

    R_angle = np.rad2deg(R_angle)

    # dictionary of results
    results = {'error vectors': E,
               'L2 errors': E_L2,
               'normal error vectors': N,
               'L2 normal errors': N_L2,
               'normal error signs': N_signs,
               'tangential error vectors': T,
               'L2 tangential errors': T_L2,
               'rotation matrix': R,
               'rotation angle': R_angle,
               'translation vector': t,
               'translation norm': t_norm,
               'source cloud': cloud_source_icp,
               'aligned source cloud': cloud_source_aligned,
               'target cloud': cloud_target_icp,
               'target cloud closest': cloud_target_closest,
               'normals': normals,
               }
    
    return results

def shape_tester_path(file_path_source : str, file_path_target : str, n_source : int,
                 n_target : int,  box_bound : int = 3, dist_bound : int = 3,
                 max_iter : int = 10, tol : float = 1e-5,
                 n_neighbours : int = 10,
                 ) -> dict:
    
    """
    Wrapper function for shape_tester that takes the paths to the .txt files
    containing the point clouds.
    """

    cloud_source = path_to_cloud(file_path_source)

    cloud_target = path_to_cloud(file_path_target)

    results = shape_tester(cloud_source, cloud_target, n_source, n_target,
                           box_bound, dist_bound, max_iter, tol, n_neighbours)
    
    return results

