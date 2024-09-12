"""
normals.py
----------
Function(s) for calculating the normal vectors of a point cloud.
"""

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from .voxelizer import voxelizer

def get_normals(P : np.ndarray, k : int = 10) -> np.ndarray:

    """
    Finds the unit normals of a point cloud using the k nearest neighbours and
    PCA.

    Parameters
    ----------
    P : (n, 3) np.ndarray
        Point cloud.
    k : int
        The number of nearest neighbours used in the calculation of the normals.

    Returns
    -------
    normals : (n, 3) np.ndarray
        The unit normal vector of each point in the point cloud oriented in the
        positive z sense.
    """

    # create empty array for normals
    normals = np.zeros(P.shape)

    # create kd-tree for cloud
    tree = KDTree(P)

    # iterate over all points in cloud
    for i in tqdm(range(P.shape[0])):

        # get indices of k nearest neighbors
        indices = tree.query(P[i,:], k=k)[1]

        # get k nearest neighbors
        neighbors = P[indices,:]

        # calculate covariance matrix
        cov = np.cov(neighbors.T)

        # calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # get index of smallest eigenvalue
        index = np.argmin(eigenvalues)

        # get corresponding eigenvector
        normal = eigenvectors[:,index]

        # ensure normal point up (z sense)
        if i > 0:
            if np.dot(normal, P[i,:]) < 0:
                normal = -normal

        # normalize normal
        normal = normal / np.linalg.norm(normal)

        # save normal to array
        normals[i,:] = normal

    return normals

def get_normals_adaptive(P : np.ndarray, grain_mm : float) -> np.ndarray:

    """
    Finds the unit normals of a point cloud using the k nearest neighbours and
    PCA. k is adaptively determined for each point based on the number of points
    in its respective voxel of side length grain_mm.

    Parameters
    ----------
    P : (n, 3) np.ndarray
        Point cloud.
    grain_mm : float
        The side length of each voxel cube in mm.

    Returns
    -------
    normals : (n, 3) np.ndarray
        The unit normals of each point in the point cloud.
    """

    # Create amps with 1 for each point
    amps = np.ones(P.shape[0])

    # Voxelize cloud
    print('Voxelizing cloud...')
    x_grid, y_grid, z_grid, vox_val_cube = voxelizer(P, amps, grain_mm, 'sum')

    # create empty array for normals
    normals = np.zeros(P.shape)

    # create kd-tree for cloud
    tree = KDTree(P)

    # iterate over all points in cloud
    print('Calculating normals...')
    for i in tqdm(range(P.shape[0])):
        
        # get number of points in voxel
        x_ind = np.searchsorted(x_grid, P[i,0])
        y_ind = np.searchsorted(y_grid, P[i,1])
        z_ind = np.searchsorted(z_grid, P[i,2])
        k = vox_val_cube[x_ind, y_ind, z_ind]
        k = int(np.ceil(k))

        if k < 5:
            k = 5

        # get indices of k nearest neighbors
        indices = tree.query(P[i,:], k=k)[1]

        # get k nearest neighbors
        neighbors = P[indices,:]

        # calculate covariance matrix
        cov = np.cov(neighbors.T)

        # calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # get index of smallest eigenvalue
        index = np.argmin(eigenvalues)

        # get corresponding eigenvector
        normal = eigenvectors[:,index]

        # ensure normal point up (z sense)
        if i > 0:
            if np.dot(normal, np.array([0, 0, 1])) < 0:
                normal = -normal

        # save normal to array
        normals[i,:] = normal

        # normalize normal
        normals[i,:] = normals[i,:] / np.linalg.norm(normals[i,:])

    return normals
