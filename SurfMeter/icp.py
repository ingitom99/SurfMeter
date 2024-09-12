"""
icp.py
------
An implementation of the Iterative Closest Point algorithm for point cloud
registration utilising a kd-tree search for correspondence and SVD for
alignment.
"""

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

def correspondence(P : np.ndarray, Q : np.ndarray) -> tuple[np.ndarray,
                                                            np.ndarray]:
    
    """
    Finds the closest point in Q to each point in P using a kd-tree search.

    Parameters
    ----------
    P : (n, 3) np.ndarray 
        Source point cloud.
    Q : (m, 3) np.ndarray 
        Target point cloud.

    Returns
    -------
    Q_closest : (n, 3) np.ndarray
        Point cloud containing the closest points in Q to each point in P.
    closest_inds : (n,) np.ndarray
        Array containing the indices of the closest points in Q to each point in
        P.
    """

    # utilising a scipy kd-tree to find the closest points in Q to each point
    # in P efficiently
    tree = KDTree(Q)
    closests = tree.query(P)[1]
    Q_closests = Q[closests]

    return Q_closests, closests

def alignment(P : np.ndarray, Q : np.ndarray) -> tuple[np.ndarray,
                                                              np.ndarray]:
     
    """
    Finds the optimal rotation matrix and translation vector to align P to Q.

    Parameters
    ----------
    P : (n, 3) np.ndarray 
        Source point cloud.
    Q : (n, 3) np.ndarray 
        Target point cloud.
    
    Returns
    -------
    R : (3, 3) np.ndarray 
        Rotation matrix.
    t : (3,) np.ndarray
        Translation vector.
    """

    P_bar = P.mean(axis=0)
    Q_bar = Q.mean(axis=0)
    X = (P - P_bar).T
    Y = (Q - Q_bar).T
    S = X @ Y.T
    U, S, VT = np.linalg.svd(S)
    MID = np.eye(VT.T.shape[1])
    MID[-1,-1] = np.linalg.det(VT.T @ U.T)
    R = VT.T @ MID @ U.T
    t = Q_bar - R @ P_bar

    return R, t

def icp(P : np.ndarray, Q : np.ndarray, max_iter: int = 20,
        tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray]:

    """
    Performs the Iterative Closest Point algorithm on two point clouds utilising
    the correspondence and alignment functions.

    Parameters
    ----------
    P : (n, 3) np.ndarray
        Source (moving) point cloud.
    Q : (m, 3) np.ndarray
        Target (fixed) point cloud.
    max_iter : int
        The maximum number of iterations to perform.
    tol : float
        The tolerance for the translation vector norm.
    
    Returns
    -------
    P_k : (n, 3) np.ndarray 
        The transformed point cloud.
    closest_k : (n,) np.ndarray 
        Array containing the indices of the closest points in Q to each point in
        P_k.
    R : (3, 3) np.ndarray 
        Rotation matrix.
    t : (3,) np.ndarray 
        Translation vector.
    """

    print('Running ICP...')

    # initialise variables
    R = np.eye(3)
    t = np.array([0, 0, 0])
    P_k = P
    
    for i in tqdm(range(max_iter)):

        # find closest points in Q to each point in P_k
        Q_hat_k, closest_k = correspondence(P_k, Q)

        # find optimal rotation matrix and translation vector
        R_k, t_k = alignment(P_k, Q_hat_k)

        # transform P_k
        P_k = P_k @ R_k.T + t_k

        # update rotation matrix and translation vector
        R = R_k @ R
        t = t_k + t
        
        # check for convergence
        if np.linalg.norm(t_k) < tol:
            print('Tolerance reached!')
            break

        if (i == max_iter - 1):
            print('Maximum iterations reached!')

    # find closest points in Q to each point in P_k for the final time
    Q_hat_k, closest_k = correspondence(P_k, Q)

    return P_k, closest_k, R, t
