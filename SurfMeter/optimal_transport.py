"""
optimal_transport.py
--------------------
Function(s) for calculating optimal transport distances between two point
clouds.
"""

import numpy as np

def cost(P : np.ndarray, Q : np.ndarray) -> np.ndarray:

    """
    Calculates the squared Euclidean distance cost matrix between two point
    clouds P and Q.

    Parameters
    ----------
    P : (n, 3) np.ndarray 
        First point cloud.
    Q : (m, 3) np.ndarray
        Second point cloud.

    Returns
    -------
    C : (n, m) np.ndarray
        Cost matrix between P and Q.
    """

    # initialise cost matrix
    C = np.zeros((len(P), len(Q)))

    #ierate over all points in P and Q
    for i in range(len(P)):
        for j in range(len(Q)):
            C[i,j] = np.linalg.norm(P[i]-Q[j])**2

    return C

def sinkhorn(mu : np.ndarray, nu : np.ndarray, C : np.ndarray,
                eps : float, maxiter : int = 100) -> tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray,
                                                           float]:
    
    """
    Calculates the Sinkhorn approximation of the regularized optimal
    transport problem.

    Parameters
    ----------
    mu : (n,) np.ndarray
        The source point cloud weights.
    nu : (m,) np.ndarray
        The target point cloud weights.
    C : (n, m) np.ndarray
        The cost matrix between the source and target point clouds.
    eps : float
        The regularization parameter.

    Returns
    -------
    u : (n,) np.ndarray
        First Sinkhorn scaling vector.
    v : (m,) np.ndarray
        Second Sinkhorn scaling vector.
    G : (n, m) np.ndarray
        The optimal transport plan.
    dist : float
        The optimal transport distance.
    """

    K = np.exp(-C/eps)
    v = np.ones_like(nu)

    for _ in range(maxiter):

        # scaling factor updates
        u = mu / (K @ v)
        v = nu / (K.T @ u)

    # approximate optimal transport plan
    G = np.diag(u)@K@np.diag(v)

    # approximate optimal transport distance 
    dist = np.trace(C.T@G)

    return u, v, G, dist
