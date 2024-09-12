"""
kmeans.py
---------
Function(s) for performing the k-means algorithm on a point cloud.
"""

import numpy as np
from tqdm import tqdm

def kmeans(P : np.ndarray, k : int,
           maxiter : int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the k-means algorithm on a point cloud P to find k clusters.

    Parameters
    ----------
    P : (n, 3) np.ndarray 
        Point cloud to be clustered.
    k : int
        The number of clusters.
    max_iter : int
        The maximum number of iterations to perform.

    Returns
    -------
    centroids : (k, 3) np.ndarray
        Array of containing the cluster centroids.
    """

    print('Running k-means...')

    # initialize centroids
    centroids = np.zeros((k, 3))
    for i in range(k):
        rand_idx = np.random.randint(0, P.shape[0])
        centroids[i] = P[rand_idx]

    # initialize closest
    closest = np.zeros(P.shape[0], dtype=int)
    
    for _ in tqdm(range(maxiter)):
        
        # assign points to clusters
        for i in range(P.shape[0]):
            closest[i] = np.argmin(np.linalg.norm(P[i] - centroids, axis=1))

        # update centroids
        for i in range(k):
            centroids[i] = P[closest == i].mean(axis=0)

    return closest, centroids