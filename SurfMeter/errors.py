"""
errors.py
---------
Function(s) for calculating different types of errors between two point clouds.
"""

import numpy as np

def get_errors(P : np.ndarray, Q : np.ndarray,
               Q_normals : np.ndarray) -> dict:

    """
    Calculates pointwise errors between two point clouds P and Q as well as
    normal and tangential components of the errors with respect to the normals
    of Q.

    Parameters
    ----------
    P : (n, 3) np.ndarray 
        Source point cloud.
    Q : (m, 3) np.ndarray 
        Target point cloud.
    Q_normals : (m, 3) np.ndarray 
        Unit normal vector of each point in Q.
    
    Returns
    -------
    errors : dict
        Dictionary containing the following arrays:
            E : (n, 3) np.ndarray 
                Pointwise errors between P and Q.
            E_L2 : (n,) np.ndarray 
                Pointwise L2 errors between P and Q.
            N : (n, 3) np.ndarray 
                Normal errors between P and Q.
            N_L2 : (n,) np.ndarray 
                Normal L2 errors between P and Q.
            N_signs : (n,) np.ndarray 
                Signs of the normal errors between P and Q.
            T : (n, 3) np.ndarray 
                Tangential errors between P and Q.
            T_L2 : (n,) np.ndarray 
                Tangential L2 errors between P and Q.
    """

    # calculate error vectors
    E = Q - P

    # calculate dot product of E and Q_normals to scale the normal errors
    scale_fact = np.sum(E * Q_normals, axis=1, keepdims=True) 

    # calculate normal and tangential components of errors
    N =  scale_fact * Q_normals
    T = E - N

    # calculate sizes of errors with L2 norm
    E_L2 = np.linalg.norm(E, axis=1)
    N_L2 = np.linalg.norm(N, axis=1)
    T_L2 = np.linalg.norm(T, axis=1)

    # calculate signs of normal errors
    N_signs = np.sign(scale_fact).flatten()

    return E, E_L2, N, N_L2, N_signs, T, T_L2