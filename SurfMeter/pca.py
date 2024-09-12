"""
pca.py
This file contains the function pca, which performs principal component
analysis on the specified data matrix X and returns the top k eigenvectors of
the covariance matrix of X, where k is the specified number of components.
"""

import numpy as np

def pca(X : np.ndarray, num_components : int) -> np.ndarray:
    """
    Performs principal component analysis on the specified data matrix X and
    returns the top k eigenvectors of the covariance matrix of X, where k is
    the specified number of components.

    Parameters
    ----------
    X : np.ndarray
        The data matrix to perform PCA on.
    num_components : int
        The number of components to return.

    Returns
    -------
    k_eigvecs : np.ndarray
        The top k eigenvectors of the covariance matrix of X, where k is the
        specified number of components.
    """
    # Standardize the feature matrix
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(X.T)
    
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort the eigenvalues and corresponding eigenvectors in descending order
    eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Select the top k eigenvectors based on the specified number of components
    k_eigvecs = np.array([eigen_pairs[i][1] for i in range(num_components)])
    
    return k_eigvecs
