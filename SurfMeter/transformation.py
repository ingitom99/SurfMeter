"""
transformations.py
------------------
Contains functions for transforming point clouds between coordinate systems.
These functions are used in particular for the setting of transforming between
sensor coordinate systems and robot coordinate systems in robotic ultrasound
imaging.
"""

import numpy as np

def transmat(angles : np.ndarray, xtrans : np.ndarray) -> np.ndarray:

    """
    Returns the transformation matrix for a given set of angles and translation
    vector.

    Parameters
    ----------
    angles : np.ndarray
        Array of shape (3,) containing the yaw, pitch and roll angles of the
        manipulator. [yaw pitch roll]-angles of manipulator
    xtrans : np.ndarray
        Array of shape (3,) containing the translation vector after rotation.

    Returns
    -------
    F : np.ndarray
        Transformation matrix of shape (4, 4).
    """

    # angles = [yaw pitch roll]-angles of manipulator
    # xtrans = [x y z]-translation vector AFTER rotation
    a = angles[2]
    b = angles[1]
    c = angles[0]

    F = np.zeros((4,4))

    F[0,:] = [np.cos(b)*np.cos(a),
              np.sin(c)*np.sin(b)*np.cos(a)-np.cos(c)*np.sin(a),
              np.cos(c)*np.sin(b)*np.cos(a)+np.sin(c)*np.sin(a),
              xtrans[0]]
    
    F[1,:] = [np.cos(b)*np.sin(a),
              np.sin(c)*np.sin(b)*np.sin(a)+np.cos(c)*np.cos(a),
              np.cos(c)*np.sin(b)*np.sin(a)-np.sin(c)*np.cos(a),
              xtrans[1]]
    
    F[2,:] = [-np.sin(b),
              np.sin(c)*np.cos(b),
              np.cos(c)*np.cos(b),
              xtrans[2]]
    
    F[3,3] = 1.0

    return F


def coord_transform(points_in,rob_ang,rob_pos,inverse):

    """
    Transforms a point cloud to the coordinate system of the robot.

    Parameters
    ----------
    points_in : np.ndarray
        Point cloud of shape (n, 3).
    rob_ang : np.ndarray
        Array of shape (3,) containing the yaw, pitch and roll angles of the
        manipulator.
    rob_pos : np.ndarray
        Array of shape (3,) containing the translation vector after rotation.
    inverse : bool
        Whether to perform the inverse transformation.
    
    Returns
    -------
    points_out : np.ndarray
        Point cloud of shape (n, 3) containing the transformed point cloud.
    """


    n = len(points_in)
    points_out = np.zeros((n,3))

    F = transmat(rob_ang,rob_pos)
    if inverse:
        F = np.linalg.inv(F)

    points_in_expanded = np.hstack((points_in, np.ones((len(points_in), 1))))
    points_out_expanded = F @ points_in_expanded.T
    points_out = points_out_expanded[0:3, :].T

    return points_out
