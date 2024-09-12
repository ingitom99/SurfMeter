"""
test_shapes.py
--------------------
Functions for sampling different types of point clouds from different geometries
as well as well as auxiliary functions (such as noise injection) for creating
test data.
"""

import numpy as np

def make_sphere(n : int, r : float) -> np.ndarray:

    """
    Generates a sphere of n points with radius r.

    Parameters:
    -----------
    n : int
        The number of points to generate.
    radius : float
        The radius of the ball.

    Returns:
    -------
    points : (n, 3) np.ndarray 
        The generated spherical point cloud.
    """

    # generate random points in the unit cube
    points = np.random.randn(n, 3)

    # normalize the points to the unit sphere
    points /= np.linalg.norm(points, axis=1)[:, None]

    # scale the points to the desired radius
    points *= r

    return points

def make_cube(n : int, l : float) -> np.ndarray:
    
    """
    Generates a cube of n points with side length l.

    Parameters:
    ----------
    n : int
        The number of points to generate.
    l : float
        The side length of the cube.

    Returns:
    -------
    points : (n, 3) np.ndarray 
        The generated cubic point cloud.
    """
    
    cube = np.zeros((n, 3))
    for i in range(n):
        point = l*(np.random.rand(2) - 0.5)
        # get random int from (0 to 2)
        face = np.random.randint(0, 3)
        # get random sign
        sign = np.random.randint(0, 2) * 2 - 1
        point_ind = 0 
        for j in range(3):
            if j == face:
                cube[i, j] = sign * l/2
            else:
                cube[i, j] = point[point_ind]
                point_ind += 1
    return cube

def make_paraboloid(n : int, a : int = 1, b : int = 1,
                    l : int = 1) -> np.ndarray:
    
    """
    Generates a paraboloid of n points with parameters a, b and l.

    Parameters:
    ----------
    n : int
        The number of points to generate.
    a : int
        The x scaling parameter.
    b : int
        The y scaling parameter.
    l : int
        The side length of the square grid on which the paraboloid is sampled.

    Returns:
    -------
    points : (n, 3) np.ndarray
        The generated paraboloid point cloud.
    """
    
    # sample n points from square grid
    x = np.random.uniform(-l, l, n)
    y = np.random.uniform(-l, l, n)
    xy = np.vstack([x, y]).T

    # calculate the z coordinates
    z = xy[:, 0]**2 / a**2 + xy[:, 1]**2 / b**2

    # stack the coordinates into a (n,3) array of (x,y,z) coordinates
    points = np.vstack([xy.T, z]).T

    return points

def make_hyperbolic_paraboloid(n : int, a : int = 1, b : int = 1,
                                 l : int = 1) -> np.ndarray:
    
    """
    Generates a hyperbolic paraboloid of n points with parameters a, b and l.

    Parameters:
    ----------
    n : int
        The number of points to generate.
    a : int
        The x scaling parameter.
    b : int
        The y scaling parameter.
    l : int
        The side length of the square grid on which the paraboloid is sampled.

    Returns:
    -------
    points : (n, 3) np.ndarray
        The generated hyperbolic paraboloid point cloud.
    """
    
    # sample n points from square grid
    x = np.random.uniform(-l, l, n)
    y = np.random.uniform(-l, l, n)
    xy = np.vstack([x, y]).T

    # calculate the z coordinates
    z = xy[:, 0]**2 / a**2 - xy[:, 1]**2 / b**2

    # stack the coordinates into a (n,3) array of (x,y,z) coordinates
    points = np.vstack([xy.T, z]).T

    return points
    
def add_gaussian_noise(P : np.ndarray, sigma: float = 0.01) -> np.ndarray:

    """
    Adds gaussian noise to a point cloud.

    Parameters:
    ----------
    P : (n, 3) np.ndarray
        The point cloud to add noise to.
    sigma : float
        The standard deviation of the gaussian noise.

    Returns:
    -------
    P : (n, 3) np.ndarray
        The point cloud with added noise.
    """

    noise = np.random.normal(0, sigma, P.shape[0])
    P[:, 2] += noise

    return P

def add_normal_gauss_noise(P : np.ndarray, normals : np.ndarray,
                           sigma : float = 0.01) -> np.ndarray:

    """
    Adds gaussian noise to a point cloud.

    Parameters:
    -----------
    P : (n, 3) np.ndarray
        The point cloud to add noise to.
    normals : (n, 3) np.ndarray
        The normals of the point cloud.
    sigma : float
        The standard deviation of the gaussian noise.

    Returns:
    --------
    points : (n, 3) np.ndarray
        The point cloud with added noise in the normal directions.
    """

    noise = np.random.normal(0, sigma, points.shape[0])
    additions = noise[:, np.newaxis] * normals 
    points += additions
    
    return points
