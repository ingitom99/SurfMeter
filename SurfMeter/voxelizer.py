"""
voxelizer.py

Voxelizes a point cloud.
"""

import numpy as np
from tqdm import tqdm

def voxelizer(cloud : np.ndarray, amps : np.ndarray, grain_mm : float,
                    style : str = 'max') -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    
    """
    Takes a point cloud and returns a voxelizes it.

    Parameters
    ----------
    cloud : np.ndarray
        Point cloud of shape (n, 3).
    amps : np.ndarray
        Array of shape (n,) containing the associated amplitude of each point
        in the point cloud.
    grain_mm : float
        The side length of each voxel cube in mm.
    style : str
        The style of voxelization to be used. Can be 'max', 'mean' or 'sum'.

    Returns
    -------
    x_grid : np.ndarray
        Array of shape (x_num,) containing the x-coordinates of the voxel grid.
    y_grid : np.ndarray
        Array of shape (y_num,) containing the y-coordinates of the voxel grid.
    z_grid : np.ndarray
        Array of shape (z_num,) containing the z-coordinates of the voxel grid.
    vox_val_cube : np.ndarray
        Array of shape (x_num, y_num, z_num) containing the values of each
        voxel.
    """

    x_num = int(np.ceil((cloud[:,0].max() - cloud[:,0].min()))*(1/grain_mm))
    y_num = int(np.ceil((cloud[:,1].max() - cloud[:,1].min()))*(1/grain_mm))
    z_num = int(np.ceil((cloud[:,2].max() - cloud[:,2].min()))*(1/grain_mm))

    x_grid = np.linspace(cloud[:,0].min(), cloud[:,0].max(), num=x_num)
    y_grid = np.linspace(cloud[:,1].min(), cloud[:,1].max(), num=y_num)
    z_grid = np.linspace(cloud[:,2].min(), cloud[:,2].max(), num=z_num)

    voxel_xs = np.searchsorted(x_grid, cloud[:,0])
    voxel_ys = np.searchsorted(y_grid, cloud[:,1])
    voxel_zs = np.searchsorted(z_grid, cloud[:,2])

    voxel_coords = np.vstack((voxel_xs, voxel_ys, voxel_zs)).T

    vox_val_cube =  np.zeros((x_num, y_num, z_num))

    vox_val_cube_contrib =  np.zeros((x_num, y_num, z_num))

    if (style == 'mean'):

        for i, (x_ind, y_ind, z_ind) in tqdm(enumerate(voxel_coords)):

            vox_val_cube[x_ind, y_ind, z_ind] += amps[i]

            vox_val_cube_contrib[x_ind, y_ind, z_ind] += 1
    
        vox_val_cube = np.divide(vox_val_cube, vox_val_cube_contrib, 
                                     where=vox_val_cube_contrib!=0,
                                     out=np.zeros_like(vox_val_cube))
            
        return x_grid, y_grid, z_grid, vox_val_cube
    
    elif (style == 'sum'):
        for i, (x_ind, y_ind, z_ind) in tqdm(enumerate(voxel_coords)):

            vox_val_cube[x_ind, y_ind, z_ind] += amps[i]

        return x_grid, y_grid, z_grid, vox_val_cube
    
    elif (style == 'max'):
        for i, (x_ind, y_ind, z_ind) in tqdm(enumerate(voxel_coords)):

            if (amps[i] > vox_val_cube[x_ind, y_ind, z_ind]):
                vox_val_cube[x_ind, y_ind, z_ind] = amps[i]

        return x_grid, y_grid, z_grid, vox_val_cube
    