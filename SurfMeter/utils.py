"""
utils.py
--------
Small, auxiliary, functions that can be reused in multiple places in this
project.
"""

import stl
import numpy as np
import plotly.graph_objects as go
from pymatreader import read_mat
from tqdm import tqdm

from .transformation import coord_transform

def get_results(path : str) -> dict:

    """
    Returns the results dict containing the information from an ultrasonic 
    testing experiment from a .mat file.

    Parameters
    ----------
    path : str
        Path to the .mat file containing the results dict.

    Returns
    -------
    results : dict
        Dictionary containing the results of the ultrasonic testing experiment.
    """

    results = read_mat(path)['results']

    return results

def cloud_to_txt(cloud : np.ndarray, file_path_txt : str) -> None:

    """
    Saves a point cloud to a .txt file.
    
    Parameters
    ----------
    cloud : np.ndarray
        Point cloud to be saved.
    file_path_txt : str
        Path to the .txt file to be saved.
    """

    np.savetxt(file_path_txt, cloud, delimiter=',')

    return None

def path_to_cloud(file_path: str) -> np.ndarray:

    """
    Returns the point cloud from a .stl or .txt file.
    
    Parameters
    ----------
    file_path: str
        Path to the .stl or .txt file containing the STL data or point cloud.

    Returns
    -------
    cloud : np.ndarray
        Point cloud.
    """

    if file_path[-4:] == '.stl':
        your_mesh = stl.mesh.Mesh.from_file(file_path)
        cloud = np.vstack(your_mesh.vectors)
        
    elif file_path[-4:] == '.txt':
        data = np.loadtxt(file_path, delimiter=',')
        cloud = data[:,0:3]

    else:
        raise ValueError('File must be .stl or .txt')

    return cloud

def image_to_cloud(image : np.ndarray, x_vals : np.ndarray, z_vals : np.ndarray,
                   thresh_fact=0.75) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Converts an image of amplitude values into a point cloud of locations in the
    x_vals and z_vals coordinates with entries only where the amplitude is above
    a thresholded fraction of the max amplitude in the image.  A vector of 
    the corresponding amplitudes to the location point cloud is also returned.

    Parameters
    ----------
    image : (n, m) np.ndarray
        Image (matrix) of amplitude values.
    x_vals : (m,) np.ndarray
        Vector of location x values corresponding to the columns of the image.
    z_vals : (n,) np.ndarray
        Vector of location z values corresponding to the rows of the image.
    thresh_fact : float, optional
        Fraction of the max amplitude in the image to threshold the point cloud
        entries at, by default 0.75.
    """
    
    n = image.shape[0]
    m = image.shape[1]

    cloud = []
    amps = []

    thresh = thresh_fact * np.max(image) 

    for i in range(n):
        for j in range(m):
            amp = image[i,j]
            x = x_vals[j]
            z = z_vals[i]
            if amp > thresh:   
                cloud.append([0.0, x, z - 70])
                amps.append(amp)

    cloud = np.array(cloud)
    amps = np.array(amps)
    
    return cloud, amps

def results_to_cloud(results, wall,  thresh_fact):
    for i in tqdm(range(len(results['robang']))):
        robang = results['robang'][i]
        robpos = results['robpos'][i]
        x_vals = results[wall][i]['x']
        z_vals = results[wall][i]['z']
        image = results[wall][i]['aimage']
        cloud, amps = image_to_cloud(image, x_vals, z_vals,
                                     thresh_fact=thresh_fact)
        cloud_rob = coord_transform(cloud, robang, robpos, 0)
        if (i == 0):
            cloud_global = cloud_rob
            amps_global = amps
        else:
            cloud_global = np.vstack((cloud_global, cloud_rob))
            amps_global = np.hstack((amps_global, amps))
    return cloud_global, amps_global

def rob_to_sie(cloud_rob, orientation):
    if orientation == 'convex':
        cloud_sie = coord_transform(cloud_rob, np.deg2rad([4.65, -27.14, -87.32]),
                         [-897.15, 127.91, -91.40], 1)
    elif orientation == 'concave':
        cloud_sie = coord_transform(cloud_rob, np.deg2rad([-179.92, 26.98, 87.61]),
                         [-897.22, 184.28, -28.35], 1)
    else:
        raise ValueError('Orientation must be "convex" or "concave"')
    return cloud_sie

def plot_cloud(P : np.ndarray, plot_title : str, save : bool = False) -> None:
    
    """
    Plots a point cloud in a 3D plot.

    Parameters
    ----------
    P : (n, 3) np.ndarray 
        Point cloud.
    plot_title : str
        Title of the plot.
    save : bool, optional
        If True, saves the plot as a .png file, by default False.

    Returns
    -------
    None
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=P[:,0],
                                    y=P[:,1],
                                    z=P[:,2],
                                    mode='markers',
                                    marker=dict(size=2,
                                                    color='blue',
                                                    opacity=0.5,
                                                    ),))
    fig.update_layout(title=plot_title)

    if save:
        fig.write_image(save)
    else:
        fig.show()

    return None

def plot_clouds(P : np.ndarray, Q : np.ndarray, plot_title : str, name_P : str,
                name_Q : str, save : bool = False) -> None:
    
    """
    Plots two point clouds in a 3D plot.

    Parameters
    ----------
    P : (n, 3) np.ndarray 
        First point cloud.
    Q : (m, 3) np.ndarray
        Second point cloud.
    name_P : str, optional
        Name of the first point cloud, by default 'source'.
    name_Q : str, optional
        Name of the second point cloud, by default 'target'.
    save : bool, optional
        If True, saves the plot as a .png file, by default False.

    Returns
    -------
    None
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=P[:,0],
                                    y=P[:,1],
                                    z=P[:,2],
                                    mode='markers',
                                    marker=dict(size=2,
                                                    color='blue',
                                                    opacity=0.5,
                                                    ),
                                    name=name_P))
    fig.add_trace(go.Scatter3d(x=Q[:,0],
                                    y=Q[:,1],
                                    z=Q[:,2],
                                    mode='markers',
                                    marker=dict(size=2,
                                                    color='red',
                                                    opacity=0.5,
                                                    ),
                                    name=name_Q))
    fig.update_layout(title=plot_title)

    if save:
        fig.write_image(save)
    else:
        fig.show()

    return None

def plot_heat_cloud(P : np.ndarray, temps : np.ndarray, plot_title : str,
                    cloud_title : str, heat_title : str,
                    save : bool = False) -> None:
    
    """
    Plots a point cloud with a temperature for each point (3D heat map).

    Parameters
    ----------
    P : (n, 3) np.ndarray 
        Point cloud.
    temps : (n,) np.ndarray
        Array containing the temperature for each point in P.
    plot_title : str
        Title of the plot.
    cloud_title : str
        Title of the point cloud.
    heat_title : str
        Title of the heat map.
    save : bool, optional
        If True, saves the plot as a .png file, by default False.

    Returns
    -------
    None
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=P[:,0],
                                    y=P[:,1],
                                    z=P[:,2],
                                    mode='markers',				
                                    marker=dict(size=2,				
                                                    color=temps,				
                                                    opacity=0.5,
                                                    colorscale='magma',				
                                                    colorbar=dict(title=heat_title),							
                                                    ),
                                    name=cloud_title))				
    fig.update_layout(title=plot_title)									
    
    if save:
        fig.write_image(save)
    else:
        fig.show()

    return None

def plot_histogram(X : np.ndarray, Y : np.ndarray, name_X : str, name_Y : str,
                   plot_title : str, save = False) -> None:
    
    """
    Plots a histogram of two arrays (typically normal and tangential errors).

    Parameters
    ----------
    X : (n,) np.ndarray
        First array.
    Y : (n,) np.ndarray
        Second array.
    name_X : str
        Name of the first array.
    name_Y : str
        Name of the second array.
    plot_title : str
        Title of the plot.
    save : bool, optional
        If True, saves the plot as a .png file, by default False.
    
    Returns
    -------
    None
    """

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=X,
                               name=name_X,
                               marker_color='blue',
                               opacity=0.75))
    fig.add_trace(go.Histogram(x=Y,
                               name=name_Y,
                               marker_color='red',
                               opacity=0.75))
    fig.update_layout(title=plot_title,
                      barmode='overlay')
    
    if save:
        fig.write_image(save)
    else:
        fig.show()

    return None

def plot_arrows(P : np.ndarray, D : np.ndarray, D_name : str, plot_title : str,
                save : bool = False) -> None:

    """
    Plots a point cloud with an arrow at each point (typically for normals).

    Parameters
    ----------
    P : (n, 3) np.ndarray
        Point cloud.
    D : (n, 3) np.ndarray
        Array containing the direction of the arrows at each point in P.
    plot_title : str
        Title of the plot.
    save : bool, optional
        If True, saves the plot as a .png file, by default False.

    Returns
    -------
    None
    """

    fig = go.Figure()								
    fig.add_trace(go.Cone(x=P[:,0],
                          y=P[:,1],			
                          z=P[:,2],								
                          u=D[:,0],							
                          v=D[:,1],				
                          w=D[:,2],							
                          sizemode='absolute',									
                          sizeref=10,
                          colorbar=None,			
                          anchor='tail',					
                          opacity=0.25,						
                          name=D_name))						
    fig.update_layout(title=plot_title)
    
    if save:
        fig.write_image(save)
    else:
        fig.show()

    return None