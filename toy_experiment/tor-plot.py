#%%

import os 

os.system('pip install pyro-ppl')
os.system('pip install ipympl')
os.system('pip install entrypoints')

#%%

import numpy as np
from data import (
    LiftingDist2Dto3D)
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", project_root_env_var=True, pythonpath=True)

major_radius = 2
minor_radius = 1
R = major_radius
r = minor_radius
N = 10
x = np.linspace(-(R+2*r), (R+2*r), N)
y = np.linspace(-(R+2*r), (R+2*r), N)
z = np.linspace(-r, r, N)
X_test = np.stack([x,z], axis=1) #(N,2)
Y_test = np.stack([x,y,z], axis=1) #(N,3)

distribution = LiftingDist2Dto3D(
            major_radius=2,
            minor_radius=1,
            weights= [0.3, 0.4,0.2,0.1],
            modes = [(-3.1415, 0), (0,3.1415/4),(0.5, -3.1415/4), (2*3.1415/3,3.1415/2)],
           dispersions=[(2,2), (4,4),(3,3),(10,10)],
            random_state=123)

_, points, components = distribution.sample(size=1000, output_components=True)
angles = distribution.torus_cartesian_to_angles_batch(major_radius=2,minor_radius=1,points = points)

#%%

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def plot_torus(angles, point_colors, R=2, r=1):
    """
    Visualize points on a torus given their angular coordinates.

    :param angles: A (N, 2) shaped array of angular coordinates (u, v).
    :param R: Major radius of the torus.
    :param r: Minor radius of the torus.
    """
    # Convert angular coordinates to Cartesian coordinates
    u, v = angles[:, 0], angles[:, 1]
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Labeling the axes
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Creating a mesh for the torus
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    u, v = np.meshgrid(u, v)
    x_tor = (R + r * np.cos(v)) * np.cos(u)
    y_tor = (R + r * np.cos(v)) * np.sin(u)
    z_tor = r * np.sin(v)

    # Plotting the torus mesh
    ax.plot_wireframe(x_tor, y_tor, z_tor, color='gray', alpha=0.3,zorder=1)

    ax.scatter(x, y, z, c=point_colors,zorder=2)

    elevation = 60
    azimuth = -45
    ax.view_init(azim=azimuth, elev=elevation)
    
    # plt.title('Distribution of Points on the Torus')

    plt.show()

colors = ['blue', 'green', 'red', 'purple']
point_colors = [colors[comp] for comp in components]
# Example usage
plot_torus(angles, point_colors= point_colors)  # Visualize the distribution on a torus

if not os.path.exists(os.path.join(os.environ['PROJECT_ROOT'],'toy_experiment/images')):
    os.makedirs(os.path.join(os.environ['PROJECT_ROOT'],'toy_experiment/images'))

filepath = os.path.join(os.environ['PROJECT_ROOT'],'toy_experiment/images/torus.pdf')

plt.savefig(filepath)
# %%
