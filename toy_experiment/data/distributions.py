import numbers
from typing import Tuple
from pyro.distributions.sine_bivariate_von_mises import SineBivariateVonMises

import numpy as np
from scipy.special import i0

from utils.utils import check_random_state, polar2cartesian


class MixtureVonMises:
    def __init__(
        self,
        weights: Tuple,
        modes: Tuple,
        dispersions: Tuple,
        random_state: int,
    ):

        assert sum(weights) == 1
        self.weights = np.array(weights)
        assert all(self.weights >= 0)
        self.modes = np.array(modes)
        self.dispersions = np.array(dispersions)
        assert (
            self.weights.shape[0]
            == self.modes.shape[0]
            == self.dispersions.shape[0]
        )

        self.rng = check_random_state(random_state)
        self.components = np.arange(0, self.weights.shape[0])

    def sample(self, size: int) -> np.array:
        picked_components = self.rng.choice(
            self.components,
            size=size,
            p=self.weights,
        )

        samples = np.empty(size)
        for c, mu, kappa in zip(self.components, self.modes, self.dispersions):
            mask = picked_components == c
            size_c = sum(mask)
            samples[mask] = self.rng.vonmises(mu, kappa=kappa, size=size_c)

        return samples

    def pdf(self, theta):
        theta = np.array(theta)
        if len(theta.shape) > 0:
            theta = theta[:, None]
        return np.sum(
            self.weights
            * np.exp(self.dispersions * np.cos(theta - self.modes))
            / (2 * np.pi * i0(self.dispersions)),
            axis=1,
        )


class BivariateVonMises:

    def __init__(self,phi_loc,
        psi_loc,
        phi_concentration,
        psi_concentration,
        correlation=None):

        self.Dist = SineBivariateVonMises(phi_loc=phi_loc,psi_loc=psi_loc,phi_concentration=phi_concentration,psi_concentration=psi_concentration,correlation=correlation)

    def sample(self,n_samples):
    
        return self.Dist.sample(sample_shape=(n_samples,1)).squeeze(1)

class BivariateVonMisesMixture:

    def __init__(self,weights: Tuple,
        modes: Tuple,
        dispersions: Tuple,
        random_state: int = 1234,
        correlation=None):

        assert abs(sum(weights)-1) <= 1e-5
        self.weights = np.array(weights)
        assert all(self.weights >= 0)
        self.modes = np.array(modes)
        self.dispersions = np.array(dispersions)
        self.correlation = correlation

        assert (
            self.weights.shape[0]
            == self.modes.shape[0]
            == self.dispersions.shape[0]
        )

        self.rng = check_random_state(random_state)
        self.components = np.arange(0, self.weights.shape[0])

    def torusanglestocartesian(self, major_radius, minor_radius, angles):
        """
        Converts angles on a torus to points in 3D Euclidean space.

        Parameters:
        - major_radius (float): The major radius of the torus.
        - minor_radius (float): The minor radius of the torus.
        - angles (numpy.ndarray): An array of shape (n_samples, 2) containing the angles.

        Returns:
        - numpy.ndarray: An array of shape (n_samples, 3) containing the Cartesian coordinates.
        """
        x = (major_radius + minor_radius * np.cos(angles[:, 0])) * np.cos(angles[:, 1])
        y = (major_radius + minor_radius * np.cos(angles[:, 0])) * np.sin(angles[:, 1])
        z = minor_radius * np.sin(angles[:, 0])

        return np.stack((x, y, z), axis=1)  # shape (n_samples, 3)
    
    def torus_cartesian_to_angles_batch(self,major_radius, minor_radius, points):
        """
        Converts a batch of points in 3D Euclidean space to angles on a torus.

        Parameters:
        - major_radius (float): The major radius of the torus.
        - minor_radius (float): The minor radius of the torus.
        - points (numpy.ndarray): A batch of points in 3D space, represented as a numpy array of shape (B, 3).

        Returns:
        - numpy.ndarray: An array of shape (B, 2) containing the angles (phi, theta) for each input point.
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Calculate the angle phi
        phi = np.arctan2(y, x)

        # Calculate the angle theta
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(z, r - major_radius)

        # Ensure theta is within the correct range (0 to 2*pi)
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        return np.column_stack((phi, theta))
    
    def pdf(self, angles):
        angles = np.array(angles)
        # if len(angles.shape) > 0:
            # angles = angles[:, None]
        if self.correlation is None :

            # d = np.cos(np.repeat(np.expand_dims(angles[:,0],1),2,axis=-1) - np.repeat(np.expand_dims(self.modes[:,0],0),repeats=angles.shape[0],axis=0))
            # c = np.cos(np.repeat(np.expand_dims(angles[:,0],1),2,axis=-1) - np.repeat(np.expand_dims(self.modes[:,0],0),repeats=angles.shape[0],axis=0))+np.repeat(np.expand_dims(self.dispersions[:,1],0),repeats=angles.shape[0],axis=0) * np.cos(np.repeat(np.expand_dims(angles[:,1],1),repeats=2,axis=-1) - np.repeat(np.expand_dims(self.modes[1,:],0),repeats=angles.shape[0],axis=0))
            # a = np.repeat(self.weights.reshape(1,-1),angles.shape[0],axis=0) * np.exp(np.cos(np.repeat(np.expand_dims(angles[:,0],1),2,axis=-1) - np.repeat(np.expand_dims(self.modes[:,0],0),repeats=angles.shape[0],axis=0))+np.repeat(np.expand_dims(self.dispersions[:,1],0),repeats=angles.shape[0],axis=0) * np.cos(np.repeat(np.expand_dims(angles[:,1],1),repeats=2,axis=-1) - np.repeat(np.expand_dims(self.modes[1,:],0),repeats=angles.shape[0],axis=0)))
            # e = np.sum(np.repeat(self.weights.reshape(1,-1),angles.shape[0],axis=0) * np.exp(np.cos(np.repeat(np.expand_dims(angles[:,0],1),2,axis=-1) - np.repeat(np.expand_dims(self.modes[:,0],0),repeats=angles.shape[0],axis=0))+np.repeat(np.expand_dims(self.dispersions[:,1],0),repeats=angles.shape[0],axis=0) * np.cos(np.repeat(np.expand_dims(angles[:,1],1),repeats=2,axis=-1) - np.repeat(np.expand_dims(self.modes[1,:],0),repeats=angles.shape[0],axis=0))),
            #               axis=1)
            # h = self.dist.Dist.norm_const.cpu().numpy()

            norm_c = np.concatenate([np.array([e.Dist.norm_const.cpu().numpy()]) for e in self.dist_list],axis=0).reshape(1,-1)
            norm_c = np.repeat(norm_c,angles.shape[0],axis=0)

            # return np.sum(np.repeat(self.weights.reshape(1,-1),angles.shape[0],axis=0) * np.exp(np.cos(np.repeat(np.expand_dims(angles[:,0],1),2,axis=-1) - np.repeat(np.expand_dims(self.modes[:,0],0),repeats=angles.shape[0],axis=0))+np.repeat(np.expand_dims(self.dispersions[:,1],0),repeats=angles.shape[0],axis=0) * np.cos(np.repeat(np.expand_dims(angles[:,1],1),repeats=2,axis=-1) - np.repeat(np.expand_dims(self.modes[1,:],0),repeats=angles.shape[0],axis=0))),
            #               axis=1)

            N_modes = self.modes.shape[0]

            before_sum = np.repeat(self.weights.reshape(1,-1),angles.shape[0],axis=0) * np.exp(np.cos(np.repeat(np.expand_dims(angles[:,0],1),N_modes,axis=-1) - np.repeat(np.expand_dims(self.modes[:,0],0),repeats=angles.shape[0],axis=0))+np.repeat(np.expand_dims(self.dispersions[:,1],0),repeats=angles.shape[0],axis=0) * np.cos(np.repeat(np.expand_dims(angles[:,1],1),repeats=N_modes,axis=-1) - np.repeat(np.expand_dims(self.modes[:,1],0),repeats=angles.shape[0],axis=0)))

            return np.sum(np.repeat(self.weights.reshape(1,-1),angles.shape[0],axis=0) * np.exp(np.cos(np.repeat(np.expand_dims(angles[:,0],1),N_modes,axis=-1) - np.repeat(np.expand_dims(self.modes[:,0],0),repeats=angles.shape[0],axis=0))+np.repeat(np.expand_dims(self.dispersions[:,1],0),repeats=angles.shape[0],axis=0) * np.cos(np.repeat(np.expand_dims(angles[:,1],1),repeats=N_modes,axis=-1) - np.repeat(np.expand_dims(self.modes[:,1],0),repeats=angles.shape[0],axis=0)))/norm_c,
                          axis=1)

            # return np.sum(np.repeat(self.weights.reshape(1,-1),angles.shape[0],axis=0) * np.exp(np.cos(np.repeat(np.expand_dims(angles[:,0],1),2,axis=-1) - np.repeat(np.expand_dims(self.modes[:,0],0),repeats=angles.shape[0],axis=0))+np.repeat(np.expand_dims(self.dispersions[:,1],0),repeats=angles.shape[0],axis=0) * np.cos(np.repeat(np.expand_dims(angles[:,1],1),repeats=2,axis=-1) - np.repeat(np.expand_dims(self.modes[1,:],0),repeats=angles.shape[0],axis=0))),
            #               axis=1)/ self.dist.Dist.norm_const.cpu().numpy()

        else :
            # raise error
            raise NotImplementedError

    def pytorch_torusanglestocartesian(self,major_radius, minor_radius, angles) :
        """Converts angles on a torus to points the 3D euclidean space"""
        # angles: array of shape (n_samples, 2)
        # radius: float

        x = (major_radius + minor_radius*torch.cos(angles[:,0]))*torch.cos(angles[:,1])
        y = (major_radius + minor_radius*torch.cos(angles[:,0]))*torch.sin(angles[:,1])
        z = minor_radius*torch.sin(angles[:,0])

        return torch.stack((x,y,z), dim=1) # shape (n_samples, 3)

    def sample(self, size: int) -> np.array:
        self.picked_components = self.rng.choice(
            self.components,
            size=size,
            p=self.weights,
        )

        samples = np.empty(shape=(size,2))

        for c, mu, kappa in zip(self.components, self.modes, self.dispersions):
            
            self.dist = BivariateVonMises(phi_loc=mu[0], psi_loc=mu[1],phi_concentration=kappa[0],psi_concentration=kappa[1],correlation=0.)

            mask = self.picked_components == c
            size_c = sum(mask)
            samples[mask] = self.dist.sample(n_samples=size_c)  

        return samples
    
    def visualize(self,n_samples,r=0.5,R=3): 

        angles = self.sample(n_samples)
                
        # Convert angles to Cartesian coordinates
        x = (R + r * np.cos(angles[:, 0])) * np.cos(angles[:, 1])
        y = (R + r * np.cos(angles[:, 0])) * np.sin(angles[:, 1])
        z = r * np.sin(angles[:, 0])

        # Create a 3D plot with wireframe torus
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Create a meshgrid for the torus
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, 2 * np.pi, 100)
        U, V = np.meshgrid(u, v)
        X = (R + r * np.cos(V)) * np.cos(U)
        Y = (R + r * np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)

        # Plot the torus surface
        ax.plot_surface(X, Y, Z, rstride=5, cstride=5,color='k', edgecolors='w',alpha=.1)

        # Create a colormap for the modes
        # cmap = plt.get_cmap("hsv", len(self.components))
        cmap = ['red','green','yellow','purple']

        # Plot each mode with a different color
        for c, mu, kappa in zip(self.components, self.modes, self.dispersions):
            mask = self.picked_components == c
            x_mode = x[mask]
            y_mode = y[mask]
            z_mode = z[mask]

            ax.scatter(x_mode, y_mode, z_mode, c=cmap[c], marker='o', s=5, label=f'Mode {c}')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mixture of Bivariate Cauchy Distributions Torus')

        # Set axis limits for better visualization
        ax.set_xlim(-R - r, R + r)
        ax.set_ylim(-R - r, R + r)
        ax.set_zlim(-r, r)

        # Set a better viewing angle
        # ax.view_init(elev=0, azim=100)
        ax.view_init(elev=15, azim=-160)

        # Add a legend for mode colors
        ax.legend()

        # Show the plot
        plt.show()

class LiftingDist1Dto2D(MixtureVonMises):
    def __init__(
        self,
        radius: float,
        weights: Tuple,
        modes: Tuple,
        dispersions: Tuple,
        random_state: int,
    ):
        super().__init__(weights, modes, dispersions, random_state)
        assert isinstance(radius, numbers.Real)
        assert radius > 0
        self.radius = radius

    def sample(self, size: int) -> Tuple[np.array, np.array]:
        angles = super().sample(size)
        x, y = polar2cartesian(self.radius, angles)
        return x, np.hstack([x[:, None], y[:, None]])




class LiftingDist2Dto3D(BivariateVonMisesMixture):
    def __init__(
        self,
        major_radius: float,
        minor_radius: float,
        weights: Tuple,
        modes: Tuple,
        dispersions: Tuple,
        random_state: int,
    ):
        super().__init__(weights, modes, dispersions, random_state)
        assert isinstance(major_radius, numbers.Real)
        assert major_radius > 0
        assert isinstance(minor_radius, numbers.Real)
        assert minor_radius > 0
        self.major_radius = major_radius
        self.minor_radius = minor_radius

        self.dist_list  = list()

        for c, mu, kappa in zip(self.components, self.modes, self.dispersions):
            self.dist_list.append(BivariateVonMises(phi_loc=mu[0], psi_loc=mu[1],phi_concentration=kappa[0],psi_concentration=kappa[1],correlation=0.))

    def sample(self, size: int) -> Tuple[np.array, np.array]:
        angles = super().sample(size)
        cartesian_points = self.torusanglestocartesian(major_radius=self.major_radius,minor_radius=self.minor_radius,angles=angles)
        assert np.shape(np.stack((cartesian_points[:,0],cartesian_points[:,2]),axis=-1)) == (size,2)
        assert np.shape(cartesian_points) == (size,3)
        return np.stack((cartesian_points[:,0],cartesian_points[:,2]),axis=-1), cartesian_points