""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np
import scipy.stats

from utils import minimized_angle


class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE

        # Samples new particle positions, based on old positions, the odometry
        # measurements and the motion noise
        # (probabilistic motion models slide 27)
        # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
        alpha = self.alphas.ravel()

        # generate new particle set after motion update
        new_particles = np.zeros((self.num_particles, 3))
        idx = 0        
        for particle in self.particles:
            #sample noisy motions
            noisy_delta_rot1, noisy_delta_trans, noisy_delta_rot2 = env.sample_noisy_action(u, alpha)
            #calculate new particle pose
            new_particle = np.array([particle[0] + noisy_delta_trans * np.cos(particle[2] + noisy_delta_rot1), 
                                    particle[1] + noisy_delta_trans * np.sin(particle[2] + noisy_delta_rot1),
                                    particle[2] + noisy_delta_rot1 + noisy_delta_rot2])
            new_particles[idx] = new_particle.reshape(1,3)
            idx += 1
        
        new_weights = np.zeros(self.num_particles)
        idx=0
        #rate each particle
        for particle in new_particles:
            #calculate expected bearing measurement
            z_hat = env.observe(particle, marker_id)
            new_weights[idx] = env.likelihood(minimized_angle(z - z_hat), self.beta)
            idx += 1

        #normalize weights
        normalizer = sum(new_weights)
        weights = new_weights / normalizer
        self.weights = weights

        self.particles, self.weights = self.resample(new_particles, weights)

        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        # new_particles, new_weights = particles, weights
        # YOUR IMPLEMENTATION HERE
        # Resample particles
        # Returns a new set of particles obtained by performing
        # stochastic universal sampling, according to the particle
        # weights.
        new_particles = np.zeros_like(particles)
        n = len(particles)
        step = 1.0/n
        # random start of first pointer
        r = np.random.uniform(0,step)
        # where we are along the weights
        c = weights[0]        
        # index of weight container and corresponding particle
        i = 0
        #loop over all particle weights
        for  m in range(n):
            u = r + m * step
            #go through the weights until you find the particle
            #to which the pointer points
            while u > c:
                i += 1
                c += weights[i]
            #add that particle
            new_particles[m] = particles[i]
            #increase the threshold
            # u = u + step

        return new_particles, weights

    def mean_and_variance(self, particles):
        """Compute the mean and covariance matrix for a set of equally-weighted
        particles.

        particles: (n x 3) matrix of poses
        """
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.cos(particles[:, 2]).sum(),
            np.sin(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])
        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov

