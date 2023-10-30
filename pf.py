""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

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

        # generate new particle set after motion update
        new_particles = np.zeros((self.num_particles, 3))
        idx = 0
        for particle in self.particles:
            #sample noisy motions
            noisy_delta_rot1, noisy_delta_trans, noisy_delta_rot2 = env.sample_noisy_action(u, self.alphas.ravel())
            #calculate new particle pose
            new_particle = np.array([particle[0] + noisy_delta_trans * np.cos(particle[2] + noisy_delta_rot1), 
                                    particle[1] + noisy_delta_trans * np.sin(particle[2] + noisy_delta_rot1),
                                    particle[2] + noisy_delta_rot1 + noisy_delta_rot2])
            new_particles[idx] = new_particle.reshape(1,3)
            idx += 1

        # Given z as the angle to marker_id, calculate the expected observation
        # for each particle
        z_hat = np.zeros((self.num_particles, 1))
        diff = np.zeros((self.num_particles, 1))
        idx = 0
        for particle in new_particles:
            z_hat[idx] = env.observe(particle, marker_id)

            # Compute the difference between the actual and expected observation
            diff[idx] = minimized_angle(z - z_hat[idx])

            # Compute the weight for each particle using the Gaussian distribution
            self.weights[idx] = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (diff[idx] ** 2))
            idx += 1
        
        # Normalize the weights
        self.weights /= np.sum(self.weights)

        # Resample particles
        self.particles, self.weights = self.resample(new_particles, self.weights)

        mean, cov = self.mean_and_variance(self.particles)
        return mean, cov

    def resample(self, particles, weights):
        """Sample new particles and weights given current particles and weights. Be sure
        to use the low-variance sampler from class.

        particles: (n x 3) matrix of poses
        weights: (n,) array of weights
        """
        new_particles, new_weights = particles, weights
        # YOUR IMPLEMENTATION HERE

        n = len(particles)
        indices = np.arange(n)
        new_particles = np.zeros_like(particles)
        new_weights = np.ones(n) / n
        r = np.random.uniform(0, 1 / n)
        c = weights[0]
        i = 0

        for m in range(n):
            u = r + m * (1 / n)
            while u > c:
                i += 1
                c += weights[i]

            new_particles[m] = particles[i]
        return new_particles, new_weights

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

        # Check the condition of the covariance matrix
        condition_number = np.linalg.cond(cov)

        # Set a threshold for the condition number
        condition_threshold = 1e-12  # Adjust this threshold as needed

        if condition_number > condition_threshold:
            # If the covariance matrix is ill-conditioned, set it to the identity matrix
            print("Badly conditioned covariance matrix (setting to identity):", condition_number)
            cov = np.identity(3)  # Replace with the appropriate dimension

        return mean.reshape((-1, 1)), cov
