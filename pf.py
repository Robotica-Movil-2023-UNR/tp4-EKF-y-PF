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
        d_rot1, d_trans, d_rot2 = u
        # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
        alpha = self.alphas.ravel()

        # Find Std. Dev. of Normal Distribution for noise in rotation and translation.
        # sigma_rot1 = np.sqrt(alpha[0]*abs(d_rot1) + alpha[1]*d_trans)
        # sigma_rot2 = np.sqrt(alpha[0]*abs(d_rot2) + alpha[1]*d_trans)
        # sigma_trans = np.sqrt(alpha[2]*abs(d_trans) + alpha[3]*(abs(d_rot1) + abs(d_rot2)))
        # generate new particle set after motion update
        new_particles = np.zeros((self.num_particles, 3))
        idx = 0        
        for particle in self.particles:
            #sample noisy motions
            # noisy_delta_rot1 = d_rot1 + np.random.normal(0, sigma_rot1)
            # noisy_delta_trans = d_trans + np.random.normal(0, sigma_rot2)
            # noisy_delta_rot2 = d_rot2 + np.random.normal(0, sigma_trans)
            noisy_delta_rot1, noisy_delta_trans, noisy_delta_rot2 = env.sample_noisy_action(u, alpha)
            #calculate new particle pose
            new_particle = np.array([particle[0] + noisy_delta_trans * np.cos(particle[2] + noisy_delta_rot1), 
                                    particle[1] + noisy_delta_trans * np.sin(particle[2] + noisy_delta_rot1),
                                    particle[2] + noisy_delta_rot1 + noisy_delta_rot2])
            new_particles[idx] = new_particle.reshape(1,3)
            idx += 1
        
        # Computes the observation likelihood of all particles, given the
        # particle and landmark positions and sensor measurements
        # (probabilistic sensor models slide 33)
        #
        sigma_r = self.beta.item()
        new_weights = np.zeros(self.num_particles)
        idx=0
        #rate each particle
        for particle in new_particles:
            meas_bearing = z
            #calculate expected range measurement
            # dx = env.MARKER_X_POS[marker_id] - particle[0]
            # dy = env.MARKER_Y_POS[marker_id] - particle[1]
            # exp_bearing = np.array([minimized_angle(np.arctan2(dy, dx) - particle[2])]).reshape((-1, 1))            
            exp_bearing = env.observe(particle, marker_id)
            #evaluate sensor model (probability density function of normal distribution)
            meas_likelihood = scipy.stats.norm.pdf(meas_bearing, exp_bearing, sigma_r)
            new_weights[idx] = meas_likelihood
            idx += 1

        #normalize weights
        normalizer = sum(new_weights)
        weights = new_weights / normalizer

        self.weights = weights

        # Resample particles
        # Returns a new set of particles obtained by performing
        # stochastic universal sampling, according to the particle
        # weights.
        # new_particles = []
        # distance between pointers
        step = 1.0/len(new_particles)
        # random start of first pointer
        u = np.random.uniform(0,step)
        # where we are along the weights
        c = weights[0]        
        # index of weight container and corresponding particle
        i = 0
        new_particles2 = np.zeros((self.num_particles, 3))
        idx = 0
        #loop over all particle weights
        for particle in new_particles:
            #go through the weights until you find the particle
            #to which the pointer points
            while u > c:
                i = i + 1
                c = c + weights[i]
            #add that particle
            new_particles2[idx] = new_particles[i]
            #increase the threshold
            u = u + step
            idx += 1

        self.particles = new_particles2

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

        return mean.reshape((-1, 1)), cov

    # Sample from Normal Distribution
    def sample12_normal(sigma):
        total = 0.0
        for i in range(12):
            total += (-2*sigma*np.random.rand()+sigma)
        return (total/2.0)

    # Odometry Based Motion Model.
    def odom_motion_model(self, particle, odometry, alpha):
        # Arguments
            # particle 		-> (x, y, theta)
            # odometry 		-> (delta-rot1, delta-rot2, delta-trans)
            # alpha 		-> (alpha1, alpha2, alpha3, alpha4)	
            # 				   (alpha1, alpha2 associate with noise in rotation and the other two are related to noise in translation.)
        # Output
            # new_particle	-> (x', y', theta') 

        u_t = np.array([odometry['r1'], odometry['r2'], odometry['t']])
        # Find Std. Dev. of Normal Distribution for noise in rotation and translation.
        sigma_rot1 = np.sqrt(alpha[0]*abs(odometry['r1']) + alpha[1]*odometry['t'])
        sigma_rot2 = np.sqrt(alpha[0]*abs(odometry['r2']) + alpha[1]*odometry['t'])
        sigma_trans = np.sqrt(alpha[2]*abs(odometry['t']) + alpha[3]*(abs(odometry['r1']) + abs(odometry['r2'])))

        # Add the sampled noise to the u_t (kind of measurement noise in sensor).
        new_odometry = dict()
        new_odometry['r1'] = odometry['r1'] + self.sample12_normal(sigma_rot1)
        new_odometry['r2'] = odometry['r2'] + self.sample12_normal(sigma_rot2)
        new_odometry['t'] = odometry['t'] + self.sample12_normal(sigma_trans)

        # Update the position of robot using the del_u_t.
        new_particle = dict()
        new_particle['x'] = particle['x'] + new_odometry['t']*np.cos(particle['theta']+new_odometry['r1'])
        new_particle['y'] = particle['y'] + new_odometry['t']*np.sin(particle['theta']+new_odometry['r1'])
        new_particle['theta'] = particle['theta'] + new_odometry['r1'] + new_odometry['r2']

        return new_particle