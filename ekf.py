""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark
        observation.

        u: action
        z: landmark observation
        marker_id: landmark ID
        """
        # YOUR IMPLEMENTATION HERE

        prev_x, prev_y, prev_theta = u.ravel()
        q = np.square(env.MARKER_X_POS[marker_id] - prev_x) + np.square(env.MARKER_Y_POS[marker_id] - prev_y)
        z_hat = np.array([np.sqrt(q),
                          minimized_angle(
                              np.arctan2(env.MARKER_Y_POS[marker_id] - prev_y,
                                         env.MARKER_X_POS[marker_id] - prev_x)
                                         - prev_theta)]).reshape(2,1)
        H = env.H(u, marker_id)

        print("mu", self.mu)
        print("beta", self.beta)
        print("alphas", self.alphas)
        print("sigma", self.sigma)
        print("u: ", u)
        print("z: ", z)
        print("H: ", H)

        S = H.dot(self.sigma).dot(H.T) + self.beta
        K = self.sigma.dot(H.T).dot(np.linalg.inv(S))

        self.mu = self.mu + K.dot(z - z_hat)
        self.sigma = (np.eye(3) - K.dot(H)).dot(self.sigma)

        return self.mu, self.sigma
