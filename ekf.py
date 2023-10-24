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

        # Ejecuto el paso de predición, tomando la estimación actual y el comando
        mu_pred = env.forward(self.mu, u)
        # 
        G = env.G(mu_pred, u)
        V = env.V(mu_pred, u)
        d_rot1, d_tran, d_rot2 = u.ravel()
        alfa1, alfa2, alfa3, alfa4 = self.alphas.ravel()
        # M = env.noise_from_motion(u, self.alphas)
        M = np.array([[alfa1*np.square(d_rot1) + alfa2*np.square(d_tran), 0, 0], 
                      [0, alfa3*np.square(d_tran) + alfa4*np.square(d_rot1+d_rot2), 0], 
                      [0, 0, alfa1*np.square(d_rot1) + alfa2*np.square(d_tran)]])
        sigma_pred = G.dot(self.sigma).dot(G.T) + V.dot(M).dot(V.T)
        

        prev_x, prev_y, prev_theta = mu_pred.ravel()
        dx = env.MARKER_X_POS[marker_id] - prev_x
        dy = env.MARKER_Y_POS[marker_id] - prev_y
        z_hat = np.array([minimized_angle(np.arctan2(dy,dx)- prev_theta)]).reshape(1,1)
        # z = minimized_angle(z)

        print("z", z )
        print("z_hat", z_hat)

        H = env.H(u, marker_id)
        S = H.dot(sigma_pred).dot(H.T) + self.beta
        K = sigma_pred.dot(H.T).dot(np.linalg.inv(S))

        # print("z", z)
        # print("z_hat", z_hat)
        # print("mu", self.mu)
        # print("S", S.shape)
        # print("K", K)
        # print("H: ", H)
        # print("u: ", u)
        # print("prev_mu", self.mu)
        # print("prev_sigma", self.sigma)
        error_z = minimized_angle(z - z_hat)
        print("error_z", error_z)
        mu_corrected = mu_pred + K.dot(error_z)
        # mu_corrected[2] = minimized_angle(mu_corrected[2])
        sigma_corrected = (np.eye(3) - K.dot(H)).dot(sigma_pred)

        self.mu = mu_pred
        self.sigma = sigma_pred
        self.mu = mu_corrected
        self.sigma = sigma_corrected

        # print("new_mu", self.mu)
        # print("mew_sigma", self.sigma)

        return self.mu, self.sigma
