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

        # Ejecuto el paso de predición, tomando la estimación actual y el comando, sin ruido
        # O VA CON RUIDO???????????
        u_noisy = env.sample_noisy_action(u, self.alphas)
        mu_pred = env.forward(self.mu, u)
        # Jacobiano del modelo de odometría respecto del estado
        G = env.G(self.mu, u)
        # Jacobiano del modelo de odometría respecto del control
        V = env.V(self.mu, u)
        # Matriz de covarianzas de las acciones de control
        M = env.noise_from_motion(u, self.alphas)
        # Caluclo el ruido devido al control y al movimiento
        sigma_pred = G.dot(self.sigma).dot(G.T) + V.dot(M).dot(V.T)
        
        # Calculo las mediciones esperadas
        z_hat = env.observe(mu_pred, marker_id)
        # El jacobiano de las mediciones
        H = env.H(u, marker_id)
        S = H.dot(sigma_pred).dot(H.T) + self.beta
        # Ganancia del filtro
        K = sigma_pred.dot(H.T).dot(np.linalg.inv(S))
        # Calculo el estado y las covarianzas correjidas
        # print(z.ravel(), z_hat.ravel(), z - z_hat)
        innovation = minimized_angle(z - z_hat)
        mu_corrected = mu_pred + K.dot(innovation)
        sigma_corrected = (np.eye(3) - K.dot(H)).dot(sigma_pred)

        self.mu = mu_pred
        self.sigma = sigma_pred
        self.mu = mu_corrected
        self.sigma = sigma_corrected

        return self.mu, self.sigma
