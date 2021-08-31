import numpy as np
from math import pi, log, exp


class KernelDensityEstimator:
    """Kernel Density Estimation.
    """

    def __init__(self, train_data, standard_deviation=1.0):
        """Init Kernel Density model with the training data.
        Parameters
        ----------
        train_data : training dataset(D_A) of shape (n_observations, n_dims)
        standard_deviation : standart deviation(mu) of the Gaussian Kernel
        Returns
        -------
        self : object
            Returns instance of object.
        """
        self.D_A = np.atleast_2d(train_data).astype(np.float32)
        self.dimension = self.D_A.shape[1]
        if standard_deviation > 0:
            self.standard_deviation = standard_deviation
        else:
            raise ValueError("Standard_deviation must be bigger than 0!")

        self.k = len(self.D_A)  # training sample count
        self.constant1 = np.float32(2 * (self.standard_deviation ** 2))
        self.constant2 = np.float32((log(2 * pi * (self.standard_deviation ** 2))) / 2)
        self.constant_log_k = np.float32(log(1 / self.k))

    def score_samples(self, test_data):
        """Evaluate the log density model on the data.
        Parameters
        ----------
        test_data : test dataset(D_B) of shape (n_observations, n_dims)
        Returns
        -------
        log_density : ndarray of shape (n_observations,)
            We can compute for each instance in DB its log-probability
            under the model trained using DA
        """
        test_data = np.atleast_2d(test_data)
        if self.dimension != test_data.shape[1]:
            raise ValueError("Dimensions do not match with the training data!")

        log_density = []
        for xb_i in test_data:
            sum_k = 0
            for xa_i in self.D_A:
                sum_d = 0
                for xa_ij, xb_ij in zip(xa_i, xb_i):
                    sum_d -= (((xb_ij - xa_ij) ** 2) / self.constant1) + self.constant2
                sum_k += exp(self.constant_log_k + sum_d)
            log_density.append(log(sum_k))
        return log_density

    def score(self, test_data):
        """ Mean log-probability of DB
        Parameters
        ----------
        test_data : test dataset(D_B) of shape (n_observations, n_dims)
        Returns
        -------
        density : ndarray of shape (n_observations,)
            We can compute for each instance in DB its log-probability
            under the model trained using DA
        """
        log_density = self.score_samples(test_data)
        return np.mean(log_density)
