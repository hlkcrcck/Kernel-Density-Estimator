import numpy as np
from kde import KernelDensityEstimator


class VectorKernelDensityEstimator(KernelDensityEstimator):
    """Kernel Density Estimation.
    """

    def __init__(self, training_data, standard_deviation=1.0):
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
        super().__init__(training_data, standard_deviation)
        self.preallocate_matrix = np.zeros_like(self.D_A)
        self.preallocate_array = np.zeros(self.k)

    def log_proba(self, xb_i):
        """return np.log(
                np.sum(
                    np.exp(
                    self.constant_log_k -
                    np.sum(((((xb_i - self.D_A) ** 2) / self.constant1) + self.constant2)
                           , axis=1))))"""
        np.subtract(xb_i, self.D_A, out=self.preallocate_matrix)
        np.square(self.preallocate_matrix, out=self.preallocate_matrix)
        np.divide(self.preallocate_matrix, self.constant1, out=self.preallocate_matrix)
        np.add(self.preallocate_matrix, self.constant2, out=self.preallocate_matrix)
        return np.logaddexp.reduce((self.constant_log_k -
                                    np.sum(
                                        self.preallocate_matrix
                                        , axis=1)))

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
        log_density = np.apply_along_axis(self.log_proba, 1, test_data)
        return log_density
