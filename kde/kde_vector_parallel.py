import multiprocessing

import numpy as np
from kde import VectorKernelDensityEstimator
from kde.utils import parallel_apply_along_axis


class ParallelVectorKernelDensityEstimator(VectorKernelDensityEstimator):
    """Kernel Density Estimation.
    With parallelization of the numpy apply_along_axis function
    from the implementation:
    https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
    """

    def log_proba(self, xb_i):
        """return np.log(
                np.sum(
                    np.exp(
                    self.constant_log_k -
                    np.sum(((((xb_i - self.D_A) ** 2) / self.constant1) + self.constant2)
                           , axis=1))))"""
        np.subtract(xb_i, self.D_A, out=self.preallocate_matrix)
        np.einsum('ij,ij->i',
                  self.preallocate_matrix, self.preallocate_matrix,
                  out=self.preallocate_array)
        np.divide(self.preallocate_array, self.constant1, out=self.preallocate_array)
        np.add(self.preallocate_array, (self.constant2 * self.dimension), out=self.preallocate_array)
        return np.logaddexp.reduce((self.constant_log_k - self.preallocate_array))

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

        if len(test_data) > multiprocessing.cpu_count() + 1:
            log_density = parallel_apply_along_axis(self.log_proba, 1, test_data)
        else:
            # Fallback to nonparallel implementation
            log_density = np.apply_along_axis(self.log_proba, 1, test_data)
        return log_density
