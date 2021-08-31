import pytest
import numpy as np
from kde import VectorKernelDensityEstimator
from kde import ParallelVectorKernelDensityEstimator
from kde.utils import make_data


@pytest.mark.parametrize(
    "standard_deviation, x_a_obs, x_b_obs, dim",
    [
        (0.2, 50, 5, 2),
        (0.2, 1, 1, 2),
        (0.2, 1, 1, 1),
    ],
)
def test_fallback_to_single(standard_deviation, x_a_obs, x_b_obs, dim):
    x_a = make_data(x_a_obs, dim)
    x_b = make_data(x_b_obs, dim)
    log_density_single = VectorKernelDensityEstimator(x_a, standard_deviation=standard_deviation)\
        .score_samples(x_b)
    log_density_parallel = ParallelVectorKernelDensityEstimator(x_a, standard_deviation=standard_deviation)\
        .score_samples(x_b)
    assert np.allclose(log_density_single, log_density_parallel)


@pytest.mark.parametrize(
    "standard_deviation, x_a_obs, x_b_obs, dim",
    [
        (0.5, 10, 10, 5),
    ],
)
def test_compare_single_parallel(standard_deviation, x_a_obs, x_b_obs, dim):
    x_a = make_data(x_a_obs, dim)
    x_b = make_data(x_b_obs, dim)
    log_density_single = VectorKernelDensityEstimator(x_a, standard_deviation=standard_deviation)\
        .score_samples(x_b)
    log_density_parallel = ParallelVectorKernelDensityEstimator(x_a, standard_deviation=standard_deviation)\
        .score_samples(x_b)

    assert np.allclose(log_density_single, log_density_parallel)