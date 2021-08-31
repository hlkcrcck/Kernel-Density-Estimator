import pytest
import numpy as np
from kde import KernelDensityEstimator
from kde import VectorKernelDensityEstimator
from kde.utils import make_data


@pytest.mark.parametrize(
    "standard_deviation, x_a_obs, x_b_obs, dim",
    [
        (0.2, 50, 5, 2),
        (0.8, 10, 10, 5),
        (1000, 50, 5, 2),
    ],
)
def test_compare_slow_fast(standard_deviation, x_a_obs, x_b_obs, dim):
    x_a = make_data(x_a_obs, dim)
    x_b = make_data(x_b_obs, dim)
    log_density_slow = KernelDensityEstimator(x_a, standard_deviation=standard_deviation)\
        .score_samples(x_b)
    log_density_fast = VectorKernelDensityEstimator(x_a, standard_deviation=standard_deviation)\
        .score_samples(x_b)

    assert np.allclose(log_density_slow, log_density_fast)
