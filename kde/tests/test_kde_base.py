import pytest
import numpy as np
from kde import KernelDensityEstimator
from kde.utils import make_data


def test_dimensions():
    x_a = make_data(5, 3)
    x_b = make_data(5, 2)
    kde = KernelDensityEstimator(x_a)
    with pytest.raises(ValueError):
        kde.score(x_b)


def test_unexpected_args():
    x_a = np.array([1, 1])
    with pytest.raises(ValueError):
        KernelDensityEstimator(x_a, standard_deviation=0)


@pytest.mark.parametrize(
    "standard_deviation, x_a_obs, x_b_obs, dim, expected_result",
    [
        (0.5, 50, 5, 1, np.array([-2.77961202, -1.28150126, -1.26767026, -1.58569689, -1.81823223])),
        (0.2, 50, 5, 2, np.array([-2.03952222, -3.01882899, -7.28451713, -10.00375604,
                                  -1.64191657])),
    ],
)
def test_expected_results(standard_deviation, x_a_obs, x_b_obs, dim, expected_result):
    """
    Test results with the scikit-learn outputs
    """
    x_a = make_data(x_a_obs, dim)
    x_b = make_data(x_b_obs, dim)
    log_density = KernelDensityEstimator(x_a, standard_deviation=standard_deviation)\
        .score_samples(x_b)
    assert np.allclose(log_density, expected_result)
