from itertools import product

import numpy as np
from scipy import stats

import distromax

import pytest

from utils_for_testing import is_flaky

@pytest.fixture(params=product([1, 10, 50], [1, 5]))
def ground_truth_gumbel(request):
    return stats.gumbel_r(*request.param)


@pytest.mark.parametrize(
    "distribution_parameters",
    [
        {"dofs": 4.0},
        {"shape": 2.0, "scale": 2.0},
        {"shape": 2.0, "rate": 0.5},
    ],
)
@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
def test_AnalyticalGammaToGumbel(distribution_parameters):
    analytical = distromax.AnalyticalGammaToGumbel(**distribution_parameters)
    assert analytical.shape == 2.0
    assert analytical.scale == 2.0

    for method in ["get_gumbel_loc_scale", "get_gumbel_mean_std"]:
        getattr(analytical, method)(n=1000)


@pytest.mark.parametrize(
    "failing_parameters",
    [
        {"dofs": 1, "shape": 1},
        {"shape": 1, "scale": 1, "rate": 1},
        {},
    ],
)
def test_AnalyticalGammaToGumbel_failure(failing_parameters):
    with pytest.raises(ValueError):
        distromax.AnalyticalGammaToGumbel(**failing_parameters)


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
@pytest.mark.parametrize("batch_size", [None, 1])
def test_BatchMaxGumbel_gumbel_samples(ground_truth_gumbel, batch_size):
    fg = distromax.BatchMaxGumbel(
        ground_truth_gumbel.rvs(size=1000000), batch_size=batch_size
    )
    for i in range(2):
        np.testing.assert_allclose(
            fg.gumbel.args[i],
            ground_truth_gumbel.args[i],
            rtol=2e-2
        )

    # Test batch propagation parameters
    num_batches = 1000
    loc, scale = fg.gumbel.args
    prop_loc, prop_scale = fg.max_propagation(num_batches)
    np.testing.assert_allclose(prop_loc, loc + scale * np.log(num_batches))


@pytest.mark.flaky(max_runs=3, min_passes=1, rerun_filter=is_flaky)
@pytest.mark.parametrize("batch_size", [None, 1])
def test_BatchMaxGumbelNotchingOutliers_gumbel_samples(ground_truth_gumbel, batch_size):
    num_samples = 1000000

    fg = distromax.BatchMaxGumbelNotchingOutliers(
        np.vstack([np.arange(num_samples), ground_truth_gumbel.rvs(size=num_samples)]).T,
        batch_size=batch_size,
        stopping_quantile=1.0,
    )
    for i in range(2):
        np.testing.assert_allclose(
            fg.gumbel.args[i],
            ground_truth_gumbel.args[i],
            rtol=2e-2,
        )

    # Test batch propagation parameters
    num_batches = 1000
    loc, scale = fg.gumbel.args
    prop_loc, prop_scale = fg.max_propagation(num_batches)
    np.testing.assert_allclose(prop_loc, loc + scale * np.log(num_batches))
