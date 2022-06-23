import glob
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import distromax

import pytest

HERE = Path(__file__).parent
EXAMPLES = (HERE.parent / "examples").glob("*.py")

ground_truth_gumbel = stats.gumbel_r(1, 1)

def test_AnalyticalGammaToGumbel():

    possible_inputs = [{"dofs": 4.}, {"shape": 2., "scale":2.}, {"shape":2., "rate":0.5}]
    for input_values in possible_inputs:
        logging.info(f"Testing input {input_values}")
        analytical = distromax.analytical.AnalyticalGammaToGumbel(**input_values)
        assert analytical.shape == 2.
        assert analytical.scale == 2.

        for method in ["get_gumbel_loc_scale", "get_gumbel_mean_std"]:
            logging.info(f"Running {method}")
            getattr(analytical, method)(n=1000)


    possible_failures = [{"dofs": 1, "shape": 1}, {"shape": 1, "scale": 1, "rate": 1}, {}]
    for input_values in possible_failures:
        logging.info(f"Testing ValueError with {input_values}")
        with pytest.raises(ValueError):
            distromax.analytical.AnalyticalGammaToGumbel(**input_values)


def test_BatchMaxGumbel(samples=None, fitting_class=None, fitting_class_kwargs=None):

    samples = samples if samples is not None else ground_truth_gumbel.rvs(size=(100000, 2))
    fitting_class = fitting_class or distromax.BatchMaxGumbel
    fitting_class_kwargs = fitting_class_kwargs or {}

    # Test fit with and without num_batches
    for batch_size in [None, 1]:
        fg = fitting_class(samples, batch_size=batch_size, **fitting_class_kwargs)
        for i in range(2):
            np.testing.assert_allclose(
                fg.gumbel.args[i], ground_truth_gumbel.args[i], rtol=1e-2
            )

    # Test batch propagation parameters
    num_batches = 1000
    loc, scale = fg.gumbel.args
    prop_loc, prop_scale = fg.max_propagation(num_batches)
    np.testing.assert_allclose(prop_loc, loc + scale * np.log(num_batches))

def test_BatchMaxGumbelNotchingOutliers():
    # Basic test to check nothing is not broken
    fitting_class = distromax.BatchMaxGumbelNotchingOutliers
    fitting_class_kwargs = {"stopping_quantile": 1.}
    test_BatchMaxGumbel(fitting_class=fitting_class, fitting_class_kwargs=fitting_class_kwargs)


@pytest.mark.parametrize(
    "example",
    [pytest.param(ex, id=ex.name) for ex in EXAMPLES]
)
def test_Examples(example):
    subprocess.run([sys.executable, "-s", str(example)])


if __name__ == "__main__":
    args = sys.argv[1:] or ["-v", "-rs"]
    sys.exit(pytest.main(args=[__file__] + args))
