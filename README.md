# distromax

Empirically estimating the distribution of the loudest candidate from a gravitational-wave search

This package implements the methods described in [Tenorio, Modafferi, Keitel, Sintes (2021)](https://dcc.ligo.org/LIGO-P2100277)
to estimate the distribution of the loudest candidate from a search. 

The actual implementation includes:

- [fit.py](distromax/fit.py):
    - `BatchMaxGumbel`: Basic `distromax` method. Construct the batchmax distribution of a set of samples and 
    return the corresponding Gumbel fit. Max. propagation is included as a method.
    - `BatchMaxGumbelNotchingOutlier`: Thin wrapper around `BatchMaxGumbel` to notch narrow-band
    outliers before computing the batchmax distribution. The specifics of this implementation are discussed
    on appendix B of the accompanying publication. 
- [analytical.py](distromax/analytical.py):
    - `AnalyticalGammaToGumbel`: Compute the Gumbel distribution associated to the maximum of
    a set of independent Gamma random variables using the formulas derived 
    in [Gasull, López-Salcedo, Utzet (2015)](https://link.springer.com/article/10.1007%2Fs11749-015-0431-9).

See the [examples](examples) for concrete applications of these classes.

## Citing this work

If `distromax` was useful for your work, we would appreciated if you cite both the
software version DOI (TBA) and one or more of the following scientific papers:

- Introduction of `distromax` and description of the method: [Tenorio, Modafferi, Keitel, Sintes, (2021)](https://dcc.ligo.org/LIGO-P2100277)
- Analytical limit of a Gamma random variable to a Gumbel distribution: 
[A. Gasull, J. López-Salcedo, F. Utzet TEST volume 24, pages 714–733 (2015)](https://link.springer.com/article/10.1007%2Fs11749-015-0431-9)

## How to install

Please, make sure you are running on a 
[virtual environment](https://docs.python.org/3/library/venv.html) to avoid
any conflicts with system libraries.

The simplest way (as of now) is to clone this repo with `git clone` and install using `pip`:
```
git clone https://github.com/Rodrigo-Tenorio/distromax.git
cd distromax
pip install .
```

Support for installations from PyPI and conda-forge will be ready in a few.

In some conservative systems the default `setuptools` may not be the latest version and
produce an erorr message such as
```
ERROR: setuptools==44.1.1 is used in combination with setuptools_scm>=6.x

Your build configuration is incomplete and previously worked by accident!


This happens as setuptools is unable to replace itself when a activated build dependency
requires a more recent setuptools version
(it does not respect "setuptools>X" in setup_requires).


setuptools>=31 is required for setup.cfg metadata support
setuptools>=42 is required for pyproject.toml configuration support

Suggested workarounds if applicable:
 - preinstalling build dependencies like setuptools_scm before running setup.py
 - installing setuptools_scm using the system package manager to ensure consistency
 - migrating from the deprecated setup_requires mechanism to pep517/518
   and using a pyproject.toml to declare build dependencies
   which are reliably pre-installed before running the build tools
```

A simple workaround in that case is to update the `pip` and `setuptools` packages
```
pip install --upgrade pip setuptools
```
