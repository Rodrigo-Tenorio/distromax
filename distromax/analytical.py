import logging

import numpy as np
from scipy.special import gamma


class AnalyticalGammaToGumbel:
    """
    Compute relevant quantities related to the Gumbel distribution arising from the
    maximum over an ensemble of iid Gamma-distributed random variables (which includes
    the chi-squared distribution as a special case).
    """

    def __init__(self, dofs=None, shape=None, scale=None, rate=None):
        """
        The single-variable probability distribution can either be described as a
        chi-squared distribution with `dofs` degrees of freedom,
        a Gamma distribution with shape and scale parameters (`shape`, `scale`),
        or a Gamma distribution with shape and rate parameters (`shape`, `rate`).

        A chi-squared distribution with `dofs` degrees of freedom corresponds to a
        Gamma distribution with `shape=dofs/2` and `scale=2`, `rate=1/2`. 

        Parameters
        ----------
        dofs: float, optional
            Degrees of freedom of the underlying chi-squared distribution.
            Authomatically implies `shape=dofs/2` and `scale=2`.
            This parameter is incompatible with `shape`, `scale`, and `rate`.
        shape: float, optional
            Shape parameter of the underlying Gamma distribution.
            This parameter must be given jointly with `scale` or `rate`
            and is incompatible with `dofs`.
        scale: float, optional
            Scale parameter of the underlying gamma distribution 
            (inverse of the rate parameter).
            This parameter must be given jointly with `shape` and is
            incompatible with `dofs` or `rate`.
        rate: float, optional
            Rate parameter of the underlying gamma distribution 
            (inverse of the scale parameter).
            This parameter must be given jointly with `shape` and is
            incompatible with `dofs` or `scale`.
        """
        
        # Check the validity of the input
        if dofs:
            if shape:
                raise ValueError("Only one out of dofs and shape can be given")
            shape = dofs / 2
            scale = 2.
        elif shape:
            if not (bool(scale) ^ bool(rate)):
                raise ValueError("Only one out of scale and rate can be given")
            elif rate:
                scale = 1 / rate
        else:
            raise ValueError("Invalid input parameters: "
                    "Please enter dofs, (scale, shape) or (scale, rate)")
    
        self._set_shape_scale(shape=shape, scale=scale)

    def _set_shape_scale(self, shape, scale):
        self.shape = shape
        self.scale = scale

    def get_gumbel_loc_scale(self, n, improved=True):
        """
        Compute the theoretical estimation of the location and scale parameters of the
        Gumbel distribution resulting from maximizing over a batch of `n` independent
        random variables following the distribution specified by this class.

        We provide two implementations of said parameters, selected by
        switching the `improved` flag.

        If improved is falsy, location and scale parameters are taken from table 3.4.4
        in "Modelling Extremal Events for Insurance and Finance", P. Embrechts,
        C. Klüppelberg, T. Mikosch (1997) https://doi.org/10.1007/978-3-642-33483-2.

        If improved is truthy (recommended), location and scale parameters are taken
        from Eqs. (32) and (33) in  Gasull, López-Salcedo, and Utzet, TEST 24, 714–733
        (2015) https://doi.org/10.1007/s11749-015-0431-9 .

        The improved option returns better estimates for n < 1e6 (and possibly beyond).

        Parameters
        ----------
        n: int
            Number of random variables within the batch being maximized.
        improved: bool
            If True (recommended), apply the improved loc and scale parameters. See
            the main docstring for details.

        Returns
        -------+
        loc: float
            Location parameter of the resulting Gumbel distribution.
        scale: float
            Scale parameter of the resulting Gumbel distribution.
        """

        logn = np.log(n)
        gamma_shape = gamma(self.shape)

        if improved:
            logn_gamma = np.log(n / gamma_shape)
            shape_m1 = self.shape - 1
            Bn = logn_gamma + shape_m1 * np.log(shape_m1)
            logBn = np.log(Bn)

            loc = self.scale * (
                logn_gamma
                + shape_m1 * logBn
                + (shape_m1 * shape_m1 * (logBn - np.log(shape_m1)) + shape_m1) / Bn
            )
            scale = (self.scale * loc * (self.scale * shape_m1 + loc)) / (
                loc * loc - self.scale * self.scale * shape_m1 * (self.shape - 2)
            )

        else:
            loc = self.scale * (
                logn + (self.shape - 1) * np.log(logn) + np.log(gamma_shape)
            )
            scale = self.scale

        return loc, scale

    def get_gumbel_mean_std(self, n, improved=True):
        """
        Compute the theoretical estimation of the mean and standard deviation parameters
        of the Gumbel distribution resulting from maximizing over a batch of `n` independent
        random variables following the distribution specified by this class.

        These results computed from the `loc` and `scale` parameters returned by
        `self.gumbel_mean_std`. See the docstring therein for a further explanation of the
        parameters.

        Parameters
        ----------
        n: int
            Number of random variables within the batch being maximized.
        improved: bool
            If True (recommended), apply the improved loc and scale parameters. See
            the main docstring for details.

        Returns
        -------
        mean: float
            Mean of the Gumbel distribution, computed as `loc + euler_gamma * scale`,
            where `euler_gamma=0.5772...` represents the Euler-Mascheroni constant
            (sequence A001620 in the OEIS).
        std: float
            Standard deviation of the Gumbel distribution,
            computed as `pi/sqrt(6) * scale`.
        """
        loc, scale = self.get_gumbel_loc_scale(n=n, improved=improved)
        mean = loc + np.euler_gamma * scale
        std = np.pi / np.sqrt(6) * scale
        return mean, std
