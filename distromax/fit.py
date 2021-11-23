import logging
import numpy as np
from scipy import signal, stats
from skimage.filters import threshold_minimum
from tqdm import tqdm, trange


class BatchMaxGumbel:
    def __init__(self, samples, batch_size=None, shuffle=True, shuffle_seed=None):
        """
        Basic implementation of the method described in [0].
        Batch a given set of samples, retrieve batch-wise maxima and fit a Gumbel distribution.

        [0] R. Tenorio, L. M. Modafferi, D. Keitel, A. M. Sintes

        Parameters
        ----------
        samples: array-like 1D or 2D
            Samples to which a Gumbel distribution will be fitted.
            If 2D, samples should be located along the last column (-1 on the last index).

        batch_size: int
            Number of samples per batch.

        shuffle: bool
            Whether to shuffle the samples before batching (recommended) or not.
            Shuffling uses numpy's permutation method from numpy.random.default_rng.

        shuffle_seed: int, optional
            Seed to initialize numpy.random.default_rng
        """

        dimension = getattr(samples, "ndim", 0)
        if dimension == 2:
            samples = samples[:, -1]
        elif dimension > 2:
            raise ValueError(
                "Input samples array is > 2D. "
                "Please, slice out samples before using this class"
            )

        logging.info("Starting distromax...")
        self.set_samples(samples)
        logging.info("Samples successfully set")
        self.batch_max_samples(batch_size, shuffle, shuffle_seed)
        logging.info("Batchmax samples successfully retrieved")
        self.fit()
        logging.info("Gumbel distribution successfully fitted")

    def set_samples(self, samples):

        self.samples = samples
        histogram, x = np.histogram(samples, density=True, bins="auto")
        self.dx = x[1] - x[0]
        self.x = 0.5 * (x[1:] + x[:-1])
        self.histogram = histogram

    def batch_max_samples(self, batch_size, shuffle, shuffle_seed):

        if not batch_size:
            self.batch_max = self.samples.copy()
        else:
            if shuffle:
                rng = np.random.default_rng(shuffle_seed)
                samples_to_batch = rng.permutation(self.samples)
            else:
                samples_to_batch = self.samples

            self.batch_max = np.maximum.reduceat(
                samples_to_batch, np.arange(0, len(samples_to_batch), batch_size)
            )

    def fit(self, samples=None):
        samples = samples if samples is not None else self.batch_max
        self.gumbel = stats.gumbel_r(*stats.gumbel_r.fit(samples))

    def max_propagation(self, num_batches):
        """
        Maximize the fitted Gumbel distribution over `num_batches` realizations
        using the analytical formula. The output is placed so as to be consistent
        with what `scipy.gumbel_r` expects.
        """
        loc, scale = self.gumbel.args
        propagated_params = (loc + scale * np.log(num_batches), scale)
        self.propagated_gumbel = stats.gumbel_r(*propagated_params)
        return propagated_params


class BatchMaxGumbelWithCleaning(BatchMaxGumbel):
    """
    Abstract implementation of the cleaning procedure described in Appendix B of [0].
    Parameter's meaning is implementation dependent.
    See specific implementations below.

    [0] R. Tenorio, L. M. Modafferi, D. Keitel, A. M. Sintes

    Parameters
    ----------
    num_iterations: int
        Number of times the notching procedure will be applied.
    stopping_det_stat: float
        If the detection statistic threshold scores below this value, no notching will take place.
    stopping_quantile: float
        Alternative way of specifying `stopping_det_stat` as a quantile of the detection statistic.
    threshold: callable or float
        Threshold to perform cleaning. By default, skimage.threshold_minim is used.
    """

    _num_iterations = 0
    _stopping_det_stat = None
    _stopping_quantile = None
    _threshold = None

    def __init__(
        self,
        samples,
        num_iterations=None,
        stopping_det_stat=None,
        stopping_quantile=None,
        threshold=None,
        **kwargs,
    ):

        if (
            not getattr(samples, "ndim", None)
            or (samples.ndim != 2)
            or (samples.shape[1] < 2)
        ):
            raise ValueError(
                "`samples` should be a 2D array containing frequency values in [:, 0] and "
                "corresponding statistic values in [:, -1]"
            )

        num_iterations = (
            num_iterations if num_iterations is not None else self._num_iterations
        )

        if (
            (threshold is not None)
            and (not callable(threshold))
            and (num_iterations > 1)
        ):
            logging.info(
                "Threshold was manually set to a value but `num_iterations > 1`. "
                "Setting num_iterations to 1."
            )
            num_iterations = 1

        self.notching_iterations = 0

        notched_samples = samples
        for iteration in trange(
            num_iterations,
            desc="Notching iterations",
            unit=" notches",
            dynamic_ncols=True,
        ):
            self.notching_iterations += 1
            notched_samples = self.notch_outliers(
                samples=notched_samples,
                stopping_det_stat=stopping_det_stat or self._stopping_det_stat,
                threshold=threshold or self._threshold,
                stopping_quantile=stopping_quantile or self._stopping_quantile,
            )
            if not notched_samples.size:
                raise RuntimeError(
                    "It appears the last notching iteration completely obliterated your samples.\n"
                    f"Current configuration is `num_iterations = {num_iterations}`, "
                    "you probably should reduce that number."
                )

        super().__init__(samples=notched_samples[:, -1], **kwargs)

    def notch_outliers(
        self,
        samples,
        stopping_quantile,
        stopping_det_stat,
        threshold,
    ):

        # Can't stress this enough: always sort by frequency
        samples = samples[samples[:, 0].argsort()]
        self.f0, self.unique_indices = np.unique(samples[:, 0], return_index=True)
        self.max_at_f0 = np.maximum.reduceat(samples[:, -1], self.unique_indices)

        self.stopping_quantile = stopping_quantile
        self.stopping_det_stat = stopping_det_stat

        if callable(threshold):
            self.threshold = threshold(self.max_at_f0)
        elif threshold is None:
            try:
                self.threshold = threshold_minimum(self.max_at_f0)
            except RuntimeError as e:
                tqdm.write(
                    f"Iteration {self.notching_iterations}: {e}, threshold goes to -np.inf"
                )
                self.threshold = -np.inf
        else:
            self.threshold = threshold

        return self.remove_samples(samples)


class BatchMaxGumbelNotchingOutliers(BatchMaxGumbelWithCleaning):
    """
    Basic implementation of the notching algorithm described in Appendix B of [0].

    This class operates on the distribution of the frequency-bin-wise maxima to compute thresholds
    and notch loud frequency bands.

    [0] R. Tenorio, L. M. Modafferi, D. Keitel, A. M. Sintes
    """

    _num_iterations = 5
    _stopping_quantile = 0.8

    def remove_samples(self, samples):

        self.stopping_det_stat = self.stopping_det_stat or np.quantile(
            self.max_at_f0, self.stopping_quantile
        )

        if self.threshold < self.stopping_det_stat:
            tqdm.write(
                f"Iteration {self.notching_iterations}: "
                f"Threshold is below {100 * self.stopping_quantile}% quantile, skipping notching..."
            )
            return samples

        notched_bands = self.max_at_f0 > self.threshold
        notched_bands_index = np.nonzero(notched_bands)[0]

        left_index_notch = notched_bands_index[
            np.diff(notched_bands_index, prepend=-1) > 1
        ]

        right_index_notch = (
            notched_bands_index[
                np.diff(notched_bands_index, append=notched_bands_index.max() + 2) > 1
            ]
            + 1
        )

        indices_to_delete = np.concatenate(
            [
                np.arange(self.unique_indices[left], self.unique_indices[right])
                for left, right in zip(left_index_notch, right_index_notch)
            ]
        )

        tqdm.write(
            f"Iteration {self.notching_iterations}: "
            f"Threshold={self.threshold:.2f}, "
            f"{100 * self.stopping_quantile}% foreground quantile at {self.stopping_det_stat:.2f}, "
            "notching {}/{} ({:.2f}%) samples in {}/{} ({:.2f}%) frequency bins".format(
                indices_to_delete.shape[0],
                samples.shape[0],
                100 * indices_to_delete.shape[0] / samples.shape[0],
                notched_bands.sum(),
                self.max_at_f0.shape[0],
                100 * notched_bands.sum() / self.max_at_f0.shape[0],
            )
        )

        return np.delete(samples, indices_to_delete, axis=0)
