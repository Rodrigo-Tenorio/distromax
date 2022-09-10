import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import windows

import distromax 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 22

logger = distromax.set_up_logger()

f""""
Example 2: :math:`\chi^2` distribution with an outlier

Similar setup as in Example 1, this time including an outlier in a narrow frequency band.
 
We generate a dataset of :math:`\chi^2` draws and introduce an outlier as 
samples from a non-central chi2 distribution. 
Then we notch that band using the procedure outlined in Appendix B of [0]
and apply `distromax` to estimate the probability distribution of the loudest candidate.

We compare the resulting distirbution (with and without notching) to the ground truth.

The example is framed as a narrow band CW search in which the 2F statistic
(chi-squared with four degrees of freedom in Gaussian noise) is used to 
evaluate a template bank over (f0, f1) with a narrow-band outlier around a certain f0.

[0] R. Tenorio, L. M. Modafferi, D. Keitel, A. M. Sintes
"""

basename = os.path.basename(sys.argv[0])[:-3]
outdir = os.path.join(sys.path[0], basename)
os.makedirs(outdir, exist_ok=True)

logger.info(f"Running example {basename}")
logger.info(f"Output will be saved into {outdir}")

# Create samples [template bank over (f0, f1) to include an outlier].
points = 1000
total_points = points * points

# Give units to both axis and include a narrow band outlier
f0 = np.linspace(100, 100.1, points)
f1 = np.linspace(0, 1, points)
f0, f1 = np.meshgrid(f0, f1)

outlier_start_f0 = 100.04
outlier_width_Hz = 5e-3
outlier_max_SNR = 100

f0 = f0.flatten()
f1 = f1.flatten()

## Samples without the outlier
raw_twoF = stats.chi2(df=4).rvs(total_points)

outlier_range = np.logical_and(f0 > outlier_start_f0, 
                               f0 < outlier_start_f0 + outlier_width_Hz) 
outlier_f0 = np.sort(np.unique(f0[outlier_range]))
mismatch_window = windows.bartlett(outlier_f0.shape[0])

outlier_twoF = raw_twoF.copy()
for ind, f0_bin in enumerate(outlier_f0):
    bin_mask = f0 == f0_bin
    outlier_twoF[bin_mask] = (
        stats.ncx2(df=4, nc=outlier_max_SNR * mismatch_window[ind])
        .rvs(bin_mask.sum())
    )

## Samples with the outlier
data = np.vstack([f0, f1, outlier_twoF]).T

# Apply distromax with notching
num_batches = 5000
batch_size = total_points // num_batches
bmgno = distromax.BatchMaxGumbelNotchingOutliers(data, batch_size=batch_size)

# Plot samples with outliers
fig, ax = plt.subplots(figsize=(16, 10))
ax.set(xlabel=r"$f_{0}$ [Hz]", ylabel=r"$2\mathcal{F}$", title="Notching outliers: Final iteration")
ax.grid()

bg_mask = np.in1d(data[:, -1], bmgno.samples)
ax.plot(data[bg_mask, 0], data[bg_mask, -1], 'o', color="slateblue", rasterized=True,
        alpha=0.4, label="Background samples", markerfacecolor="none")
ax.plot(data[~bg_mask, 0], data[~bg_mask, -1], '.', color="aqua", rasterized=True,
        alpha=0.4, label="Notched samples", markerfacecolor="none")

ax.plot(bmgno.f0, bmgno.max_at_f0, "d", color="orange", 
        label=r"Max per $f_{0}$", markerfacecolor="none")
mask = bmgno.max_at_f0 > bmgno.threshold
ax.plot(bmgno.f0[mask], bmgno.max_at_f0[mask], "x", color="red", 
        label=r"Notched $f_0$ bins in this iteration", markerfacecolor="none")
for f0_bin, max_2F in zip(bmgno.f0[mask], bmgno.max_at_f0[mask]):
    ax.vlines(f0_bin, ymin=-1, ymax=max_2F, color="red", zorder=10)

ax.axhline(bmgno.stopping_det_stat, ls=":", color="gray", label="Stopping det. stat.")
ax.axhline(bmgno.threshold, color="gray", label="Threshold")

ax.legend()

fig.savefig(os.path.join(outdir, "Samples.pdf"), bbox_inches="tight")
logger.info("Plot of samples: Success!")

# Plot histogram of samples with and withouth outlier
fig, ax = plt.subplots(figsize=(16, 10))
ax.grid()
ax.set(xlabel=r"$2\mathcal{F}$", ylabel="PDF", yscale="log")
ax.hist(raw_twoF, density=True, histtype="step", ls="--",
        bins="auto", color="black", label="Ground truth");
ax.hist(outlier_twoF, density=True, histtype="step",
        bins="auto", color="blue", label="Ground truth + Outlier");
ax.hist(bmgno.samples, density=True, histtype="step", ls="-",
        bins="auto", color="red", label="Notched Outlier");

ax.axvline(bmgno.stopping_det_stat, ls=":", color="gray", label="Stopping det. stat.")
ax.axvline(bmgno.threshold, color="gray", label="Threshold")
ax.legend(loc="upper right")

fig.savefig(os.path.join(outdir, "HistogramSamples.pdf"), bbox_inches="tight")
logger.info("Plot of sample histograms: Success!")

raw_max = np.random.permutation(raw_twoF).reshape((-1, batch_size)).max(axis=1)
raw_max_gumbel = stats.gumbel_r(*stats.gumbel_r.fit(raw_max))

raw_max_out = np.random.permutation(outlier_twoF).reshape((-1, batch_size)).max(axis=1)
raw_max_out_gumbel = stats.gumbel_r(*stats.gumbel_r.fit(raw_max_out))

# Plot batchmax distributions
fig, ax = plt.subplots(figsize=(16, 10))
ax.grid()
ax.set(xlabel=r"Max $2\mathcal{F}$", ylabel="PDF")
ax.hist(raw_max, density=True, histtype="step", bins="auto", ls="--",
        color="black", label=r"Ground truth ({:.3f}, {:.3f})".format(*raw_max_gumbel.args))
ax.hist(raw_max_out, density=True, histtype="step", bins="auto",
        color="blue", label=r"Ground truth + Outlier ({:.3f}, {:.3f})".format(*raw_max_out_gumbel.args))
ax.hist(bmgno.batch_max, density=True, histtype="step", bins="auto", ls="-",
        color="red", label=r"distromax notching outliers ({:.3f}, {:.3f})".format(*bmgno.gumbel.args))

ax.axvline(bmgno.stopping_det_stat, ls=":", color="gray", label="Stopping det. stat.")
ax.axvline(bmgno.threshold, color="gray", label="Threshold")

ax.legend(loc="upper right")

fig.savefig(os.path.join(outdir, "BatchmaxSamples.pdf"), bbox_inches="tight")
logger.info("Plot of sample histograms: Success!")

mean_gt = raw_max_gumbel.args[0] + np.euler_gamma * raw_max_gumbel.args[1] 
mean_distro = bmgno.gumbel.args[0] + np.euler_gamma * bmgno.gumbel.args[1] 

# Plot CDF of 
fig, ax = plt.subplots(figsize=(16, 10))
ax.grid()
ax.set(xlabel="Ground truth CDF", ylabel="Outlier CDF", xlim=(0, 1), ylim=(0, 1))
gt_pdf  = stats.rv_histogram(np.histogram(raw_max, bins="auto"))

x = np.linspace(0, 100, 10000)
ax.plot(raw_max_gumbel.cdf(x), raw_max_out_gumbel.cdf(x), color="blue", label="Ground truth + Outlier")
ax.plot(raw_max_gumbel.cdf(x), bmgno.gumbel.cdf(x), color="red", label="distromax notching outliers")
ax.plot([0, 1], [0, 1], color="gray", ls="--")

ax.legend()
fig.savefig(os.path.join(outdir, "CDFComparison.pdf"), bbox_inches="tight")
logger.info("Plot CDF comparison: Success!")

logger.info("GT vs. distromax relative difference:")
logger.info("    Location parameter: {:.2f}%".format(100 * (bmgno.gumbel.args[0]/raw_max_gumbel.args[0] - 1)))
logger.info("    Scale parameter: {:.2f}%".format(100 * (bmgno.gumbel.args[1]/raw_max_gumbel.args[1] - 1)))

