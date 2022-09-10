import logging
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import distromax 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 22

logger = distromax.set_up_logger()

r""""
# Example 1: :math:`\chi^2` distribution
 
:math:`\chi^2` distributions fall under Gumbel distribution's domain of attraction
and appear very frequently in gravitational wave data analysis.
 
We generate a dataset of :math:`\chi^2` samples and apply `distromax` to estimate 
the probability distribution of the loudest candidate.
 
The resulting batchmax distribution is compared to the theoretical distribution.
"""

basename = os.path.basename(sys.argv[0])[:-3]
outdir = os.path.join(sys.path[0], basename)
os.makedirs(outdir, exist_ok=True)

logger.info(f"Running example {basename}")
logger.info(f"Output will be saved into {outdir}")

# Create samples
dofs = 4
total_points = 2000000
data  = stats.chi2(df=dofs).rvs(total_points)

# Apply distromax
num_batches = 10000
batch_size = total_points // num_batches
bmg = distromax.BatchMaxGumbel(data, batch_size=batch_size)
bmg.max_propagation(num_batches=num_batches)

# Compute theoretical parameters of the batchmax distribution
th_loc, th_scale = distromax.analytical.AnalyticalGammaToGumbel(dofs=dofs).get_gumbel_loc_scale(batch_size)

logger.info("Data successfully generated, starting to plot results")

# Plot samples and 95% credible region of the loudest candidate
fig, ax = plt.subplots(figsize=(16, 10))
ax.set(xlabel="Sample index", ylabel=r"$\chi^2_{4}$-distributed detection statistic")
ax.grid()

ax.plot(data, 'o', rasterized=True, color="slateblue",
        alpha=0.4, label=f"{total_points:.2g} Background samples", 
        markerfacecolor="none")

loudest_mean = bmg.propagated_gumbel.mean()
loudest_credible_interval = bmg.propagated_gumbel.interval(0.95)
ax.axhline(loudest_mean, color="red", ls="-", label="Expected loudest candidate")
ax.axhspan(*loudest_credible_interval, color="red", alpha=0.3, label=r"95% Credibility")

ax.legend(loc="lower right")

fig.savefig(os.path.join(outdir, "SamplesAndExpectedMax.pdf"), bbox_inches="tight")
logger.info("Plot of samples and expected maxima: Success!")

# Plot batchmax samples and compare the obtained Gumbel distribution to the theoretical one
fig, ax = plt.subplots(figsize=(16, 10))
ax.grid()
ax.set(xlabel=r"$\chi^2_{4}$-distributed detection statistic", 
       ylabel="PDF", yscale="linear")

ax.hist(bmg.samples, density=True, histtype="step", ls="--",
        bins="auto", color="blue", label=f"{total_points:.2g} Background samples");

ax.hist(bmg.batch_max, density=True, histtype="step", bins="auto", ls="-",
        color="blue", label=f"batchmax samples - {batch_size} samples per batch")

x = np.linspace(0.9 * bmg.x.min(), 1.1 * bmg.x.max(), 1000)
ax.plot(x, bmg.gumbel.pdf(x), color="red", 
        label="batchmax Gumbel fit ({:.2f} {:.2f})".format(*bmg.gumbel.args))
ax.plot(x, stats.gumbel_r(th_loc, th_scale).pdf(x), color="black", ls="--",
        label="Theoretical Gumbel ({:.2f} {:.2f})".format(th_loc, th_scale))

ax.legend()

fig.savefig(os.path.join(outdir, "BatchmaxAndTheoretical.pdf"), bbox_inches="tight")
logger.info("Plot of batchmax distribution and comparison to theoretical: Success!")

logger.info(f"Location relative error: {100 * (bmg.gumbel.args[0]/th_loc - 1):.2f} %")
logger.info(f"Scale relative error: {100 * (bmg.gumbel.args[1]/th_scale - 1):.2f} %")

