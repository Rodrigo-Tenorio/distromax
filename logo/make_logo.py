import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gumbel_r

fig, ax = plt.subplots(figsize=(1, 1), dpi=80)
ax.set(aspect="equal", xlim=(0, 1), ylim=(0, 1))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

x = np.linspace(0, 1, 1000)

lower_gumbel = gumbel_r(loc=0.2, scale=0.1)
lg = lower_gumbel.pdf(x)
lg /= lg.max()

upper_gumbel = gumbel_r(loc=-(1 - 0.2), scale=0.1)
ug = upper_gumbel.pdf(-x)
ug /= ug.max()

lg *= 1 - upper_gumbel.pdf(-0.15)
ug *= 1 - lower_gumbel.pdf(1-0.15)

ug = 1 - ug

ax.plot(x, lg, color="slateblue")
ax.fill_between(x, 0, lg, color="slateblue")

ax.plot(x, ug, color="orange")
ax.fill_between(x, ug, 1, color="orange")

ax.text(0.1, 0.4, "dMax", fontdict={"family": "serif", "size": 15})

plt.gca().set_axis_off()

fig.savefig("distromax_logo.png", bbox_inches="tight", dpi=80, pad_inches=0)
