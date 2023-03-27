import numpy as np
import os
import sys

# import boost_histogram as bh
import h5py
import matplotlib.pyplot as plt

from plotter import *
import argparse

parser = argparse.ArgumentParser(description="File name")
parser.add_argument("-n", "--Name", help="Name of h5 file", required=True)

args = parser.parse_args()
fname = args.Name

nbines = 50 + 20
Ebins = np.logspace(0, 2, nbines + 1)
Ebins = np.logspace(-0.5, 2.5, nbines + 1)
midEbins = (Ebins[:-1] + Ebins[1:]) / 2.0
cmin = 0.00000000001


# latexify()
y = None
y_pred = None
weights = None
with h5py.File(fname) as data:
    print(data["y"], data["y_pred"], data["weights"])
    y = np.copy(10 ** np.asarray(data["y"]))
    # JE = np.copy(10 **(10** np.asarray(data["JENERGY"])))
    # JS = np.copy(10 **(10** np.asarray(data["JSHOWERFIT"])))
    # JElen = np.copy(10 **(10** np.asarray(data["JSTART"])))*0.25
    # JSlen = np.copy(10 **(10** np.asarray(data["JSHOWERFIT_LENGTH"])))*0.25
    JE = np.copy(10 ** np.asarray(data["JENERGY"]))
    JS = np.copy(10 ** np.asarray(data["JSHOWERFIT"]))
    JElen = np.copy(10 ** np.asarray(data["JSTART"])) * 0.25
    JSlen = np.copy(10 ** np.asarray(data["JSHOWERFIT_LENGTH"])) * 0.25
    y_pred = np.copy(10 ** np.asarray(data["y_pred"][:, 0]))
    weights = np.copy(data["weights"])
    support = np.copy(data["support"])


gfg = get_unbiasing_factor(y, y_pred, histweights=weights)

xsize = 3 * 1.5  # 6
ysize = 2 * 1.5  # 4

recos = [y_pred, JE, JElen, JS, JSlen]
ltrueE = r"True E [GeV]"
reco_labels = [
    r"E$_{Est}$ [GeV]",
    r"E$_{JENERGY}$ [GeV]",
    r"E$_{JE-Len}$ [GeV]",
    r"E$_{JSHOWER}$ [GeV]",
    r"E$_{JS-Len}$ [GeV]",
]
reco_unbiased_labels = [
    r"E$_{Est}*s$ [GeV]",
    r"E$_{JENERGY}$*s [GeV]",
    r"E$_{JE-Len}$*s [GeV]",
    r"E$_{JSHOWER}$*s [GeV]",
    r"E$_{JS-Len}$*s [GeV]",
]
reco_name = ["Eest", "JE", "JE-Len", "JS", "JS-Len"]
colors = ["red", "blue", "green", "orange", "purple"]

xnum = 4  # len(reco_labels)
ynum = 4

mask = (np.abs(support["type"]) == 12) | (np.abs(support["type"]) == 14)
print(mask)

if not os.path.exists("performance"):
    os.makedirs("performance")
fig, ax = plt.subplot_mosaic(
    """
        ABCD
        EFGH
        KLMN
        IIJJ
        """,
    figsize=(xsize * xnum, ysize * ynum),
    constrained_layout=True,
)
# fig.set_facecolor("white")
ax0 = ["A", "B", "C", "D"]
ax1 = ["E", "F", "G", "H"]
ax2 = ["I", "J"]
ax3 = ["K", "L", "M", "N"]
for i in range(xnum):
    gfg = get_unbiasing_factor(y, recos[i], histweights=weights)
    do_2dcorr(
        ax[ax0[i]],
        y,
        recos[i],
        histweights=weights,
        x_label=ltrueE,
        y_label=reco_labels[i],
    )
    do_2dcorr(
        ax[ax1[i]],
        y,
        recos[i] * gfg(recos[i]),
        histweights=weights,
        y_label=reco_unbiased_labels[i],
        x_label=ltrueE,
    )
    # plot_unbiasing_factor(ax[ax3[i]], y, recos[i], histweights=weights)
    plot_unbiasing_factor_numu_nue(
        ax[ax3[i]], y, recos[i], histweights=weights)
    # ax[ax0[i]].set_xlim([1, 100])
    # ax[ax0[i]].set_ylim([1, 100])
    # ax[ax1[i]].set_xlim([1, 100])
    # ax[ax1[i]].set_ylim([1, 100])
for i in range(xnum):
    do_1dperf(
        ax[ax2[0]],
        y,
        recos[i],
        weights,
        x_label=ltrueE,
        y_label="Perf reco E",
        color=colors[i],
        curve_label=reco_name[i],
    )
    do_1dperf(
        ax[ax2[1]],
        y,
        recos[i] * gfg(recos[i]),
        weights,
        x_label=ltrueE,
        y_label="Perf unbiased reco E",
        color=colors[i],
        curve_label=reco_name[i],
    )

ax[ax2[0]].set_xlim([1, 100])
ax[ax2[0]].set_ylim([0, 3])
ax[ax2[0]].legend()
ax[ax2[1]].set_xlim([1, 100])
ax[ax2[1]].set_ylim([0, 3])
ax[ax2[1]].legend()
tit = fname.split("/")[-1].split(".h5")[0]

fig.suptitle(tit)
plt.savefig("performance/" + tit + ".png")
plt.savefig("performance/" + tit + ".pdf")
plt.clf()
exit()
# plt.show()
