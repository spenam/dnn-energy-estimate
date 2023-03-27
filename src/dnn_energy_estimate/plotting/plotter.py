import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d

nbines = 50 + 20
Ebins = np.logspace(0, 2, nbines + 1)
Ebins = np.logspace(-0.5, 2.5, nbines + 1)
midEbins = (Ebins[:-1] + Ebins[1:]) / 2.0
cmin = 0.00000000001

def latexify(fig_width=None, ratio = (np.sqrt(5)-1.0)/2.0 ,fig_height=None):
    
    '''Make plots have latex font and size equal to text'''
    
    fig_width_pt = 412.56497     # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.0#.27  # Convert pt to inch

    if fig_width is None:
        fig_width = fig_width_pt*inches_per_pt
    elif fig_height is None:
        fig_width = fig_width*fig_width_pt*inches_per_pt

    if fig_height is None:
        #golden_mean = 0.8
        golden_mean = ratio# Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 10.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
                 "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES
    matplotlib.rc('axes',edgecolor="#595959")
    params = {
          #'backend': 'ps',
          #'text.latex.preamble': ['\usepackage{gensymb}'],
          'axes.labelsize': 12, # fontsize for x and y labels (was 10)
          'axes.titlesize': 11,
          'font.size':       13, # 11 is footnotesize, 12 captionsize
          'legend.fontsize': 10, # was 10
          'xtick.labelsize': 9,
          'ytick.labelsize': 9,
          #'text.usetex': True,
          #'pgf.texsystem': "pdflatex",
          'figure.figsize': [fig_width,fig_height],
          'font.family': "DejaVu Sans",
          'font.serif': ['Computer Modern Roman'],  # blank entries should cause plots to inherit fonts from the document
          'font.sans-serif': ['Computer Modern Sans serif']
        }
        
    matplotlib.rcParams.update(params)


def get_percentiles(x_array, y_array, histweights):
    hbin_dE_ = np.zeros(nbines)
    hbin_dE_15_ = np.zeros(nbines)
    hbin_dE_85_ = np.zeros(nbines)
    whereisE = np.digitize(x_array, Ebins)  # -binstep
    Ebins_ = np.logspace(-0.5, 2.5, nbines*10 + 1)
    lower = 0.16
    upper = 0.84
    for i in range(nbines):
        mini_mask = whereisE == i + 1
        vals = y_array[mini_mask]
        vals_mask = (vals < 50000) & (vals > 0.001)
        # histo, bines_ = np.histogram(
        histo, bines_ = np.histogram(
            vals[vals_mask],
            bins=Ebins_, 
            weights=(histweights[mini_mask])[vals_mask],
        )
        mids_bins = 0.5 * (bines_[1:] + bines_[:-1])
        total = 0
        median_index = (sum(histo)) / 2
        quant_15 = (sum(histo)) * lower
        quant_85 = (sum(histo)) * upper
        median_histo = 0
        quant_15_histo = 0
        quant_85_histo = 0

        for jj, item in enumerate(histo):
            total += item
            # print(total, median_index)
            if (total > quant_15) & (quant_15_histo == 0):
                quant_15_histo = mids_bins[jj]
            if (total > median_index) & (median_histo == 0):
                median_histo = mids_bins[jj]
            if (total > quant_85) & (quant_85_histo == 0):
                quant_85_histo = mids_bins[jj]
                break
        total = 0
        hbin_dE_[i] = median_histo
        hbin_dE_15_[i] = quant_15_histo
        hbin_dE_85_[i] = quant_85_histo
    return hbin_dE_, hbin_dE_15_, hbin_dE_85_



def get_unbiasing_factor(x_array, y_array, histweights):
    hbin_dE, hbin_dE_15, hbin_dE_85 = get_percentiles(
        y_array, x_array, histweights=histweights
    )
    #factor_function = make_interp_spline(midEbins, hbin_dE / midEbins, k=1)
    factor_function = interp1d(midEbins, hbin_dE / midEbins, fill_value = (1,1), bounds_error=False)
    return factor_function


def plot_unbiasing_factor(ax0, x_array, y_array, histweights):
    gfg0 = get_unbiasing_factor(x_array, y_array, histweights)
    #xnew = np.logspace(np.log10(y_array.min()), np.log10(y_array.max()), 10000)
    xnew = Ebins
    ynew = gfg0(xnew)
    ax0.plot(xnew, ynew) 
    ax0.set_ylabel(r"$s$")
    ax0.set_xscale('log') 
    ax0.set_xlabel(r"$E_{Reco}$ [GeV]") 
    #ax0.set_xticks([1, 10, 100,1000])  


def plot_unbiasing_factor_numu_nue(ax0, x_array, y_array, histweights):
    gfg0 = get_unbiasing_factor(x_array, y_array, histweights)
    #xnew = np.logspace(np.log10(y_array.min()), np.log10(y_array.max()), 10000)
    xnew = Ebins
    ynew = gfg0(xnew)
    ax0.plot(xnew, ynew) 
    ax0.set_ylabel(r"$s$")
    ax0.set_xscale('log') 
    ax0.set_xlabel(r"$E_{Reco}$ [GeV]") 
    #ax0.set_xticks([1, 10, 100,1000])  

def do_2dcorr(
    ax0,
    x_array,
    y_array,
    histweights,
    do_percentile=1,
    x_label="x_label",
    y_label="y_label",
):
    ax0.hist2d(x_array, y_array, weights=histweights, bins=[Ebins, Ebins], cmin=cmin)
    ax0.plot(midEbins, midEbins, color="gray")
    if do_percentile == 1:
        hbin_dE_, hbin_dE_15_, hbin_dE_85_ = get_percentiles(
            x_array, y_array, histweights=histweights
        )
        ax0.plot(midEbins, hbin_dE_, color="r")
        ax0.plot(midEbins, hbin_dE_15_, color="r", linestyle="--")
        ax0.plot(midEbins, hbin_dE_85_, color="r", linestyle="--")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label)


def do_2dperf(
    ax0,
    x_array,
    y_array,
    histweights,
    do_percentile=1,
    x_label="x_label",
    y_label="y_label",
):
    ax0.hist2d(
        x_array,
        np.abs(y_array - x_array) / x_array,
        weights=histweights,
        bins=[Ebins, np.linspace(0, 3, nbines)],
        cmin=cmin,
    )
    if do_percentile == 1:
        hbin_dE_, hbin_dE_15_, hbin_dE_85_ = get_percentiles(
            x_array, np.abs(y_array - x_array) / x_array, histweights=histweights
        )
        ax0.plot(midEbins, hbin_dE_, color="r")
        ax0.plot(midEbins, hbin_dE_15_, color="r", linestyle="--")
        ax0.plot(midEbins, hbin_dE_85_, color="r", linestyle="--")
    ax0.set_xscale("log")
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label)


def do_1dperf(
    ax0,
    x_array,
    y_array,
    histweights,
    do_percentile=1,
    x_label="x_label",
    y_label="y_label",
    color="k",
    curve_label="curve_label",
):
    a0 = 0.1
    hbin_dE_, hbin_dE_15_, hbin_dE_85_ = get_percentiles(
        x_array, np.abs(y_array - x_array) / x_array, histweights=histweights
    )
    ax0.plot(midEbins, hbin_dE_, label=curve_label, color=color)
    ax0.fill_between(midEbins, hbin_dE_15_, hbin_dE_85_, alpha=a0, color=color)
    ax0.set_xscale("log")
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label)

