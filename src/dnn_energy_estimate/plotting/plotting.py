import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


#########################################
#########################################
#######         AVG_STD_           ######
#########################################
#########################################


def weighted_avg_and_std(X0, values, weights, is_err=0):
    """
    Return the weighted average and standard deviation.
    """
    if sum(weights == 0) == len(values):
        average = 0
        variance = 0
    else:
        # average = values[weights==weights.max()][0]# np.average(values, weights=weights)
        average = np.average(values, weights=weights)
        # Fast and numerically precise
        if is_err == 1:
            variance = np.average((values - average) ** 2, weights=weights)
        else:
            variance = np.average((values - X0) ** 2, weights=weights)  # RMS
        # print(variance)
    return average, np.sqrt(variance)


def avg_and_std(h, x, y, err=0):
    """
    Function to define average and standard deviation of histogram
    """
    n, _ = h.shape
    # 3 columns with: avg xbin, weighted_avg, and weighted_std
    out = np.zeros((n, 3))
    # assuming uniform/constant bining
    # y_shift = (y[1]-y[0])/2
    # x_shift = (x[1]-x[0])/2
    # X = x[0:-1] + x_shift
    # Y = y[0:-1] + y_shift
    X = (x[0:-1] + x[1:]) / 2.0
    Y = (y[0:-1] + y[1:]) / 2.0
    # print("This is X: ", X, " This is Y: ", Y)

    out[:, 0] = X
    h[np.isnan(h)] = 0
    for i in range(n):
        avg, std = weighted_avg_and_std(X[i], Y, weights=h[i, :], is_err=err)
        out[i, 1] = avg
        out[i, 2] = std
    return out


#########################################
#########################################
#######         PLOTTING           ######
#########################################
#########################################


def normal_2d_compare(
    X_test0,
    y_test0,
    y_pred0,
    mycmap="BuPu",
    name_suffix="draft",
    Gcols=0,
    Jshf=0,
    n_features=0,
    LR=0.1,
    wstr="w",
    save=True,
):
    cmin = 1e-10
    n_features = str(n_features)
    if len(Gcols) > 0:
        if Jshf != 0:
            dims = [2, 5]
        else:
            dims = [2, 4]
    else:
        if Jshf != 0:
            dims = [2, 4]
        else:
            dims = [2, 3]

    col = 0

    xlim = [1, 100]
    ylim = [1, 100]
    xlim_log = np.log10(xlim)
    ylim_log = np.log10(ylim)

    fig, ax = plt.subplots(
        dims[0], dims[1], figsize=(6 * dims[1], 4 * dims[0]), constrained_layout=True
    )
    nbines = 50
    fig.set_facecolor("white")
    # fig.tight_layout(pad =3.0)

    ###### ENERGY ESTIMATED BY THE NN ######

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
    # im = ax[0][0].hist2d(10**y_test0,10**y_pred0, bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim], weights = X_test0['weights'])
    im = ax[0][0].hist2d(
        10**y_test0,
        10**y_pred0,
        bins=[x_bins, y_bins],
        range=[xlim, ylim],
        weights=X_test0["weights"],
        cmap=mycmap,
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2])
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][col].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
    ax[0][col].set_xlabel(r"$E_{True}$ [GeV]")
    ax[0][col].set_ylabel(r"$E_{Est}$ [GeV]")
    with open(
        "plots/" + name_suffix + "/" + wstr + "/" + str(LR) + "/Eest.pkl", "wb"
    ) as f:
        pkl.dump(10**y_test0, f)
        pkl.dump(X_test0["weights"], f)
    with open(
        "plots/" + name_suffix + "/" + wstr +
            "/" + str(LR) + "/ETrue.pkl", "wb"
    ) as f:
        pkl.dump(10**y_pred0, f)
    ax[0][col].set_xscale("log")
    ax[0][col].set_yscale("log")
    ax[0][col].set_xticks([1, 10, 100])
    ax[0][col].set_yticks([1, 10, 100])
    ax[0][col].set_xlim(xlim)
    ax[0][col].set_ylim(ylim)
    fig.colorbar(im[3], ax=ax[0][col])
    # plt.show()

    ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
    ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].set_ylabel(r"$\sigma_{Est}/E_{True}$")
    ax[1][col].set_xscale("log")
    ax[1][col].set_xlabel(r"$E_{True}$ [GeV]")
    # ax[1][col].set_yscale("log")
    ax[1][col].set_xticks([1, 10, 100])
    # ax[1][col].set_yticks([1, 10, 100])
    ax[1][col].set_xlim(xlim)
    ax[1][col].set_ylim(0, 4)

    ###### ENERGY ESTIMATED BY JENERGY ######

    col += 1

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
    # im = ax[0][col].hist2d(10**y_test0,X_test0['JENERGY_ENERGY'], bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
    im = ax[0][col].hist2d(
        10**y_test0,
        10 ** X_test0["JENERGY_ENERGY"],
        bins=[x_bins, y_bins],
        range=[xlim, ylim],
        weights=X_test0["weights"],
        cmap=mycmap,
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2])
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][col].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
    ax[0][col].set_xlabel(r"$E_{True}$ [GeV]")
    ax[0][col].set_ylabel(r"$E_{JENERGY}$ [GeV]")
    with open(
        "plots/" + name_suffix + "/" + wstr +
            "/" + str(LR) + "/JENERGY.pkl", "wb"
    ) as f:
        pkl.dump(10 ** X_test0["JENERGY_ENERGY"], f)
        pkl.dump(X_test0["weights"], f)
    ax[0][col].set_xscale("log")
    ax[0][col].set_yscale("log")
    ax[0][col].set_xticks([1, 10, 100])
    ax[0][col].set_yticks([1, 10, 100])
    ax[0][col].set_xlim(xlim)
    ax[0][col].set_ylim(ylim)
    fig.colorbar(im[3], ax=ax[0][col])
    # plt.colorbar(im, ax=ax[col])

    ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
    ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].set_ylabel(r"$\sigma_{JENERGY}/E_{True}$")
    ax[1][col].set_xscale("log")
    ax[1][col].set_xlabel(r"$E_{True}$ [GeV]")
    # ax[1][1].set_yscale("log")
    ax[1][col].set_xticks([1, 10, 100])
    # ax[1][col].set_yticks([1, 10, 100])
    ax[1][col].set_xlim(xlim)
    ax[1][col].set_ylim(0, 4)
    # ax[1][1].set_ylim(1,100)

    ###### ENERGY ESTIMATED BY JLENGTH ######

    col += 1

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
    # im = ax[0][col].hist2d(10**y_test0,X_test0['JSTART_LENGTH_METRES']*0.25, bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
    im = ax[0][col].hist2d(
        10**y_test0,
        10 ** (X_test0["JSTART_LENGTH_METRES"]) * 0.25,
        bins=[x_bins, y_bins],
        range=[xlim, ylim],
        weights=X_test0["weights"],
        cmap=mycmap,
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2])
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][col].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
    ax[0][col].set_xlabel(r"$E_{True}$ [GeV]")
    ax[0][col].set_ylabel(r"$0.25*E_{JSTART-LENGTH}$ [GeV]")
    with open(
        "plots/" + name_suffix + "/" + wstr +
            "/" + str(LR) + "/JSTART.pkl", "wb"
    ) as f:
        pkl.dump(10 ** (X_test0["JSTART_LENGTH_METRES"]) * 0.25, f)
        pkl.dump(X_test0["weights"], f)
    ax[0][col].set_xscale("log")
    ax[0][col].set_yscale("log")
    ax[0][col].set_xticks([1, 10, 100])
    ax[0][col].set_yticks([1, 10, 100])
    ax[0][col].set_xlim(xlim)
    ax[0][col].set_ylim(ylim)
    fig.colorbar(im[3], ax=ax[0][col])
    # plt.colorbar(im, ax=ax[1])

    ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
    ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].set_ylabel(r"$\sigma_{0.25*E_{JSTART-lENGTH}}/E_{True}$")
    ax[1][col].set_xscale("log")
    ax[1][col].set_xlabel(r"$E_{True}$ [GeV]")
    # ax[1][col].set_yscale("log")
    ax[1][col].set_xticks([1, 10, 100])
    # ax[1][col].set_yticks([1, 10, 100])
    ax[1][col].set_xlim(xlim)
    ax[1][col].set_ylim(0, 4)
    # ax[1][1].set_ylim(1,100)

    ###### ENERGY ESTIMATED BY JSHOWERFIT ######

    if Jshf != 0:
        col += 1

        x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
        y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
        # im = ax[0][col].hist2d(10**y_test0,X_test0['JENERGY_ENERGY'], bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
        im = ax[0][col].hist2d(
            10**y_test0,
            10 ** X_test0["JSHOWERFIT_ENERGY"],
            bins=[x_bins, y_bins],
            range=[xlim, ylim],
            weights=X_test0["weights"],
            cmap=mycmap,
            cmin=cmin,
        )
        avg = avg_and_std(im[0], im[1], im[2])
        avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
        ax[0][col].errorbar(
            avg_x,
            avg_y,
            yerr=std,
            fmt=".",
            markersize=6,
            linewidth=1,
            c="black",
            label="weighted mean and std",
        )
        ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
        ax[0][col].set_xlabel(r"$E_{True}$ [GeV]")
        ax[0][col].set_ylabel(r"$E_{JSHOWERFIT}$ [GeV]")
        with open(
            "plots/" + name_suffix + "/" + wstr +
                "/" + str(LR) + "/JSHOWERFIT.pkl",
            "wb",
        ) as f:
            pkl.dump(10 ** X_test0["JSHOWERFIT_ENERGY"], f)
            pkl.dump(X_test0["weights"], f)
        ax[0][col].set_xscale("log")
        ax[0][col].set_yscale("log")
        ax[0][col].set_xticks([1, 10, 100])
        ax[0][col].set_yticks([1, 10, 100])
        ax[0][col].set_xlim(xlim)
        ax[0][col].set_ylim(ylim)
        fig.colorbar(im[3], ax=ax[0][col])
        # plt.colorbar(im, ax=ax[col])

        ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
        ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
        ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
        ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
        ax[1][col].set_ylabel(r"$\sigma_{JSHOWERFIT}/E_{True}$")
        ax[1][col].set_xscale("log")
        ax[1][col].set_xlabel(r"$E_{True}$ [GeV]")
        # ax[1][1].set_yscale("log")
        ax[1][col].set_xticks([1, 10, 100])
        # ax[1][col].set_yticks([1, 10, 100])
        ax[1][col].set_xlim(xlim)
        ax[1][col].set_ylim(0, 4)
        # ax[1][1].set_ylim(1,100)

    ###### ENERGY ESTIMATED BY GNN ######

    if len(Gcols) > 0:
        col += 1
        x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
        y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
        # im = ax[0][1].hist2d(10**y_test0,X_test0['pred_energy'], bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
        im = ax[0][col].hist2d(
            10**y_test0,
            10 ** X_test0["pred_energy"],
            bins=[x_bins, y_bins],
            range=[xlim, ylim],
            weights=X_test0["weights"],
            cmap=mycmap,
            cmin=cmin,
        )
        avg = avg_and_std(im[0], im[1], im[2])
        avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
        ax[0][col].errorbar(
            avg_x,
            avg_y,
            yerr=std,
            fmt=".",
            markersize=6,
            linewidth=1,
            c="black",
            label="weighted mean and std",
        )
        ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
        ax[0][col].set_xlabel(r"$E_{True}$ [GeV]")
        ax[0][col].set_ylabel(r"$E_{GNN}$ [GeV]")
        with open(
            "plots/" + name_suffix + "/" + wstr +
                "/" + str(LR) + "/GNN.pkl", "wb"
        ) as f:
            pkl.dump(10 ** X_test0["pred_energy"], f)
            pkl.dump(X_test0["weights"], f)
        ax[0][col].set_xscale("log")
        ax[0][col].set_yscale("log")
        ax[0][col].set_xticks([1, 10, 100])
        ax[0][col].set_yticks([1, 10, 100])
        ax[0][col].set_xlim(xlim)
        ax[0][col].set_ylim(ylim)
        fig.colorbar(im[3], ax=ax[0][col])
        # plt.colorbar(im, ax=ax[1])

        ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
        ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
        ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
        ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
        ax[1][col].set_ylabel(r"$\sigma_{GNN}/E_{True}$")
        ax[1][col].set_xscale("log")
        ax[1][col].set_xlabel(r"$E_{True}$ [GeV]")
        # ax[1][col].set_yscale("log")
        ax[1][col].set_xticks([1, 10, 100])
        # ax[1][col].set_yticks([1, 10, 100])
        ax[1][col].set_xlim(xlim)
        ax[1][col].set_ylim(0, 4)
        # ax[1][col].set_ylim(1,100)

        fig.suptitle(
            "Normal plots with GNN for NN "
            + name_suffix
            + "\n and "
            + n_features
            + " features with "
            + str(LR)
            + " as LR"
        )
        # fig.tight_layout()
        if save is True:
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/normal_2d_wGNN_"
                + name_suffix
                + "_f"
                + n_features
                + ".pdf"
            )
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/normal_2d_wGNN_"
                + name_suffix
                + "_f"
                + n_features
                + ".png"
            )
    else:
        fig.suptitle(
            "Normal plots for NN "
            + name_suffix
            + "\n and "
            + n_features
            + " features with "
            + str(LR)
            + " as LR"
        )
        # fig.tight_layout()
        if save is True:
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/normal_2d_"
                + name_suffix
                + "_f"
                + n_features
                + ".pdf"
            )
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/normal_2d_"
                + name_suffix
                + "_f"
                + n_features
                + ".png"
            )
    plt.clf()


def inv_2d_compare(
    X_test0,
    y_test0,
    y_pred0,
    mycmap="BuPu",
    name_suffix="draft",
    Gcols=0,
    Jshf=0,
    n_features=0,
    LR=0.1,
    wstr="w",
    save=True,
):
    cmin = 1e-10
    n_features = str(n_features)
    if len(Gcols) > 0:
        if Jshf != 0:
            dims = [2, 5]
        else:
            dims = [2, 4]
    else:
        if Jshf != 0:
            dims = [2, 4]
        else:
            dims = [2, 3]

    xlim = [1, 100]
    ylim = [1, 100]
    xlim_log = np.log10(xlim)
    ylim_log = np.log10(ylim)

    fig, ax = plt.subplots(
        dims[0], dims[1], figsize=(6 * dims[1], 4 * dims[0]), constrained_layout=True
    )
    fig.set_facecolor("white")
    nbines = 50
    # fig.tight_layout(pad =3.0)

    ###### ENERGY ESTIMATED BY the NN ######

    col = 0

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
    # im = ax[0][col].hist2d(10**y_pred0, 10**y_test0, bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
    im = ax[0][col].hist2d(
        10**y_pred0,
        10**y_test0,
        bins=[x_bins, y_bins],
        range=[xlim, ylim],
        weights=X_test0["weights"],
        cmap=mycmap,
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2])
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][col].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
    ax[0][col].set_ylabel(r"$E_{True}$ [GeV]")
    ax[0][col].set_xlabel(r"$E_{Est}$ [GeV]")
    ax[0][col].set_xscale("log")
    ax[0][col].set_yscale("log")
    ax[0][col].set_xticks([1, 10, 100])
    ax[0][col].set_yticks([1, 10, 100])
    ax[0][col].set_xlim(xlim)
    ax[0][col].set_ylim(ylim)
    fig.colorbar(im[3], ax=ax[0][col])
    # plt.show()

    ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
    ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].set_xlabel(r"$E_{Est}$ [GeV]")
    ax[1][col].set_ylabel(r"$\sigma_{True}/E_{Est}$")
    ax[1][col].set_xscale("log")
    # ax[1][col].set_yscale("log")
    ax[1][col].set_xticks([1, 10, 100])
    # ax[1][col].set_yticks([1, 10, 100])
    ax[1][col].set_xlim(xlim)
    ax[1][col].set_ylim(0, 4)

    ###### ENERGY ESTIMATED BY JENERGY  ######

    col += 1

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
    # im = ax[0][col].hist2d( X_test0['JENERGY_ENERGY'], 10**y_test0, bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
    im = ax[0][col].hist2d(
        10 ** X_test0["JENERGY_ENERGY"],
        10**y_test0,
        bins=[x_bins, y_bins],
        range=[xlim, ylim],
        weights=X_test0["weights"],
        cmap=mycmap,
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2])
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][col].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
    ax[0][col].set_ylabel(r"$E_{True}$ [GeV]")
    ax[0][col].set_xlabel(r"$E_{JENERGY}$ [GeV]")
    ax[0][col].set_xscale("log")
    ax[0][col].set_yscale("log")
    ax[0][col].set_xticks([1, 10, 100])
    ax[0][col].set_yticks([1, 10, 100])
    ax[0][col].set_xlim(xlim)
    ax[0][col].set_ylim(ylim)
    fig.colorbar(im[3], ax=ax[0][col])
    # plt.colorbar(im, ax=ax[1])

    ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
    ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].set_xlabel("True E")
    ax[1][col].set_ylabel(r"$\sigma_{True}/E_{JENERGY}$")
    ax[1][col].set_xscale("log")
    ax[1][col].set_xlabel(r"$E_{JENERGY}$ [GeV]")
    # ax[1][col].set_yscale("log")
    ax[1][col].set_xticks([1, 10, 100])
    # ax[1][col].set_yticks([1, 10, 100])
    ax[1][col].set_xlim(xlim)
    ax[1][col].set_ylim(0, 4)
    # ax[1][col].set_ylim(1,100)

    ###### ENERGY ESTIMATED BY JLENGTH ######

    col += 1

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
    # im = ax[0][col].hist2d(X_test0['JSTART_LENGTH_METRES']*0.25, 10**y_test0, bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
    im = ax[0][col].hist2d(
        10 ** (X_test0["JSTART_LENGTH_METRES"]) * 0.25,
        10**y_test0,
        bins=[x_bins, y_bins],
        range=[xlim, ylim],
        weights=X_test0["weights"],
        cmap=mycmap,
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2])
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][col].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
    ax[0][col].set_ylabel(r"$E_{True}$ [GeV]")
    ax[0][col].set_xlabel(r"$0.25*E_{JSTART-LENGTH}$ [GeV]")
    ax[0][col].set_xscale("log")
    ax[0][col].set_yscale("log")
    ax[0][col].set_xticks([1, 10, 100])
    ax[0][col].set_yticks([1, 10, 100])
    ax[0][col].set_xlim(xlim)
    ax[0][col].set_ylim(ylim)
    fig.colorbar(im[3], ax=ax[0][col])
    # plt.colorbar(im, ax=ax[1])

    ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
    ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].set_ylabel(r"$\sigma_{True}/0.25*E_{JSTART-LENGTH}$")
    ax[1][col].set_xscale("log")
    ax[1][col].set_xlabel(r"$0.25*E_{JSTART-lENGTH}$ [GeV]")
    # ax[1][col].set_yscale("log")
    ax[1][col].set_xticks([1, 10, 100])
    # ax[1][col].set_yticks([1, 10, 100])
    ax[1][col].set_xlim(xlim)
    ax[1][col].set_ylim(0, 4)
    # ax[1][col].set_ylim(1,100)

    ###### ENERGY ESTIMATED BY JSHOWERFIT  ######

    col += 1

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
    # im = ax[0][col].hist2d( X_test0['JENERGY_ENERGY'], 10**y_test0, bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
    im = ax[0][col].hist2d(
        10 ** X_test0["JSHOWERFIT_ENERGY"],
        10**y_test0,
        bins=[x_bins, y_bins],
        range=[xlim, ylim],
        weights=X_test0["weights"],
        cmap=mycmap,
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2])
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][col].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
    ax[0][col].set_ylabel(r"$E_{True}$ [GeV]")
    ax[0][col].set_xlabel(r"$E_{JSHOWERFIT}$ [GeV]")
    ax[0][col].set_xscale("log")
    ax[0][col].set_yscale("log")
    ax[0][col].set_xticks([1, 10, 100])
    ax[0][col].set_yticks([1, 10, 100])
    ax[0][col].set_xlim(xlim)
    ax[0][col].set_ylim(ylim)
    fig.colorbar(im[3], ax=ax[0][col])
    # plt.colorbar(im, ax=ax[1])

    ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
    ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
    ax[1][col].set_xlabel("True E")
    ax[1][col].set_ylabel(r"$\sigma_{True}/E_{JSHOWERFIT}$")
    ax[1][col].set_xscale("log")
    ax[1][col].set_xlabel(r"$E_{JSHOWERFIT}$ [GeV]")
    # ax[1][col].set_yscale("log")
    ax[1][col].set_xticks([1, 10, 100])
    # ax[1][col].set_yticks([1, 10, 100])
    ax[1][col].set_xlim(xlim)
    ax[1][col].set_ylim(0, 4)
    # ax[1][col].set_ylim(1,100)

    ###### ENERGY ESTIMATED BY GNN  ######

    if len(Gcols) > 0:
        col += 1

        x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
        y_bins = np.logspace(ylim_log[0], ylim_log[1], nbines)
        # im = ax[0][col].hist2d( X_test0['pred_energy'], 10**y_test0, bins=[x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,ylim])
        im = ax[0][col].hist2d(
            10 ** X_test0["pred_energy"],
            10**y_test0,
            bins=[x_bins, y_bins],
            range=[xlim, ylim],
            weights=X_test0["weights"],
            cmap=mycmap,
            cmin=cmin,
        )
        avg = avg_and_std(im[0], im[1], im[2])
        avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
        ax[0][col].errorbar(
            avg_x,
            avg_y,
            yerr=std,
            fmt=".",
            markersize=6,
            linewidth=1,
            c="black",
            label="weighted mean and std",
        )
        ax[0][col].plot(10**y_test0, 10**y_test0, color="black")
        ax[0][col].set_ylabel(r"$E_{True}$ [GeV]")
        ax[0][col].set_xlabel(r"$E_{GNN}$ [GeV]")
        ax[0][col].set_xscale("log")
        ax[0][col].set_yscale("log")
        ax[0][col].set_xticks([1, 10, 100])
        ax[0][col].set_yticks([1, 10, 100])
        ax[0][col].set_xlim(xlim)
        ax[0][col].set_ylim(ylim)
        fig.colorbar(im[3], ax=ax[0][col])
        # plt.colorbar(im, ax=ax[1])

        ax[1][col].scatter(avg_x, std / avg_x, s=10, c="black")
        ax[1][col].plot(avg_x, np.ones(len(avg_x)), "--", c="red")
        ax[1][col].plot(avg_x, 0.5 * np.ones(len(avg_x)), "--", c="red")
        ax[1][col].plot(avg_x, 0.25 * np.ones(len(avg_x)), "--", c="red")
        ax[1][col].set_xlabel("True E")
        ax[1][col].set_ylabel(r"$\sigma_{True}/E_{GNN}$")
        ax[1][col].set_xscale("log")
        ax[1][col].set_xlabel(r"$E_{GNN}$ [GeV]")
        # ax[1][col].set_yscale("log")
        ax[1][col].set_xticks([1, 10, 100])
        # ax[1][col].set_yticks([1, 10, 100])
        ax[1][col].set_xlim(xlim)
        ax[1][col].set_ylim(0, 4)
        # ax[1][col].set_ylim(1,100)

        fig.suptitle(
            "Inverted plots with GNN for NN "
            + name_suffix
            + "\n and "
            + n_features
            + " features with "
            + str(LR)
            + " as LR"
        )
        # fig.tight_layout()
        if save is True:
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/inv_2d_wGNN_"
                + name_suffix
                + "_f"
                + n_features
                + ".pdf"
            )
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/inv_2d_wGNN_"
                + name_suffix
                + "_f"
                + n_features
                + ".png"
            )
    else:
        fig.suptitle(
            "Inverted plots for NN "
            + name_suffix
            + "\n and "
            + n_features
            + " features with "
            + str(LR)
            + " as LR"
        )
        # fig.tight_layout()
        if save is True:
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/inv_2d_"
                + name_suffix
                + "_f"
                + n_features
                + ".pdf"
            )
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/inv_2d_"
                + name_suffix
                + "_f"
                + n_features
                + ".png"
            )
    plt.clf()


def error_plots(
    X_test0,
    y_test0,
    y_pred0,
    mycmap="viridis",
    name_suffix="draft",
    Gcols=0,
    n_features=0,
    LR=0.1,
    wstr="w",
    save=True,
):
    cmin = 1e-10
    n_features = str(n_features)

    if len(Gcols) > 0:
        dims = [4, 4]
    else:
        dims = [4, 3]

    xlim = [1, 100]
    ylim = [1, 100]
    error_ylim = [-1.5, 5]
    logerror_ylim = [-2.5, 2.5]
    xlim_log = np.log10(xlim)
    ylim_log = np.log10(ylim)

    fig, ax = plt.subplots(
        dims[0], dims[1], figsize=(6 * dims[1], 4 * dims[0]), constrained_layout=True
    )
    fig.set_facecolor("white")
    nbines = 50
    # fig.tight_layout(pad =3.0)

    ###### ENERGY ESTIMATED BY the NN ######

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(error_ylim[0], error_ylim[1], nbines)
    # im = ax[0][0].hist2d(10**y_test0,((10**y_pred0-10**y_test0)/10**y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,error_ylim])
    im = ax[0][0].hist2d(
        10**y_test0,
        ((10**y_pred0 - 10**y_test0) / 10**y_test0),
        bins=[x_bins, y_bins],
        range=[xlim, error_ylim],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][0].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][0].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[0][0].plot(10**y_test0, np.zeros(len(y_test0)), color="red")
    ax[0][0].plot(10**y_test0, -np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[0][0].set_ylabel(r"$\frac{E_{Est}-E_{True}}{E_{True}}$")
    ax[0][0].set_xlabel(r"$E_{True}$ [GeV]")
    ax[0][0].set_xscale("log")
    ax[0][0].set_xticks([1, 10, 100])
    ax[0][0].set_xlim(xlim)
    ax[0][0].set_ylim(error_ylim)
    fig.colorbar(im[3], ax=ax[0][0])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(-2.5, 2.5, nbines)
    # im = ax[1][0].hist2d(10**y_test0,((y_pred0-y_test0)/y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[-2.5,2.5]], weights = X_test0['weights'])
    im = ax[1][0].hist2d(
        10**y_test0,
        ((y_pred0 - y_test0) / y_test0),
        bins=[x_bins, y_bins],
        range=[xlim, [-2.5, 2.5]],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[1][0].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[1][0].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[1][0].plot(10**y_test0, np.zeros(len(y_test0)), color="red")
    ax[1][0].plot(10**y_test0, -np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[1][0].set_ylabel(
        r"$\frac{L_{10}E_{Est}-L_{10}E_{True}}{L_{10}E_{True}}$")
    ax[1][0].set_xlabel(r"$E_{True}$ [GeV]")
    ax[1][0].set_xscale("log")
    ax[1][0].set_xticks([1, 10, 100])
    ax[1][0].set_xlim(xlim)
    ax[1][0].set_ylim(-2.5, 2.5)
    fig.colorbar(im[3], ax=ax[1][0])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(0, error_ylim[1], nbines)
    # im = ax[2][0].hist2d(10**y_test0,np.abs((10**y_pred0-10**y_test0)/10**y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[0, error_ylim[1]]])
    im = ax[2][0].hist2d(
        10**y_test0,
        np.abs((10**y_pred0 - 10**y_test0) / 10**y_test0),
        bins=[x_bins, y_bins],
        range=[xlim, [0, error_ylim[1]]],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[2][0].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[2][0].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[2][0].plot(
        10**y_test0, 0.5 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[2][0].plot(
        10**y_test0, 0.25 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[2][0].set_ylabel(r"$\frac{|E_{Est}-E_{True}|}{E_{True}}$")
    ax[2][0].set_xlabel(r"$E_{True}$ [GeV]")
    ax[2][0].set_xscale("log")
    ax[2][0].set_xticks([1, 10, 100])
    ax[2][0].set_xlim(xlim)
    ax[2][0].set_ylim(0, error_ylim[1])
    fig.colorbar(im[3], ax=ax[2][0])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(0, 2.5, nbines)
    # im = ax[3][0].hist2d(10**y_test0,np.abs((y_pred0-y_test0)/y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[0,2.5]])
    im = ax[3][0].hist2d(
        10**y_test0,
        np.abs((y_pred0 - y_test0) / y_test0),
        bins=[x_bins, y_bins],
        range=[xlim, [0, 2.5]],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[3][0].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[3][0].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[3][0].plot(
        10**y_test0, 0.5 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[3][0].plot(
        10**y_test0, 0.25 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[3][0].set_ylabel(
        r"$\frac{|L_{10}E_{Est}-L_{10}E_{True}|}{L_{10}E_{True}}$")
    ax[3][0].set_xlabel(r"$E_{True}$ [GeV]")
    ax[3][0].set_xscale("log")
    ax[3][0].set_xticks([1, 10, 100])
    ax[3][0].set_xlim(xlim)
    ax[3][0].set_ylim(0, 2.5)
    fig.colorbar(im[3], ax=ax[3][0])

    ###### ENERGY ESTIMATED BY the JENERGY ######

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(error_ylim[0], error_ylim[1], nbines)
    # im = ax[0][1].hist2d(10**y_test0,((X_test0['JENERGY_ENERGY']-10**y_test0)/10**y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,error_ylim])
    im = ax[0][1].hist2d(
        10**y_test0,
        ((10 ** X_test0["JENERGY_ENERGY"] - 10**y_test0) / 10**y_test0),
        bins=[x_bins, y_bins],
        range=[xlim, error_ylim],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][1].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][1].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[0][1].plot(10**y_test0, np.zeros(len(y_test0)), color="red")
    ax[0][1].plot(10**y_test0, -np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[0][1].set_ylabel(r"$\frac{E_{JENERGY}-E_{True}}{E_{True}}$")
    ax[0][1].set_xlabel(r"$E_{True}$ [GeV]")
    ax[0][1].set_xscale("log")
    ax[0][1].set_xticks([1, 10, 100])
    ax[0][1].set_xlim(xlim)
    ax[0][1].set_ylim(error_ylim)
    fig.colorbar(im[3], ax=ax[0][1])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(-2.5, 2.5, nbines)
    # im = ax[1][1].hist2d(10**y_test0,((np.log10(X_test0['JENERGY_ENERGY'])-y_test0)/y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[-2.5,2.5]])
    im = ax[1][1].hist2d(
        10**y_test0,
        ((X_test0["JENERGY_ENERGY"] - y_test0) / y_test0),
        bins=[x_bins, y_bins],
        range=[xlim, [-2.5, 2.5]],
        weights=X_test0["weights"],
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[1][1].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[1][1].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[1][1].plot(10**y_test0, np.zeros(len(y_test0)), color="red")
    ax[1][1].plot(10**y_test0, -np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[1][1].set_ylabel(
        r"$\frac{L_{10}E_{JENERGY}-L_{10}E_{True}}{L_{10}E_{True}}$")
    ax[1][1].set_xlabel(r"$E_{True}$ [GeV]")
    ax[1][1].set_xscale("log")
    ax[1][1].set_xticks([1, 10, 100])
    ax[1][1].set_xlim(xlim)
    ax[1][1].set_ylim(-2.5, 2.5)
    fig.colorbar(im[3], ax=ax[1][1])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(0, error_ylim[1], nbines)
    # im = ax[2][1].hist2d(10**y_test0,np.abs((X_test0['JENERGY_ENERGY']-10**y_test0)/10**y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[0,error_ylim[1]]])
    im = ax[2][1].hist2d(
        10**y_test0,
        np.abs((10 ** X_test0["JENERGY_ENERGY"] - 10**y_test0) / 10**y_test0),
        bins=[x_bins, y_bins],
        range=[xlim, [0, error_ylim[1]]],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[2][1].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[2][1].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[2][1].plot(
        10**y_test0, 0.5 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[2][1].plot(
        10**y_test0, 0.25 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[2][1].set_ylabel(r"$\frac{|E_{JENERGY}-E_{True}|}{E_{True}}$")
    ax[2][1].set_xlabel(r"$E_{True}$ [GeV]")
    ax[2][1].set_xscale("log")
    ax[2][1].set_xticks([1, 10, 100])
    ax[2][1].set_xlim(xlim)
    ax[2][1].set_ylim(0, error_ylim[1])
    fig.colorbar(im[3], ax=ax[2][1])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(0, 2.5, nbines)
    # im = ax[3][1].hist2d(10**y_test0,np.abs((np.log10(X_test0['JENERGY_ENERGY'])-y_test0)/y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[0,2.5]])
    im = ax[3][1].hist2d(
        10**y_test0,
        np.abs((X_test0["JENERGY_ENERGY"] - y_test0) / y_test0),
        bins=[x_bins, y_bins],
        range=[xlim, [0, 2.5]],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[3][1].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[3][1].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[3][1].plot(
        10**y_test0, 0.5 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[3][1].plot(
        10**y_test0, 0.25 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[3][1].set_ylabel(
        r"$\frac{|L_{10}E_{JENERGY}-L_{10}E_{True}|}{L_{10}E_{True}}$")
    ax[3][1].set_xlabel(r"$E_{True}$ [GeV]")
    ax[3][1].set_xscale("log")
    ax[3][1].set_xticks([1, 10, 100])
    ax[3][1].set_xlim(xlim)
    ax[3][1].set_ylim(0, 2.5)
    fig.colorbar(im[3], ax=ax[3][1])

    ###### ENERGY ESTIMATED BY the JLENGTH ######

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(error_ylim[0], error_ylim[1], nbines)
    # im = ax[0][2].hist2d(10**y_test0,((0.25*X_test0['JSTART_LENGTH_METRES']-10**y_test0)/10**y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,error_ylim])
    im = ax[0][2].hist2d(
        10**y_test0,
        (
            (10 ** (X_test0["JSTART_LENGTH_METRES"]) * 0.25 - 10**y_test0)
            / 10**y_test0
        ),
        bins=[x_bins, y_bins],
        range=[xlim, error_ylim],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[0][2].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[0][2].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[0][2].plot(10**y_test0, np.zeros(len(y_test0)), color="red")
    ax[0][2].plot(10**y_test0, -np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[0][2].set_ylabel(r"$\frac{0.25*E_{JSTART-LENGTH}-E_{True}}{E_{True}}$")
    ax[0][2].set_xlabel(r"$E_{True}$ [GeV]")
    ax[0][2].set_xscale("log")
    ax[0][2].set_xticks([1, 10, 100])
    ax[0][2].set_xlim(xlim)
    ax[0][2].set_ylim(error_ylim)
    fig.colorbar(im[3], ax=ax[0][2])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(-2.5, 2.5, nbines)
    # im = ax[1][2].hist2d(10**y_test0,((np.log10(0.25*X_test0['JSTART_LENGTH_METRES'])-y_test0)/y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[-2.5,2.5]])
    im = ax[1][2].hist2d(
        10**y_test0,
        (
            (np.log10(
                10 ** (X_test0["JSTART_LENGTH_METRES"]) * 0.25) - y_test0)
            / y_test0
        ),
        bins=[x_bins, y_bins],
        range=[xlim, [-2.5, 2.5]],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[1][2].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[1][2].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[1][2].plot(10**y_test0, np.zeros(len(y_test0)), color="red")
    ax[1][2].plot(10**y_test0, -np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[1][2].set_ylabel(
        r"$\frac{L_{10}0.25*E_{JSTART-LENGTH}-L_{10}E_{True}}{L_{10}E_{True}}$"
    )
    ax[1][2].set_xlabel(r"$E_{True}$ [GeV]")
    ax[1][2].set_xscale("log")
    ax[1][2].set_xticks([1, 10, 100])
    ax[1][2].set_xlim(xlim)
    ax[1][2].set_ylim(-2.5, 2.5)
    fig.colorbar(im[3], ax=ax[1][2])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(0, error_ylim[1], nbines)
    # im = ax[2][2].hist2d(10**y_test0,np.abs((0.25*X_test0['JSTART_LENGTH_METRES']-10**y_test0)/10**y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[0,error_ylim[1]]])
    im = ax[2][2].hist2d(
        10**y_test0,
        np.abs(
            (10 ** (X_test0["JSTART_LENGTH_METRES"]) * 0.25 - 10**y_test0)
            / 10**y_test0
        ),
        bins=[x_bins, y_bins],
        range=[xlim, [0, error_ylim[1]]],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[2][2].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[2][2].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[2][2].plot(
        10**y_test0, 0.5 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[2][2].plot(
        10**y_test0, 0.25 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[2][2].set_ylabel(
        r"$\frac{|0.25*E_{JSTART-LENGTH}-E_{True}|}{E_{True}}$")
    ax[2][2].set_xlabel(r"$E_{True}$ [GeV]")
    ax[2][2].set_xscale("log")
    ax[2][2].set_xticks([1, 10, 100])
    ax[2][2].set_xlim(xlim)
    ax[2][2].set_ylim(0, error_ylim[1])
    fig.colorbar(im[3], ax=ax[2][2])

    x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
    y_bins = np.linspace(0, 2.5, nbines)
    # im = ax[3][2].hist2d(10**y_test0,np.abs((0.25*np.log10(X_test0['JSTART_LENGTH_METRES'])-y_test0)/y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[0,2.5]])
    im = ax[3][2].hist2d(
        10**y_test0,
        (
            np.abs(
                np.log10(10 ** (X_test0["JSTART_LENGTH_METRES"]) * 0.25) - y_test0)
            / y_test0
        ),
        bins=[x_bins, y_bins],
        range=[xlim, [0, 2.5]],
        weights=X_test0["weights"],
        cmin=cmin,
    )
    avg = avg_and_std(im[0], im[1], im[2], err=1)
    avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
    ax[3][2].errorbar(
        avg_x,
        avg_y,
        yerr=std,
        fmt=".",
        markersize=6,
        linewidth=1,
        c="black",
        label="weighted mean and std",
    )
    ax[3][2].plot(10**y_test0, np.ones(len(y_test0)),
                  color="red", linestyle="--")
    ax[3][2].plot(
        10**y_test0, 0.5 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[3][2].plot(
        10**y_test0, 0.25 * np.ones(len(y_test0)), color="red", linestyle="--"
    )
    ax[3][2].set_ylabel(
        r"$\frac{|L_{10}E_{JSTART-LENGTH}-L_{10}E_{True}|}{L_{10}E_{True}}$"
    )
    ax[3][2].set_xlabel(r"$E_{True}$ [GeV]")
    ax[3][2].set_xscale("log")
    ax[3][2].set_xticks([1, 10, 100])
    ax[3][2].set_xlim(xlim)
    ax[3][2].set_ylim(0, 2.5)
    fig.colorbar(im[3], ax=ax[3][2])

    if len(Gcols) > 0:
        ###### ENERGY ESTIMATED BY the GNN ######

        x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
        y_bins = np.linspace(error_ylim[0], error_ylim[1], nbines)
        # im = ax[0][3].hist2d(10**y_test0,((X_test0['pred_energy']-10**y_test0)/10**y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,error_ylim])
        im = ax[0][3].hist2d(
            10**y_test0,
            ((10 ** X_test0["pred_energy"] - 10**y_test0) / 10**y_test0),
            bins=[x_bins, y_bins],
            range=[xlim, error_ylim],
            weights=X_test0["weights"],
            cmin=cmin,
        )
        avg = avg_and_std(im[0], im[1], im[2], err=1)
        avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
        ax[0][3].errorbar(
            avg_x,
            avg_y,
            yerr=std,
            fmt=".",
            markersize=6,
            linewidth=1,
            c="black",
            label="weighted mean and std",
        )
        ax[0][3].plot(10**y_test0, np.ones(len(y_test0)),
                      color="red", linestyle="--")
        ax[0][3].plot(10**y_test0, np.zeros(len(y_test0)), color="red")
        ax[0][3].plot(
            10**y_test0, -np.ones(len(y_test0)), color="red", linestyle="--"
        )
        ax[0][3].set_ylabel(r"$\frac{E_{GNN}-E_{True}}{E_{True}}$")
        ax[0][3].set_xlabel(r"$E_{True}$ [GeV]")
        ax[0][3].set_xscale("log")
        ax[0][3].set_xticks([1, 10, 100])
        ax[0][3].set_xlim(xlim)
        ax[0][3].set_ylim(error_ylim)
        fig.colorbar(im[3], ax=ax[0][3])

        x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
        y_bins = np.linspace(-2.5, 2.5, nbines)
        # im = ax[1][3].hist2d(10**y_test0,((np.log10(X_test0['pred_energy'])-y_test0)/y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[-2.5,2.5]])
        im = ax[1][3].hist2d(
            10**y_test0,
            ((np.log10(10 ** X_test0["pred_energy"]) - y_test0) / y_test0),
            bins=[x_bins, y_bins],
            range=[xlim, [-2.5, 2.5]],
            weights=X_test0["weights"],
            cmin=cmin,
        )
        avg = avg_and_std(im[0], im[1], im[2], err=1)
        avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
        ax[1][3].errorbar(
            avg_x,
            avg_y,
            yerr=std,
            fmt=".",
            markersize=6,
            linewidth=1,
            c="black",
            label="weighted mean and std",
        )
        ax[1][3].plot(10**y_test0, np.ones(len(y_test0)),
                      color="red", linestyle="--")
        ax[1][3].plot(10**y_test0, np.zeros(len(y_test0)), color="red")
        ax[1][3].plot(
            10**y_test0, -np.ones(len(y_test0)), color="red", linestyle="--"
        )
        ax[1][3].set_ylabel(
            r"$\frac{L_{10}E_{GNN}-L_{10}E_{True}}{L_{10}E_{True}}$")
        ax[1][3].set_xlabel(r"$E_{True}$ [GeV]")
        ax[1][3].set_xscale("log")
        ax[1][3].set_xticks([1, 10, 100])
        ax[1][3].set_xlim(xlim)
        ax[1][3].set_ylim(-2.5, 2.5)
        fig.colorbar(im[3], ax=ax[1][3])

        x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
        y_bins = np.linspace(0, error_ylim[1], nbines)
        # im = ax[2][3].hist2d(10**y_test0,np.abs((X_test0['pred_energy']-10**y_test0)/10**y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[0,error_ylim[1]]])
        im = ax[2][3].hist2d(
            10**y_test0,
            np.abs((10 ** X_test0["pred_energy"] - 10**y_test0) / 10**y_test0),
            bins=[x_bins, y_bins],
            range=[xlim, [0, error_ylim[1]]],
            weights=X_test0["weights"],
            cmin=cmin,
        )
        avg = avg_and_std(im[0], im[1], im[2], err=1)
        avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
        ax[2][3].errorbar(
            avg_x,
            avg_y,
            yerr=std,
            fmt=".",
            markersize=6,
            linewidth=1,
            c="black",
            label="weighted mean and std",
        )
        ax[2][3].plot(10**y_test0, np.ones(len(y_test0)),
                      color="red", linestyle="--")
        ax[2][3].plot(
            10**y_test0, 0.5 * np.ones(len(y_test0)), color="red", linestyle="--"
        )
        ax[2][3].plot(
            10**y_test0, 0.25 * np.ones(len(y_test0)), color="red", linestyle="--"
        )
        ax[2][3].set_ylabel(r"$\frac{|E_{GNN}-E_{True}|}{E_{True}}$")
        ax[2][3].set_xlabel(r"$E_{True}$ [GeV]")
        ax[2][3].set_xscale("log")
        ax[2][3].set_xticks([1, 10, 100])
        ax[2][3].set_xlim(xlim)
        ax[2][3].set_ylim(0, error_ylim[1])
        fig.colorbar(im[3], ax=ax[2][3])

        x_bins = np.logspace(xlim_log[0], xlim_log[1], nbines)
        y_bins = np.linspace(0, 2.5, nbines)
        # im = ax[3][3].hist2d(10**y_test0,np.abs((np.log10(X_test0['pred_energy'])-y_test0)/y_test0), bins = [x_bins, y_bins], norm=mpl.colors.LogNorm(), range =[xlim,[0,2.5]])
        im = ax[3][3].hist2d(
            10**y_test0,
            np.abs(
                (np.log10(10 ** X_test0["pred_energy"]) - y_test0) / y_test0),
            bins=[x_bins, y_bins],
            range=[xlim, [0, 2.5]],
            weights=X_test0["weights"],
            cmin=cmin,
        )
        avg = avg_and_std(im[0], im[1], im[2], err=1)
        avg_x, avg_y, std = avg[:, 0], avg[:, 1], avg[:, 2]
        ax[3][3].errorbar(
            avg_x,
            avg_y,
            yerr=std,
            fmt=".",
            markersize=6,
            linewidth=1,
            c="black",
            label="weighted mean and std",
        )
        ax[3][3].plot(10**y_test0, np.ones(len(y_test0)),
                      color="red", linestyle="--")
        ax[3][3].plot(
            10**y_test0, 0.5 * np.ones(len(y_test0)), color="red", linestyle="--"
        )
        ax[3][3].plot(
            10**y_test0, 0.25 * np.ones(len(y_test0)), color="red", linestyle="--"
        )
        ax[3][3].set_ylabel(
            r"$\frac{|L_{10}E_{GNN}-L_{10}E_{True}|}{L_{10}E_{True}}$")
        ax[3][3].set_xlabel(r"$E_{True}$ [GeV]")
        ax[3][3].set_xscale("log")
        ax[3][3].set_xticks([1, 10, 100])
        ax[3][3].set_xlim(xlim)
        ax[3][3].set_ylim(0, 2.5)
        fig.colorbar(im[3], ax=ax[3][3])

        fig.suptitle(
            "Error plots with GNN for NN "
            + name_suffix
            + "\n and "
            + n_features
            + " features with "
            + str(LR)
            + " as LR"
        )
        # fig.tight_layout()
        if save is True:
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/error_plots_wGNN_"
                + name_suffix
                + "_f"
                + n_features
                + ".pdf"
            )
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/error_plots_wGNN_"
                + name_suffix
                + "_f"
                + n_features
                + ".png"
            )
    else:
        fig.suptitle(
            "Error plots for NN "
            + name_suffix
            + "\n and "
            + n_features
            + " features with "
            + str(LR)
            + " as LR"
        )
        # fig.tight_layout()
        if save is True:
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/error_plots_"
                + name_suffix
                + "_f"
                + n_features
                + ".pdf"
            )
            fig.savefig(
                "plots/"
                + name_suffix
                + "/"
                + wstr
                + "/"
                + str(LR)
                + "/error_plots_"
                + name_suffix
                + "_f"
                + n_features
                + ".png"
            )
    plt.clf()
