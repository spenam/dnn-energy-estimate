import numpy as np
import matplotlib.pyplot as plt
import uproot
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#hist_name = 'mc_data_hists'

hist_name = 'mc_hists_v7.2'








def make_datamc_hist(ax0, hists0, xlabel0, labels0, colors0):
    

    #muons0 = hists0[0]
    NC0 = hists0[0]
    e0 = hists0[1]
    mu0 = hists0[2]
    tau0 = hists0[3]
    #mctotal0 = hists0[4]
    #ax0.hist((muons0[1][:-1], NC0[1][:-1], e0[1][:-1], mu0[1][:-1], tau0[1][:-1], mctotal0[1][:-1]),
    ax0.hist((tau0[1][:-1], NC0[1][:-1], e0[1][:-1], mu0[1][:-1]),
             #bins = muons0[1],
             bins = NC0[1],
             #weights = (muons0[0], NC0[0], e0[0], mu0[0], tau0[0], mctotal0[0]),
             weights = (tau0[0], NC0[0], e0[0], mu0[0]),
             label = labels0,
             histtype = 'step',
             zorder=10,
             color = colors0)
    #ax0.legend(loc = 'upper left')
    ax0.legend(loc = 'best', framealpha=0.5)
    ax0.set_xlabel(xlabel0)
    ax0.set_ylabel("Rate [Hz]")
    ax0.grid(True)
    ax0.set_xscale("log")
    #ax0.set_yscale("log")

def make_datamc_hist_stacked(ax0, hists0, xlabel0, labels0, colors0):
    

    #muons0 = hists0[0]
    NC0 = hists0[0]
    e0 = hists0[1]
    mu0 = hists0[2]
    tau0 = hists0[3]
    #mctotal0 = hists0[4]
    #ax0.hist((muons0[1][:-1], NC0[1][:-1], e0[1][:-1], mu0[1][:-1], tau0[1][:-1], mctotal0[1][:-1]),
    ax0.hist(( tau0[1][:-1], NC0[1][:-1], e0[1][:-1], mu0[1][:-1]),
             #bins = muons0[1],
             bins = NC0[1],
             #weights = (muons0[0], NC0[0], e0[0], mu0[0], tau0[0], mctotal0[0]),
             weights = (tau0[0], NC0[0], e0[0], mu0[0]),
             label = labels0,
             stacked = True,
             zorder=10,
             color = colors0)
    #ax0.legend(loc = 'upper left')
    ax0.legend(loc = 'best', framealpha=0.5)
    ax0.set_xlabel(xlabel0)
    ax0.set_ylabel("Rate [Hz]")
    ax0.grid(True)
    ax0.set_xscale("log")
    #ax0.set_yscale("log")


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def defaultColorsMpl(alpha=1.):
    HexColors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    RGBColors = np.ones((len(HexColors),4))

    for i,hC in enumerate(HexColors) :
        RGBColors[i,:-1] = np.array(hex_to_rgb(hC))/255.
    RGBColors[:,3] = alpha
    return RGBColors


def niceCBTop(axe,fig,obj, loc = 'upper right'):
    
    axins = inset_axes(axe,
                       width="60%",  # width = 50% of parent_bbox width
                       height="5%",  # height : 5%
                       loc=loc)
    cbar = fig.colorbar(obj,cax=axins,orientation="horizontal")
    return cbar


def getSigmaRel(x,y,values):

    yC = (y[1:]+y[:-1])/2.
    mean   = np.zeros(np.shape(values)[0])
    sigRel = np.zeros(np.shape(values)[0])

    for i in range(len(sigRel)):
        mean[i] = np.sum(yC*values[i]) / np.sum(values[i])
        sigRel[i] = np.sqrt(np.sum(values[i]*(yC-mean[i])**2)/np.sum(values[i]))
            
    return sigRel,mean


def getProp(x,y,values,threshold=0.50):
    dist = np.cumsum(values.astype(float), axis=1)/np.sum(values.astype(float), axis=1)[:,np.newaxis]
    MRE = np.zeros(np.shape(values)[0])
    for i in range(len(MRE)):
        prev = 0
        for j, v in enumerate(dist[i]):
            if v >= threshold :
                bL = y[j+1] - y[j]
                MRE[i] = y[j] + bL*(threshold-prev)/(v-prev)
                break
            prev = v
    return MRE


def plotResolution(h,axe,drawLine=True):
    values = h[0]
    x = h[1]
    xC = (x[1:] + x[:-1])/2.
    y = h[2]    
    sigRel, mean = getSigmaRel(x,y,values)
    pcl = axe.pcolormesh(x,y,np.swapaxes(values,1,0),norm=mpl.colors.LogNorm(),rasterized=True)
    fig = plt.gcf()
    # fig.colorbar(pcl, ax=axe)
    cbar = niceCBTop(axe,fig,pcl)
    sigmaBelow = getProp(x,y,values,0.159)
    median     = getProp(x,y,values,0.50)
    sigmaAbove = getProp(x,y,values,0.841)
    sigmaKwargs = {'histtype':'step','color':'C1','linewidth':2}
    medianKwargs = {'histtype':'step','color':'C3','linewidth':2}

    if drawLine:
        axe.hist(x[:-1], bins = x, weights = sigmaBelow,**sigmaKwargs)
        axe.hist(x[:-1], bins = x, weights = sigmaAbove,linestyle=':',**sigmaKwargs)
        axe.hist(x[:-1], bins = x, weights = median,**medianKwargs)
    

    return median, [-sigmaBelow+median,sigmaAbove-median]


datasets = {
    "JG_E_vs_trueEnergy_simple": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_with_simple_cuts/trueEnergy_vs_JG_E/nu mu CC tot',
        "label": "JEnergy",
    },
    "JG_Len_vs_trueEnergy_simple": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_with_simple_cuts/trueEnergy_vs_JG_Len/nu mu CC tot',
        "label": "JStart",
    },
    "JS_E_vs_trueEnergy_simple": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_with_simple_cuts/trueEnergy_vs_JS_E/nu mu CC tot',
        "label": "JShF",
    },
    "DNN_Eest_vs_trueEnergy_simple": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_with_simple_cuts/trueEnergy_vs_DNN_Eest/nu mu CC tot',
        "label": "Eest",
    },
    "JG_E_vs_trueEnergy_pure_track": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_pure_track/trueEnergy_vs_JG_E/nu mu CC tot',
        "label": "JEnergy",
    },
    "JG_Len_vs_trueEnergy_pure_track": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_pure_track/trueEnergy_vs_JG_Len/nu mu CC tot',
        "label": "JStart",
    },
    "JS_E_vs_trueEnergy_pure_track": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_pure_track/trueEnergy_vs_JS_E/nu mu CC tot',
        "label": "JShF",
    },
    "DNN_Eest_vs_trueEnergy_pure_track": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_pure_track/trueEnergy_vs_DNN_Eest/nu mu CC tot',
        "label": "Eest",
    },
    "JG_E_vs_trueEnergy_pure_shower": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_pure_shower/trueEnergy_vs_JG_E/nu mu CC tot',
        "label": "JEnergy",
    },
    "JG_Len_vs_trueEnergy_pure_shower": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_pure_shower/trueEnergy_vs_JG_Len/nu mu CC tot',
        "label": "JStart",
    },
    "JS_E_vs_trueEnergy_pure_shower": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_pure_shower/trueEnergy_vs_JS_E/nu mu CC tot',
        "label": "JShF",
    },
    "DNN_Eest_vs_trueEnergy_pure_shower": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_pure_shower/trueEnergy_vs_DNN_Eest/nu mu CC tot',
        "label": "Eest",
    },
    "JG_E_vs_trueEnergy_contaminated_track": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_contaminated_track/trueEnergy_vs_JG_E/nu mu CC tot',
        "label": "JEnergy",
    },
    "JG_Len_vs_trueEnergy_contaminated_track": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_contaminated_track/trueEnergy_vs_JG_Len/nu mu CC tot',
        "label": "JStart",
    },
    "JS_E_vs_trueEnergy_contaminated_track": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_contaminated_track/trueEnergy_vs_JS_E/nu mu CC tot',
        "label": "JShF",
    },
    "DNN_Eest_vs_trueEnergy_contaminated_track": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_contaminated_track/trueEnergy_vs_DNN_Eest/nu mu CC tot',
        "label": "Eest",
    },
    "JG_E_vs_trueEnergy_contaminated_shower": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_contaminated_shower/trueEnergy_vs_JG_E/nu mu CC tot',
        "label": "JEnergy",
    },
    "JG_Len_vs_trueEnergy_contaminated_shower": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_contaminated_shower/trueEnergy_vs_JG_Len/nu mu CC tot',
        "label": "JStart",
    },
    "JS_E_vs_trueEnergy_contaminated_shower": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_contaminated_shower/trueEnergy_vs_JS_E/nu mu CC tot',
        "label": "JShF",
    },
    "DNN_Eest_vs_trueEnergy_contaminated_shower": {
        "filename": "mc_hists_v7.2.root",
        "histpath": 'mc_hists_contaminated_shower/trueEnergy_vs_DNN_Eest/nu mu CC tot',
        "label": "Eest",
    },
}
for key, value in datasets.items():
    value["upfile"] = uproot.open(value["filename"])
    hobj = value["upfile"][value["histpath"]]
    norm = 1.0
    if "norm" in value:
        norm = value["norm"]
    value["h"] = (hobj.values() * norm,
                  hobj.axis(axis=0).edges(),
                  hobj.axis(axis=1).edges(),
                  hobj.errors() * norm)


clrs_4 = defaultColorsMpl(0.4)
clrs_6 = defaultColorsMpl(0.6)


scale=1.5
nams = ["JG_E_vs_trueEnergy_simple", "JG_Len_vs_trueEnergy_simple", "JS_E_vs_trueEnergy_simple", "DNN_Eest_vs_trueEnergy_simple"]
colors = ['C0', 'C1', 'C2', 'C3']
fig, axe = plt.subplots(figsize=[4*scale,3*scale])
for i, item in enumerate(nams):
    figt, axet = plt.subplots()
    axet.set_xscale('log')
    h2 = datasets[item]['h']
    x = (h2[1][1:] + h2[1][:-1])/2.
    median, err = plotResolution(h2, axet)
    plt.close(figt)

    axe.plot(x, median, zorder=10,color=colors[i], label=datasets[item]["label"])
    axe.fill_between(x, median-err[0], median+err[1], color=clrs_4[i])

axe.set_xscale('log')
axe.set_yscale('log')
axe.set_xlabel(r'True E [GeV]')
axe.set_ylabel('Reco. energy [GeV]')
axe.set_xlim(1,100)
axe.set_ylim(1,100)
axe.plot(x,x,color="gray")
axe.legend()
axe.set_title(r"Pre-cuts $\bar{\nu}_\mu + \nu_\mu \mathrm{CC}$")

plt.tight_layout()
plt.grid(True)
plt.savefig("pre_cuts_numu.png")
plt.savefig("pre_cuts_numu.pdf")
plt.show()


nams = ["JG_E_vs_trueEnergy_pure_track", "JG_Len_vs_trueEnergy_pure_track", "JS_E_vs_trueEnergy_pure_track",
        "DNN_Eest_vs_trueEnergy_pure_track"]
colors = ['C0', 'C1', 'C2', 'C3']
fig, axe = plt.subplots(figsize=[4*scale,3*scale])
for i, item in enumerate(nams):
    figt, axet = plt.subplots()
    axet.set_xscale('log')
    h2 = datasets[item]['h']
    x = (h2[1][1:] + h2[1][:-1])/2.
    median, err = plotResolution(h2, axet)
    plt.close(figt)

    axe.plot(x, median, zorder=10,color=colors[i], label=datasets[item]["label"])
    axe.fill_between(x, median-err[0], median+err[1], color=clrs_4[i])

axe.set_xscale('log')
axe.set_yscale('log')
axe.set_xlabel(r'True E [GeV]')
axe.set_ylabel('Reco. energy [GeV]')
axe.set_xlim(1,100)
axe.set_ylim(1,100)
axe.plot(x,x,color="gray")
axe.legend()
axe.set_title(r"Pure_track $\bar{\nu}_\mu + \nu_\mu \mathrm{CC}$")

plt.tight_layout()
plt.grid(True)
plt.savefig("pure_track_numu.png")
plt.savefig("pure_track_numu.pdf")
plt.show()


nams = ["JG_E_vs_trueEnergy_pure_shower", "JG_Len_vs_trueEnergy_pure_shower", "JS_E_vs_trueEnergy_pure_shower",
        "DNN_Eest_vs_trueEnergy_pure_shower"]
colors = ['C0', 'C1', 'C2', 'C3']
fig, axe = plt.subplots(figsize=[4*scale,3*scale])
for i, item in enumerate(nams):
    figt, axet = plt.subplots()
    axet.set_xscale('log')
    h2 = datasets[item]['h']
    x = (h2[1][1:] + h2[1][:-1])/2.
    median, err = plotResolution(h2, axet)
    plt.close(figt)

    axe.plot(x, median, zorder=10,color=colors[i], label=datasets[item]["label"])
    axe.fill_between(x, median-err[0], median+err[1], color=clrs_4[i])

axe.set_xscale('log')
axe.set_yscale('log')
axe.set_xlabel(r'True E [GeV]')
axe.set_ylabel('Reco. energy [GeV]')
axe.set_xlim(1,100)
axe.set_ylim(1,100)
axe.plot(x,x,color="gray")
axe.legend()
axe.set_title(r"Pure_shower $\bar{\nu}_\mu + \nu_\mu \mathrm{CC}$")

plt.tight_layout()
plt.grid(True)
plt.savefig("pure_shower_numu.png")
plt.savefig("pure_shower_numu.pdf")
plt.show()


nams = ["JG_E_vs_trueEnergy_contaminated_track", "JG_Len_vs_trueEnergy_contaminated_track",
        "JS_E_vs_trueEnergy_contaminated_track",
        "DNN_Eest_vs_trueEnergy_contaminated_track"]
colors = ['C0', 'C1', 'C2', 'C3']
fig, axe = plt.subplots(figsize=[4*scale,3*scale])
for i, item in enumerate(nams):
    figt, axet = plt.subplots()
    axet.set_xscale('log')
    h2 = datasets[item]['h']
    x = (h2[1][1:] + h2[1][:-1])/2.
    median, err = plotResolution(h2, axet)
    plt.close(figt)

    axe.plot(x, median, zorder=10,color=colors[i], label=datasets[item]["label"])
    axe.fill_between(x, median-err[0], median+err[1], color=clrs_4[i])

axe.set_xscale('log')
axe.set_yscale('log')
axe.set_xlabel(r'True E [GeV]')
axe.set_ylabel('Reco. energy [GeV]')
axe.set_xlim(1,100)
axe.set_ylim(1,100)
axe.plot(x,x,color="gray")
axe.legend()
axe.set_title(r"Contaminated_track $\bar{\nu}_\mu + \nu_\mu \mathrm{CC}$")

plt.tight_layout()
plt.grid(True)
plt.savefig("contaminated_track_numu.png")
plt.savefig("contaminated_track_numu.pdf")
plt.show()


nams = ["JG_E_vs_trueEnergy_contaminated_shower", "JG_Len_vs_trueEnergy_contaminated_shower",
        "JS_E_vs_trueEnergy_contaminated_shower",
        "DNN_Eest_vs_trueEnergy_contaminated_shower"]
colors = ['C0', 'C1', 'C2', 'C3']
fig, axe = plt.subplots(figsize=[4*scale,3*scale])
for i, item in enumerate(nams):
    figt, axet = plt.subplots()
    axet.set_xscale('log')
    h2 = datasets[item]['h']
    x = (h2[1][1:] + h2[1][:-1])/2.
    median, err = plotResolution(h2, axet)
    plt.close(figt)

    axe.plot(x, median, zorder=10,color=colors[i], label=datasets[item]["label"])
    axe.fill_between(x, median-err[0], median+err[1], color=clrs_4[i])

axe.set_xscale('log')
axe.set_yscale('log')
axe.set_xlabel(r'True E [GeV]')
axe.set_ylabel('Reco. energy [GeV]')
axe.set_xlim(1,100)
axe.set_ylim(1,100)
axe.plot(x,x,color="gray")
axe.legend()
axe.set_title(r"Contaminated_shower $\bar{\nu}_\mu + \nu_\mu \mathrm{CC}$")

plt.tight_layout()
plt.grid(True)
plt.savefig("contaminated_shower_numu.png")
plt.savefig("contaminated_shower_numu.pdf")
plt.show()










