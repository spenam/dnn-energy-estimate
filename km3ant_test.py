import km3ant as ka
import numpy as np


#####################################################
# Defining data sets
# ------------------
#
# In km3ant, datasets are declared in a dictionnary.
# Each dataset is defined as pair :
#
# ``<dataset label>:{"filename":<dstfilepath>}``
#

dataset = {
    "data": {
        "filename": "/sps/km3net/users/alflazo/dstProd/v7.1_v7.2_jsh/mcv7.1.gsg_anti-elec-CC_100-500GeV.sirene.jorcarec.jsh.aanet.dst_merged.root"
    }
}


#####################################################
# Setting up the pipe
# -------------------
#
# In this part, we will create the pipe layout
#
# Creating the pipe manager
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# All the modules will be registered to this module. It is also the
# one taking care of creating the pump.  It takes as argument the
# dictionnary containing the datasets.  Here, we also decide to set
# the step_size, i.e. the number of events in each data blob.

manager = ka.pipeManager(dataset, step_size=1000000)

#####################################################
# Exposure handler
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The exposure handler is a basic module that will keep track of the
# exposure time.  This is a mandatory module because per default
# histogram are filled in rates.
#
# The name of the module has to be specified, this way you can imagine
# having multiple exposure module in paralel, but that's something we
# will discuss in more advancede examples.

manager.append_module("exposure", ka.exposureHandler)

#####################################################
# Event weighter
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As the name might suggest, this module will compute the weight of
# each event.  In this example, as we use data, the weight should be
# set to 1 for all events, and that's what ``{'cst_weight':1}`` does.

manager.append_module("weighter", ka.eventWeighter, kwargs={"cst_weight": 1})

#####################################################
# Adding some cuts
# ~~~~~~~~~~~~~~~~~~~~~
#
# Cut are handled through masks. The idea is to compute
# a vector of boolean that represant which events
# survive the cuts.
#
# Here, we apply two cuts :
#
# ``E.trks.lik[:,0]`` point to first track likelihood
# and is required to be above 40.
#
# ``E.trks.E[:,0]`` point to first track energy
# and is required to be below 100 GeV.

cuts = [ka.lin_cut("E.trks.lik[:,0]", ">", 40),
        ka.lin_cut("E.trks.E[:,0]", "<", 100)]

manager.append_module("my_simple_cuts", ka.cutsHandler, kwargs={"cuts": cuts})


#####################################################
# Adding some histograms
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have some cuts, let's fill histogram, with and without the cuts.
#
# First, we are declaring a list of kc.histogram objects.  This object
# expects a name, the variable in string format and the bins.  The 2
# first histograms are 1D histograms, for cos theta and energy. Note
# that energy will have log binning. The third one shows how to do 2D
# histograms.
#
# Then, we declare 2 histogram handler that we start with the same
# list of histogram.  The first one doesn't have any mask declared, it
# will use every events.  The second one receive as ``mask_name``
# argument the name of cut handler declared previously. It will use
# only events surviving these cuts.

hists = [
    ka.histogram("cos_theta", "E.trks.dir.z[:,0]", np.linspace(-1, 1, 21)),
    ka.histogram("energy_log", "E.trks.E[:,0]", 10 ** np.linspace(0, 4, 41)),
    ka.histogram(
        "energy_vs_cos_theta",
        ["E.trks.E[:,0]", "E.trks.dir.z[:,0]"],
        [10 ** np.linspace(0, 4, 41), np.linspace(-1, 1, 21)],
    ),
]

manager.append_module(
    "my_simple_hists_without_cuts", ka.histogramHandler, kwargs={"histograms": hists}
)

manager.append_module(
    "my_simple_hists_with_cuts",
    ka.histogramHandler,
    kwargs={"mask_name": "my_simple_cuts", "histograms": hists},
)

#####################################################
# Run the pipeline
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Finally, we have defined what we want to do, we can run the analysis.

manager.run()
