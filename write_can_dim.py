import ROOT

outTree = ROOT.TFile("results/v7.10/SelectedEvents_DNN_all.root","UPDATE")
outTree.cd()
vec = ROOT.TVectorD(3)
vec[0] = 0.000
vec[1] = 275.600
vec[2] = 108.200
vec.Write("CanDimensions")
outTree.Close()

