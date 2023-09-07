import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
from numpy import genfromtxt

fin = "../../CASEUtils/jet_images/Bkg_JetImages/arrTTToHadronic2016APV.h5" #signal we're testing
plot_dir = "plotting/plots/"
model_name = "../../CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 

hbb_signal_1 = h5py.File(fin, "r")['jet1_extraInfo'][:,-1]
hbb_signal_2 = h5py.File(fin, "r")['jet2_extraInfo'][:,-1]

fsignal1 = h5py.File(fin, "r")["j1_images"][()]
fsignal2 = h5py.File(fin, "r")["j2_images"][()]

model = tf.keras.models.load_model(model_name)
fsignal1 = np.expand_dims(fsignal1, axis = -1)
fsignal2 = np.expand_dims(fsignal2, axis = -1)

reco_signal1 = model.predict(fsignal1, batch_size = 1)
reco_signal2 = model.predict(fsignal2, batch_size = 1)

sig_score1 =  np.mean(np.square(reco_signal1 - fsignal1), axis=(1,2)).reshape(-1)
sig_score2 =  np.mean(np.square(reco_signal2 - fsignal2), axis=(1,2)).reshape(-1)

#SigScore here is for HadronicAPV
#sig_score1 =  np.array(genfromtxt("sig_score1.csv", delimiter=","))
#sig_score2 =  np.array(genfromtxt("sig_score2.csv", delimiter=","))
#Logical operations to decide which event has a Y with a score vae score above 0.00005
is_j1_Y = hbb_signal_1<hbb_signal_2
is_j2_Y = hbb_signal_1>hbb_signal_2

vaecuts1 = (sig_score1>0.00005)
vaecuts2 = (sig_score2>0.00005)

keepj1 = np.logical_and(np.array(vaecuts1), is_j1_Y)
keepj2 = np.logical_and(np.array(vaecuts2), is_j2_Y)

keepevent =  np.logical_or(keepj1,keepj2)

#Now to Decide if the higgs in the kept events are passing or failing the Higgs cut, and keep events we care about
is_j1_moreHiggs = hbb_signal_1>hbb_signal_2
is_j2_moreHiggs = hbb_signal_1<hbb_signal_2
does_j1_pass_hbb = hbb_signal_1 > 0.98
does_j2_pass_bbb = hbb_signal_2 > 0.98

is_j1_higgs = np.logical_and(is_j1_moreHiggs, does_j1_pass_hbb)[keepevent]
is_j2_higgs = np.logical_and(is_j2_moreHiggs, does_j2_pass_bbb)[keepevent]

pass_boolean = np.logical_or(is_j1_higgs, is_j2_higgs)
fail_boolean = np.logical_not(pass_boolean)

tot_mjj = np.array((h5py.File(fin, "r")['jet_kinematics'][:,0]).reshape(-1))[keepevent]
tot_mj1 = np.array(h5py.File(fin, "r")['jet_kinematics'][:,5]).reshape(-1)[keepevent]
tot_mj2 = np.array(h5py.File(fin, "r")['jet_kinematics'][:,-5]).reshape(-1)[keepevent]






name = fin.split('/')[-1].split('.')[0]

#Signal Pass jet masses cut and VAE
Mjj = tot_mjj[pass_boolean]
mj1 = tot_mj1[pass_boolean]
mj2 = tot_mj2[pass_boolean]

#Signal Fail jet masses cut and VAE
FMjj = tot_mjj[fail_boolean]
Fmj1 = tot_mj1[fail_boolean]
Fmj2 = tot_mj2[fail_boolean]

#Save Pass Masses arrays as csv
np.savetxt("Bkg/Pass/"+name+"_Mjj_BKG.csv", Mjj, delimiter=",")
np.savetxt("Bkg/Pass/"+name+"_mj1_BKG.csv", mj1, delimiter=",")
np.savetxt("Bkg/Pass/"+name+"_mj2_BKG.csv", mj2, delimiter=",")


#Save Fail Masses arrays as csv
np.savetxt("Bkg/Fail/"+name+"Mjj_BKG.csv", FMjj, delimiter=",")
np.savetxt("Bkg/Fail/"+name+"mj1_BKG.csv", Fmj1, delimiter=",")
np.savetxt("Bkg/Fail/"+name+"mj2_BKG.csv", Fmj2, delimiter=",")