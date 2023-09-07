import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
from numpy import genfromtxt

fin = "../../CASEUtils/jet_images/top_smaller_TTToHadronic.h5" #signal we're testing
plot_dir = "plotting/plots/"
model_name = "../../CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 

hbb_signal_1 = h5py.File(fin, "r")['jet1_extraInfo'][:,-2]
hbb_signal_2 = h5py.File(fin, "r")['jet2_extraInfo'][:,-2]

toptagging_1 = h5py.File(fin, "r")['jet1_extraInfo'][:,-1]
toptagging_2 = h5py.File(fin, "r")['jet2_extraInfo'][:,-1]

fsignal1 = h5py.File(fin, "r")["j1_images"][()]
fsignal2 = h5py.File(fin, "r")["j2_images"][()]

model = tf.keras.models.load_model(model_name)
fsignal1 = np.expand_dims(fsignal1, axis = -1)
fsignal2 = np.expand_dims(fsignal2, axis = -1)

reco_signal1 = model.predict(fsignal1, batch_size = 1)
reco_signal2 = model.predict(fsignal2, batch_size = 1)

sig_score1 =  np.mean(np.square(reco_signal1 - fsignal1), axis=(1,2)).reshape(-1)
sig_score2 =  np.mean(np.square(reco_signal2 - fsignal2), axis=(1,2)).reshape(-1)


'''
#plotting
plt.hist2d(sig_score1, toptagging_1, bins=40)#, norm=LogNorm())
plt.colorbar()
plt.xlabel('autoencoder_score')
plt.ylabel('top_tagging_score')
plt.savefig("QCDHT1500to2000TopVsVAE_jet1.png")
plt.show()

#plotting
plt.hist2d(sig_score2, toptagging_2, bins=40)#, norm=LogNorm())
plt.colorbar()
plt.xlabel('autoencoder_score')
plt.ylabel('top_tagging_score')
plt.savefig("QCDHT1500to200TopVsVAE_jet2.png")
plt.show()


'''

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
tossevent = np.logical_not(keepevent)

#Now to Decide if the higgs in the kept events are passing or failing the Higgs cut, and keep events we care about
is_j1_moreHiggs = hbb_signal_1>hbb_signal_2
is_j2_moreHiggs = hbb_signal_1<hbb_signal_2
does_j1_pass_hbb = hbb_signal_1 > 0.98
does_j2_pass_hbb = hbb_signal_2 > 0.98


does_j1_fail_hbb = hbb_signal_1 < 0.8
does_j2_fail_hbb = hbb_signal_2 < 0.8

does_j1_loose_hbb = np.logical_and((hbb_signal_1 > 0.8), (hbb_signal_1 < 0.98))
does_j2_loose_hbb = np.logical_and((hbb_signal_2 > 0.8), (hbb_signal_2 < 0.98))

is_j1_higgs = np.logical_and(is_j1_moreHiggs, does_j1_pass_hbb)[keepevent]
is_j2_higgs = np.logical_and(is_j2_moreHiggs, does_j2_pass_hbb)[keepevent]

is_j1_loose = np.logical_and(is_j1_moreHiggs, does_j1_loose_hbb)[keepevent]
is_j2_loose = np.logical_and(is_j2_moreHiggs, does_j2_loose_hbb)[keepevent]

is_j1_fail = np.logical_and(is_j1_moreHiggs, does_j1_fail_hbb)[keepevent]
is_j2_fail = np.logical_and(is_j2_moreHiggs, does_j2_fail_hbb)[keepevent]

pass_boolean = np.logical_or(is_j1_higgs, is_j2_higgs)
loose_boolean = np.logical_or(is_j1_loose, is_j2_loose)
fail_boolean = np.logical_or(is_j1_fail, is_j2_fail)

tot_mjj = np.array(h5py.File(fin, "r")['jet_kinematics'][:,0]).reshape(-1)[keepevent]
tot_mj1 = np.array(h5py.File(fin, "r")['jet_kinematics'][:,5]).reshape(-1)[keepevent]
tot_mj2 = np.array(h5py.File(fin, "r")['jet_kinematics'][:,-5]).reshape(-1)[keepevent]






name = fin.split('/')[-1].split('.')[0]




# Signal region toptagging and VAE scores
pass_vae1 = sig_score1[keepevent][pass_boolean]
pass_vae2 = sig_score2[keepevent][pass_boolean]
pass_top1 = toptagging_1[keepevent][pass_boolean]
pass_top2 = toptagging_2[keepevent][pass_boolean]

loose_vae1 = sig_score1[keepevent][loose_boolean]
loose_vae2 = sig_score2[keepevent][loose_boolean]
loose_top1 = toptagging_1[keepevent][loose_boolean]
loose_top2 = toptagging_2[keepevent][loose_boolean]

fail_vae1 = sig_score1[keepevent][fail_boolean]
fail_vae2 = sig_score2[keepevent][fail_boolean]
fail_top1 = toptagging_1[keepevent][fail_boolean]
fail_top2 = toptagging_2[keepevent][fail_boolean]

#Control region
CR_vae1 = sig_score1[tossevent]
CR_vae2 = sig_score2[tossevent]
CR_top1 = toptagging_1[tossevent]
CR_top2 = toptagging_2[tossevent]


#plot grid of SR 
plt.hist2d(fail_vae1, fail_top1, bins = 20, norm = LogNorm())
#plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoencoder Score')
plt.title('Jet 1 Fail Region')
plt.savefig('TTToHadronicJet1FailRegion.png')

plt.hist2d(fail_vae2, fail_top2, bins = 20, norm = LogNorm())
#plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Jet 2 Fail Region')
plt.savefig('TTToHadronicJet2FailRegion.png')

plt.hist2d(loose_vae1, loose_top1, bins = 20, norm = LogNorm())
#plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Jet 1 Loose Region')
plt.savefig('TTToHadronicJet1LooseRegion.png')

plt.hist2d(loose_vae2, loose_top2, bins = 20, norm = LogNorm())
#plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Jet 2 Loose Region')
plt.savefig('TTToHadronic0Jet2LooseRegion.png')

plt.hist2d(pass_vae1, pass_top1,  bins = 20, norm = LogNorm())
#plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Jet 1 Pass Region')
plt.savefig('TTToHadronicJet1PassRegion.png')

plt.hist2d(pass_vae2, pass_top2,  bins = 20, norm = LogNorm())
#plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Jet 2 Pass Region')
plt.savefig('TTToHadronicJet2PassRegion.png')

plt.hist2d(CR_vae1, CR_top1, bins = 20, norm = LogNorm())
#plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Jet 1 Control Region')
plt.savefig('TTToHadronicJet1ControlRegion.png')

plt.hist2d(CR_vae2, CR_top2, bins = 20, norm = LogNorm())
#plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Jet 2 Control Region')
plt.savefig('TTToHadronicJet2ControlRegion.png')




'''
fig, axs = plt.subplots(4, 2)

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)


axs[0, 0].hist2d(fail_vae1, fail_top1, bins = 50, norm = LogNorm())
axs[0, 0].set_title('Jet 1 Fail Region',fontsize = 7)
axs[0, 0].tick_params(axis='both', which='minor', labelsize=7)
axs[0, 1].hist2d(fail_vae2, fail_top2, bins = 50, norm = LogNorm())
axs[0, 1].set_title('Jet 2 Fail Region',fontsize = 7)
axs[0, 1].tick_params(axis='both', which='minor', labelsize=7)
axs[1, 0].hist2d(loose_vae1, loose_top1, bins = 50, norm = LogNorm())
axs[1, 0].set_title('Jet 1 Loose Region',fontsize = 7)
axs[1, 0].tick_params(axis='both', which='minor', labelsize=7)
axs[1, 1].hist2d(loose_vae2, loose_top2, bins = 50, norm = LogNorm())
axs[1, 1].set_title('Jet 2 Loose Region',fontsize = 7)
axs[1, 1].tick_params(axis='both', which='minor', labelsize=5)
axs[2, 0].hist2d(pass_vae1, pass_top1,  bins = 50, norm = LogNorm())
axs[2, 0].set_title('Jet 1 Pass Region',fontsize = 7)
axs[2, 0].tick_params(axis='both', which='minor', labelsize=7)
axs[2, 1].hist2d(pass_vae2, pass_top2,  bins = 50, norm = LogNorm())
axs[2, 1].set_title('Jet 2 Pass Region',fontsize = 7)
axs[2, 1].tick_params(axis='both', which='minor', labelsize=7)
axs[3, 0].hist2d(CR_vae1, CR_top1, bins = 50, norm = LogNorm())
axs[3, 0].set_title('Jet 1 Control Region',fontsize = 7)
axs[3, 0].tick_params(axis='both', which='minor', labelsize=7)
axs[3, 1].hist2d(CR_vae2, CR_top2, bins = 50, norm = LogNorm())
axs[3, 1].set_title('Jet 2 Control Region',fontsize = 7)
axs[3, 1].tick_params(axis='both', which='minor', labelsize=7)

for ax in axs.flat:
    ax.set_xlabel('Autoencoder Score',  fontsize = 7)
    ax.set_ylabel('Top Tagger Score',  fontsize = 7)


# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()       

#fig.colorbar(ax = axs)

fig.show()
fig.savefig('gridQCD1500to2000.png')
'''

