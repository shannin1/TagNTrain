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


#SigScore here is for HadronicAPV
#sig_score1 =  np.array(genfromtxt("sig_score1.csv", delimiter=","))
#sig_score2 =  np.array(genfromtxt("sig_score2.csv", delimiter=","))
#Logical operations to decide which event has a Y with a score vae score above 0.00005
is_j1_Y = hbb_signal_1<hbb_signal_2
is_j2_Y = hbb_signal_1>hbb_signal_2
'''
vaecuts1 = (sig_score1>0.00005)
vaecuts2 = (sig_score2>0.00005)
keepj1 = np.logical_and(np.array(vaecuts1), is_j1_Y)
keepj2 = np.logical_and(np.array(vaecuts2), is_j2_Y)
keepevent =  np.logical_or(keepj1,keepj2)
tossevent =  np.logical_not(keepevent)
'''
#Now to Decide if the higgs in the kept events are passing or failing the Higgs cut, and keep events we care about
is_j1_moreHiggs = hbb_signal_1>hbb_signal_2
is_j2_moreHiggs = hbb_signal_1<hbb_signal_2
does_j1_pass_hbb = hbb_signal_1 > 0.98
does_j2_pass_hbb = hbb_signal_2 > 0.98


does_j1_fail_hbb = hbb_signal_1 < 0.8
does_j2_fail_hbb = hbb_signal_2 < 0.8

does_j1_loose_hbb = np.logical_and((hbb_signal_1 > 0.8), (hbb_signal_1 < 0.98))
does_j2_loose_hbb = np.logical_and((hbb_signal_2 > 0.8), (hbb_signal_2 < 0.98))

is_j1_higgs = np.logical_and(is_j1_moreHiggs, does_j1_pass_hbb)
is_j2_higgs = np.logical_and(is_j2_moreHiggs, does_j2_pass_hbb)

is_j1_loose = np.logical_and(is_j1_moreHiggs, does_j1_loose_hbb)
is_j2_loose = np.logical_and(is_j2_moreHiggs, does_j2_loose_hbb)

is_j1_fail = np.logical_and(is_j1_moreHiggs, does_j1_fail_hbb)
is_j2_fail = np.logical_and(is_j2_moreHiggs, does_j2_fail_hbb)

pass_boolean = np.logical_or(is_j1_higgs, is_j2_higgs)
loose_boolean = np.logical_or(is_j1_loose, is_j2_loose)
fail_boolean = np.logical_or(is_j1_fail, is_j2_fail)

tot_mjj = np.array(h5py.File(fin, "r")['jet_kinematics'][:,0]).reshape(-1)
tot_mj1 = np.array(h5py.File(fin, "r")['jet_kinematics'][:,5]).reshape(-1)
tot_mj2 = np.array(h5py.File(fin, "r")['jet_kinematics'][:,-5]).reshape(-1)






name = fin.split('/')[-1].split('.')[0]




# Signal region toptagging and VAE scores
pass_vaeH1 = sig_score1[np.logical_and(pass_boolean,is_j1_moreHiggs)]
pass_vaeY1 = sig_score1[np.logical_and(pass_boolean,is_j1_Y)]
pass_vaeH2 = sig_score2[np.logical_and(pass_boolean,is_j2_moreHiggs)]
pass_vaeY2 = sig_score2[np.logical_and(pass_boolean,is_j2_Y)]
pass_topH1 = toptagging_1[np.logical_and(pass_boolean,is_j1_moreHiggs)]
pass_topY1 = toptagging_1[np.logical_and(pass_boolean,is_j1_Y)]
pass_topH2 = toptagging_2[np.logical_and(pass_boolean,is_j2_moreHiggs)]
pass_topY2 = toptagging_2[np.logical_and(pass_boolean,is_j2_Y)]

loose_vaeH1 = sig_score1[np.logical_and(loose_boolean,is_j1_moreHiggs)]
loose_vaeY1 = sig_score1[np.logical_and(loose_boolean,is_j1_Y)]
loose_vaeH2 = sig_score2[np.logical_and(loose_boolean,is_j2_moreHiggs)]
loose_vaeY2 = sig_score2[np.logical_and(loose_boolean,is_j2_Y)]
loose_topH1 = toptagging_1[np.logical_and(loose_boolean,is_j1_moreHiggs)]
loose_topY1 = toptagging_1[np.logical_and(loose_boolean,is_j1_Y)]
loose_topH2 = toptagging_2[np.logical_and(loose_boolean,is_j2_moreHiggs)]
loose_topY2 = toptagging_2[np.logical_and(loose_boolean,is_j2_Y)]

fail_vaeH1 = sig_score1[np.logical_and(fail_boolean,is_j1_moreHiggs)]
fail_vaeY1 = sig_score1[np.logical_and(fail_boolean,is_j1_Y)]
fail_vaeH2 = sig_score2[np.logical_and(fail_boolean,is_j2_moreHiggs)]
fail_vaeY2 = sig_score2[np.logical_and(fail_boolean,is_j2_Y)]
fail_topH1 = toptagging_1[np.logical_and(fail_boolean,is_j1_moreHiggs)]
fail_topY1 = toptagging_1[np.logical_and(fail_boolean,is_j1_Y)]
fail_topH2 = toptagging_2[np.logical_and(fail_boolean,is_j2_moreHiggs)]
fail_topY2 = toptagging_2[np.logical_and(fail_boolean,is_j2_Y)]



'''

#plot grid of SR 
plt.hist2d(np.append(fail_vaeH1,fail_vaeH2 ), np.append(fail_topH1, fail_topH2), bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoencoder Score')
plt.title('Higgs Fail Region')
plt.savefig('TTToHadronicHiggsFailRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(fail_vaeY1,fail_vaeY2), np.append(fail_topY1, fail_topY2),bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Y Fail Region')
plt.savefig('TTToHadronicYFailRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(loose_vaeH1,loose_vaeH2 ), np.append(loose_topH1, loose_topH2), bins = 10, norm = LogNorm())
#plt.tick_params(axis='both', which='minor', labelsize=6)
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Higgs Loose Region')
plt.savefig('TTToHadronicHiggsLooseRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(loose_vaeY1,loose_vaeY2 ), np.append(loose_topY1, loose_topY2),bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Y Loose Region')
plt.savefig('TTToHadronicYLooseRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(pass_vaeH1,pass_vaeH2 ), np.append(pass_topH1, pass_topH2),  bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Higgs Pass Region')
plt.savefig('TTToHadronicHiggsPassRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(pass_vaeY1,pass_vaeY2 ), np.append(pass_topY1, pass_topY2),  bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Y Pass Region')
plt.savefig('TTToHadronicYPassRegion.png',bbox_inches='tight',dpi=100)
plt.close()

'''



