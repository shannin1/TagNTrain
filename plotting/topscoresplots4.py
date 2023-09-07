import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
from numpy import genfromtxt


fin = "../../CASEUtils/jet_images/2016JetHT_FGHruns_Jets.h5" #signal we're testing
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

#Now to Decide if the higgs in the kept events are passing or failing the Higgs cut, and keep events we care about
is_j1_moreHiggs = hbb_signal_1>hbb_signal_2
is_j2_moreHiggs = hbb_signal_1<hbb_signal_2
does_j1_pass_hbb = hbb_signal_1 > 0.98
does_j2_pass_hbb = hbb_signal_2 > 0.98


does_j1_fail_hbb = hbb_signal_1 < 0.8
does_j2_fail_hbb = hbb_signal_2 < 0.8

does_j1_loose_hbb = np.logical_and((hbb_signal_1 > 0.8), (hbb_signal_1 < 0.98))
does_j2_loose_hbb = np.logical_and((hbb_signal_2 > 0.8), (hbb_signal_2 < 0.98))
####
#here im looking at the CR where vae is between 0.00002 and 0.00004
vaecuts1 = np.logical_and((sig_score1>0.00001),(sig_score1<0.00004))  # these are the right vae cuts
vaecuts2 = np.logical_and((sig_score2>0.00001),(sig_score2<0.00004))   #right vae cuts
#vaecuts1 = sig_score1>0 
#vaecuts2 = sig_score2>0
keepj1 = np.logical_and(np.array(vaecuts1), is_j1_Y)
keepj2 = np.logical_and(np.array(vaecuts2), is_j2_Y)
keeps =  np.logical_or(keepj1,keepj2)
tossevent =  np.logical_not(keeps)

top1cut = toptagging_1<0.9
top2cut = toptagging_2<0.9
keeptop1 = np.logical_and(is_j1_moreHiggs, top1cut)
keeptop2 =  np.logical_and(is_j2_moreHiggs, top2cut)
keep_topcut = np.logical_or(keeptop1,keeptop2) #this is the booleans to keep events that pass topcut on Higgs where we cut TopScore > 0.9
####
keepevent = np.logical_and(keeps, keep_topcut)
###
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
pass_vaeH1 = sig_score1[np.logical_and(np.logical_and(pass_boolean,is_j1_moreHiggs),keepevent)]
pass_vaeY1 = sig_score1[np.logical_and(np.logical_and(pass_boolean,is_j1_Y),keepevent)]
pass_vaeH2 = sig_score2[np.logical_and(np.logical_and(pass_boolean,is_j2_moreHiggs),keepevent)]
pass_vaeY2 = sig_score2[np.logical_and(np.logical_and(pass_boolean,is_j2_Y),keepevent)]
pass_topH1 = toptagging_1[np.logical_and(np.logical_and(pass_boolean,is_j1_moreHiggs),keepevent)]
pass_topY1 = toptagging_1[np.logical_and(np.logical_and(pass_boolean,is_j1_Y),keepevent)]
pass_topH2 = toptagging_2[np.logical_and(np.logical_and(pass_boolean,is_j2_moreHiggs),keepevent)]
pass_topY2 = toptagging_2[np.logical_and(np.logical_and(pass_boolean,is_j2_Y),keepevent)]

loose_vaeH1 = sig_score1[np.logical_and(np.logical_and(loose_boolean,is_j1_moreHiggs),keepevent)]
loose_vaeH1 = sig_score1[np.logical_and(np.logical_and(loose_boolean,is_j1_moreHiggs),keepevent)]
loose_vaeH1 = sig_score1[np.logical_and(np.logical_and(loose_boolean,is_j1_moreHiggs),keepevent)]
loose_vaeY1 = sig_score1[np.logical_and(np.logical_and(loose_boolean,is_j1_Y),keepevent)]
loose_vaeH2 = sig_score2[np.logical_and(np.logical_and(loose_boolean,is_j2_moreHiggs),keepevent)]
loose_vaeY2 = sig_score2[np.logical_and(np.logical_and(loose_boolean,is_j2_Y),keepevent)]
loose_topH1 = toptagging_1[np.logical_and(np.logical_and(loose_boolean,is_j1_moreHiggs),keepevent)]
loose_topY1 = toptagging_1[np.logical_and(np.logical_and(loose_boolean,is_j1_Y),keepevent)]
loose_topH2 = toptagging_2[np.logical_and(np.logical_and(loose_boolean,is_j2_moreHiggs),keepevent)]
loose_topY2 = toptagging_2[np.logical_and(np.logical_and(loose_boolean,is_j2_Y),keepevent)]

fail_vaeH1 = sig_score1[np.logical_and(np.logical_and(fail_boolean,is_j1_moreHiggs),keepevent)]
fail_vaeY1 = sig_score1[np.logical_and(np.logical_and(fail_boolean,is_j1_Y),keepevent)]
fail_vaeH2 = sig_score2[np.logical_and(np.logical_and(fail_boolean,is_j2_moreHiggs),keepevent)]
fail_vaeY2 = sig_score2[np.logical_and(np.logical_and(fail_boolean,is_j2_Y),keepevent)]
fail_topH1 = toptagging_1[np.logical_and(np.logical_and(fail_boolean,is_j1_moreHiggs),keepevent)]
fail_topY1 = toptagging_1[np.logical_and(np.logical_and(fail_boolean,is_j1_Y),keepevent)]
fail_topH2 = toptagging_2[np.logical_and(np.logical_and(fail_boolean,is_j2_moreHiggs),keepevent)]
fail_topY2 = toptagging_2[np.logical_and(np.logical_and(fail_boolean,is_j2_Y),keepevent)]

#Masses to be sent to csv
pass_mH1 = tot_mj1[np.logical_and(np.logical_and(pass_boolean,is_j1_moreHiggs),keepevent)]
pass_mY1 = tot_mj1[np.logical_and(np.logical_and(pass_boolean,is_j1_Y),keepevent)]
pass_mH2 = tot_mj2[np.logical_and(np.logical_and(pass_boolean,is_j2_moreHiggs),keepevent)]
pass_mY2 = tot_mj2[np.logical_and(np.logical_and(pass_boolean,is_j2_Y),keepevent)]
pass_mjj = tot_mjj[np.logical_and(pass_boolean,keepevent)]


loose_mH1 = tot_mj1[np.logical_and(np.logical_and(loose_boolean,is_j1_moreHiggs),keepevent)]
loose_mY1 = tot_mj1[np.logical_and(np.logical_and(loose_boolean,is_j1_Y),keepevent)]
loose_mH2 = tot_mj2[np.logical_and(np.logical_and(loose_boolean,is_j2_moreHiggs),keepevent)]
loose_mY2 = tot_mj2[np.logical_and(np.logical_and(loose_boolean,is_j2_Y),keepevent)]
loose_mjj = tot_mjj[np.logical_and(loose_boolean,keepevent)]

fail_mH1 = tot_mj1[np.logical_and(np.logical_and(fail_boolean,is_j1_moreHiggs),keepevent)]
fail_mY1 = tot_mj1[np.logical_and(np.logical_and(fail_boolean,is_j1_Y),keepevent)]
fail_mH2 = tot_mj2[np.logical_and(np.logical_and(fail_boolean,is_j2_moreHiggs),keepevent)]
fail_mY2 = tot_mj2[np.logical_and(np.logical_and(fail_boolean,is_j2_Y),keepevent)]
fail_mjj = tot_mjj[np.logical_and(fail_boolean,keepevent)]

#Save Pass Masses arrays as csv
np.savetxt("Bkg/Pass/"+name+"_VAECR_Mjj_BKG.csv", pass_mjj, delimiter=",")
np.savetxt("Bkg/Pass/"+name+"_VAECR_mY_BKG.csv", np.append(pass_mY1, pass_mY2), delimiter=",")
np.savetxt("Bkg/Pass/"+name+"_VAECR_mH_BKG.csv", np.append(pass_mH1, pass_mH2), delimiter=",")


#Save Loose Masses arrays as csv
np.savetxt("Bkg/Loose/"+name+"_VAECR_Mjj_BKG.csv", loose_mjj, delimiter=",")
np.savetxt("Bkg/Loose/"+name+"_VAECR_mY_BKG.csv", np.append(loose_mY1, loose_mY2), delimiter=",")
np.savetxt("Bkg/Loose/"+name+"_VAECR_mH_BKG.csv", np.append(loose_mH1, loose_mH2), delimiter=",")


#Save Fail Masses arrays as csv
np.savetxt("Bkg/Fail/"+name+"_VAECR_Mjj_BKG.csv", fail_mjj, delimiter=",")
np.savetxt("Bkg/Fail/"+name+"_VAECR_mY_BKG.csv", np.append(fail_mY1, fail_mY2), delimiter=",")
np.savetxt("Bkg/Fail/"+name+"_VAECR_mH_BKG.csv", np.append(fail_mH1, fail_mH2), delimiter=",")
'''
#TopControlRegion
topCR = np.logical_and(np.logical_not(keep_topcut), keeps) #events that passed vae preselection but are being tossed bc of topTagger cut
vaeCR = tossevent #events being tossed bc of vae score cut
print('total_length = {} , kept = {} , failed vae = {} , failed top after passing vae = {}'.format(len(keepevent), keepevent.sum(), tossevent.sum(), topCR.sum()))
#TopCR Regions 
topCR_vae_H1 = sig_score1[np.logical_and(is_j1_moreHiggs, topCR)]
topCR_vae_Y1 = sig_score1[np.logical_and(is_j1_Y, topCR)]
topCR_vae_H2 = sig_score2[np.logical_and(is_j2_moreHiggs, topCR)]
topCR_vae_Y2 = sig_score2[np.logical_and(is_j2_Y, topCR)]
topCR_top_H1 = toptagging_1[np.logical_and(is_j1_moreHiggs, topCR)]
topCR_top_Y1 = toptagging_1[np.logical_and(is_j1_Y, topCR)]
topCR_top_H2 = toptagging_2[np.logical_and(is_j2_moreHiggs, topCR)]
topCR_top_Y2 = toptagging_2[np.logical_and(is_j2_Y, topCR)]

#VAECR Regions
CR_vae_H1 = sig_score1[np.logical_and(is_j1_moreHiggs, vaeCR)]
CR_vae_Y1 = sig_score1[np.logical_and(is_j1_Y, vaeCR)]
CR_vae_H2 = sig_score2[np.logical_and(is_j2_moreHiggs, vaeCR)]
CR_vae_Y2 = sig_score2[np.logical_and(is_j2_Y, vaeCR)]
CR_top_H1 = toptagging_1[np.logical_and(is_j1_moreHiggs, vaeCR)]
CR_top_Y1 = toptagging_1[np.logical_and(is_j1_Y, vaeCR)]
CR_top_H2 = toptagging_2[np.logical_and(is_j2_moreHiggs, vaeCR)]
CR_top_Y2 = toptagging_2[np.logical_and(is_j2_Y, vaeCR)]

'''
#plot grid of SR 
plt.hist2d(np.append(fail_vaeH1,fail_vaeH2 ), np.append(fail_topH1, fail_topH2), bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoencoder Score')
plt.title('Higgs Fail Region')
plt.savefig('MX3000_MY300HiggsFailRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(fail_vaeY1,fail_vaeY2), np.append(fail_topY1, fail_topY2),bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Y Fail Region')
plt.savefig('MX3000_MY300YFailRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(loose_vaeH1,loose_vaeH2 ), np.append(loose_topH1, loose_topH2), bins = 10, norm = LogNorm())
#plt.tick_params(axis='both', which='minor', labelsize=6)
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Higgs Loose Region')
plt.savefig('MX3000_MY300HiggsLooseRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(loose_vaeY1,loose_vaeY2 ), np.append(loose_topY1, loose_topY2),bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Y Loose Region')
plt.savefig('MX3000_MY300YLooseRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(pass_vaeH1,pass_vaeH2 ), np.append(pass_topH1, pass_topH2),  bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Higgs Pass Region')
plt.savefig('MX3000_MY300HiggsPassRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(pass_vaeY1,pass_vaeY2 ), np.append(pass_topY1, pass_topY2),  bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('Y Pass Region')
plt.savefig('MX3000_MY300YPassRegion.png',bbox_inches='tight',dpi=100)
plt.close()



'''
plt.hist2d(np.append(topCR_vae_H1,topCR_vae_H2 ), np.append(topCR_top_H1, topCR_top_H2),  bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('CR Top Tagger(passed autoencoder but failed toptagger) H Jet Region')
plt.savefig('MX3000_MY300CR_HTopRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(topCR_vae_Y1,topCR_vae_Y2 ), np.append(topCR_top_Y1, topCR_top_Y2),  bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('CR Top Tagger(passed autoencoder but failed toptagger) Y Jet Region')
plt.savefig('MX3000_MY300CR_YTopRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(CR_vae_H1,CR_vae_H2 ), np.append(CR_top_H1, CR_top_H2),  bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('CR (failed autoencoder) H Jet Region')
plt.savefig('MX3000_MY300CR_HRegion.png',bbox_inches='tight',dpi=100)
plt.close()

plt.hist2d(np.append(CR_vae_Y1,CR_vae_Y2 ), np.append(CR_top_Y1, CR_top_Y2),  bins = 10, norm = LogNorm())
plt.xticks(rotation=90)
plt.colorbar()
plt.ylabel('Top Tagger Score')
plt.xlabel('Autoenconcoder Score')
plt.title('CR (failed autoencoder) Y Jet Region')
plt.savefig('MX3000_MY300CR_YRegion.png',bbox_inches='tight',dpi=100)
plt.close()

'''