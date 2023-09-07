import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
sig = "../../CASEUtils/jet_images/topMX3000_MY300.h5" #signal we're testing
qcd1 = "../../CASEUtils/jet_images/topQCDHT1500to2000.h5" #background we're testing
qcd2 = "../../CASEUtils/jet_images/topQCDHT1000to1500.h5" #background we're testing
tt = "../../CASEUtils/jet_images/top_smaller_TTToHadronic.h5" #background we're testing

plot_dir = "plotting/plots/"
model_name = "../../CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 
jet_type1 = 'j1_images' # or 'j1_images'
jet_type2 = 'j2_images'

sigjets1 = np.expand_dims(h5py.File(sig, "r")[jet_type1][()], axis =-1)
qcdjets1 =np.expand_dims(h5py.File(qcd1, "r")[jet_type1][()], axis =-1)
qcbjets1 = np.expand_dims(h5py.File(qcd2, "r")[jet_type1][()], axis =-1)
ttjets1 =np.expand_dims(h5py.File(tt, "r")[jet_type1][()], axis =-1)

sigjets2 = np.expand_dims(h5py.File(sig, "r")[jet_type2][()], axis =-1)
qcdjets2 =np.expand_dims(h5py.File(qcd1, "r")[jet_type2][()], axis =-1)
qcbjets2 = np.expand_dims(h5py.File(qcd2, "r")[jet_type2][()], axis =-1)
ttjets2 =np.expand_dims(h5py.File(tt, "r")[jet_type2][()], axis =-1)

hbb_sig_1 = np.expand_dims(h5py.File(sig, "r")['jet1_extraInfo'][:,-2], axis =-1)
hbb_sig_2 = np.expand_dims(h5py.File(sig, "r")['jet2_extraInfo'][:,-2], axis =-1)
is_sig1_Y = hbb_sig_1<hbb_sig_2
is_sig2_Y = np.logical_not(is_sig1_Y)

hbb_tt_1 = np.expand_dims(h5py.File(tt, "r")['jet1_extraInfo'][:,-2], axis =-1)
hbb_tt_2 = np.expand_dims(h5py.File(tt, "r")['jet2_extraInfo'][:,-2], axis =-1)
is_tt1_Y = hbb_tt_1<hbb_tt_2
is_tt2_Y = np.logical_not(is_tt1_Y)

hbb_qcd_1 = np.expand_dims(h5py.File(qcd1, "r")['jet1_extraInfo'][:,-2], axis =-1)
hbb_qcd_2 = np.expand_dims(h5py.File(qcd1, "r")['jet2_extraInfo'][:,-2], axis =-1)
is_qcd1_Y = hbb_qcd_1<hbb_qcd_2
is_qcd2_Y = np.logical_not(is_qcd1_Y)

hbb_qcb_1 = np.expand_dims(h5py.File(qcd2, "r")['jet1_extraInfo'][:,-2], axis =-1)
hbb_qcb_2 = np.expand_dims(h5py.File(qcd2, "r")['jet2_extraInfo'][:,-2], axis =-1)
is_qcb1_Y = hbb_qcb_1<hbb_qcb_2
is_qcb2_Y = np.logical_not(is_qcb1_Y)






model = tf.keras.models.load_model(model_name)
reco_sig1 = model.predict(sigjets1, batch_size = 1)
reco_qcd1 = model.predict(qcdjets1, batch_size = 1)
reco_qcb1 = model.predict(qcbjets1, batch_size = 1)
reco_tt1 = model.predict(ttjets1, batch_size = 1)

reco_sig2 = model.predict(sigjets2, batch_size = 1)
reco_qcd2 = model.predict(qcdjets2, batch_size = 1)
reco_qcb2 = model.predict(qcbjets2, batch_size = 1)
reco_tt2 = model.predict(ttjets2, batch_size = 1)


#If we are plotting Y Jet AE Scores uncomment the next few lines and comment out the lines below that.
'''
score_sig1=  np.mean(np.square(reco_sig1 - sigjets1), axis=(1,2))[is_sig1_Y]
score_qcd1 = np.mean(np.square(reco_qcd1 - qcdjets1), axis=(1,2))[is_qcd1_Y]
score_qcb1 = np.mean(np.square(reco_qcb1 - qcbjets1), axis=(1,2))[is_qcb1_Y]
score_tt1 = np.mean(np.square(reco_tt1 - ttjets1), axis=(1,2))[is_tt1_Y]
fin_score_qcd1 = np.append(score_qcd1, score_qcb1)


score_sig2=  np.mean(np.square(reco_sig2 - sigjets2), axis=(1,2))[is_sig2_Y]
score_qcd2 = np.mean(np.square(reco_qcd2 - qcdjets2), axis=(1,2))[is_qcd2_Y]
score_qcb2 = np.mean(np.square(reco_qcb2 - qcbjets2), axis=(1,2))[is_qcb2_Y]
score_tt2 = np.mean(np.square(reco_tt2 - ttjets2), axis=(1,2))[is_tt2_Y]
fin_score_qcd2 = np.append(score_qcd2, score_qcb2)
'''
#This is if we are plottig the H Jet AE Scores
score_sig1=  np.mean(np.square(reco_sig1 - sigjets1), axis=(1,2))[np.logical_not(is_sig1_Y)]
score_qcd1 = np.mean(np.square(reco_qcd1 - qcdjets1), axis=(1,2))[np.logical_not(is_qcd1_Y)]
score_qcb1 = np.mean(np.square(reco_qcb1 - qcbjets1), axis=(1,2))[np.logical_not(is_qcb1_Y)]
score_tt1 = np.mean(np.square(reco_tt1 - ttjets1), axis=(1,2))[np.logical_not(is_tt1_Y)]
fin_score_qcd1 = np.append(score_qcd1, score_qcb1)


score_sig2=  np.mean(np.square(reco_sig2 - sigjets2), axis=(1,2))[np.logical_not(is_sig2_Y)]
score_qcd2 = np.mean(np.square(reco_qcd2 - qcdjets2), axis=(1,2))[np.logical_not(is_qcd2_Y)]
score_qcb2 = np.mean(np.square(reco_qcb2 - qcbjets2), axis=(1,2))[np.logical_not(is_qcb2_Y)]
score_tt2 = np.mean(np.square(reco_tt2 - ttjets2), axis=(1,2))[np.logical_not(is_tt2_Y)]
fin_score_qcd2 = np.append(score_qcd2, score_qcb2)


#Final Scores for Y vae scores
sigfin = np.append(score_sig1, score_sig2)
qcdfin = np.append(fin_score_qcd1, fin_score_qcd2)
ttfin = np.append(score_tt1, score_tt2)


'''

xhy = fin.split('/')[-1].split('.')[0]
qcd = bkg.split('/')[-1].split('.')[0]


#cutting on Hbb Signals, Events where at least one of the jets has a very high Hbb, retains alomst all the signal while removing a lot of background
sig_score_now = sig_score[hbb_signal_2>0.90]
bkg_score_now = bkg_score[hbb_bkg_2>0.90]
print(sig_score_now)

#Bkg jet masses cut on Hbb Alone
#Mjj = (h5py.File(bkg, "r")['jet_kinematics'][:,0])[hbb_bkg_2>0.90]  
#mj1 = (h5py.File(bkg, "r")['jet_kinematics'][:,5])[hbb_bkg_2>0.90]
#mj2 = (h5py.File(bkg, "r")['jet_kinematics'][:,-5])[hbb_bkg_2>0.90]


#Bkg jet masses cut on Hbb and VAE
Mjj = (h5py.File(bkg, "r")['jet_kinematics'][:,0])[(hbb_bkg_2>0.90)&(bkg_score>0.00005).reshape(-1)]  
mj1 = (h5py.File(bkg, "r")['jet_kinematics'][:,5])[(hbb_bkg_2>0.90)&(bkg_score>0.00005).reshape(-1)]
mj2 = (h5py.File(bkg, "r")['jet_kinematics'][:,-5])[(hbb_bkg_2>0.90)&(bkg_score>0.00005).reshape(-1)]


#Signal jet masses cut on Hbb alone 
#Mjj = (h5py.File(fin, "r")['jet_kinematics'][:,0])[hbb_signal_2>0.90] 
#mj1 = (h5py.File(fin, "r")['jet_kinematics'][:,5])[hbb_signal_2>0.90]
#mj2 = (h5py.File(fin, "r")['jet_kinematics'][:,-5])[hbb_signal_2>0.90]

#Signal jet masses cut and VAE
#Mjj = (h5py.File(fin, "r")['jet_kinematics'][:,0])[(hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1)]
#mj1 = (h5py.File(fin, "r")['jet_kinematics'][:,5])[(hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1)]
#mj2 = (h5py.File(fin, "r")['jet_kinematics'][:,-5])[(hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1)]
'''
#Plotting
colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow", "lightcoral", "gold","olive"]
hist_labels = ["Signal", "QCD", "TTBar"]
hist_colors = ["gray", "yellow", "r"]
hbb_labels = ["signal_j1", "signal_j2"]
hbb_labels_bkg = ["bkg_j1","bkg_j2"]
hbb_colors = ["b", "r"]
hist_scores = [sigfin, qcdfin, ttfin]














make_histogram(hist_scores, hist_labels, hist_colors, 'Labeler Score', "QCD + TTBar + XtoHY Signal" , 100,
            normalize = True, fname = "New_QCD_TTBar_XtoHY_H.png")

#make_histogram(hbb_scores_sig, hbb_labels, hbb_colors, 'Labeler Score',"Hbb_Sig"+ qcd +"_background_and_"+ xhy +"_XtoHY_Signal_for_" + jet_type  , 100,
#            normalize = True, fname = "Hbb_Sig"+qcd +"_Background_and_"+ xhy +"_XtoHY_Signal_for_" + jet_type +".png")

#make_histogram(hbb_scores_bkg, hbb_labels_bkg, hbb_colors, 'Labeler Score',"Hbb_Bkg"+ qcd +"_background_and_"+ xhy +"_XtoHY_Signal_for_" + jet_type  , 100,
#            normalize = True, fname = "Hbb_Bkg"+qcd +"_Background_and_"+ xhy +"_XtoHY_Signal_for_" + jet_type +".png")