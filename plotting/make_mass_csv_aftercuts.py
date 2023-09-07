import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
fin = "../../CASEUtils/jet_images/Bkg_JetImages/QCD2016APV.h5" #signal we're testing
plot_dir = "plotting/plots/"
model_name = "../../CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 
jet_type = 'j1_images' # or 'j1_images'

hbb_signal_1 = h5py.File(fin, "r")['jet1_extraInfo'][:,-1]
hbb_signal_2 = h5py.File(fin, "r")['jet2_extraInfo'][:,-1]
fsignal = h5py.File(fin, "r")[jet_type][()]

model = tf.keras.models.load_model(model_name)
fsignal = np.expand_dims(fsignal, axis = -1)

reco_signal = model.predict(fsignal, batch_size = 1)



sig_score =  np.mean(np.square(reco_signal - fsignal), axis=(1,2))



name = fin.split('/')[-1].split('.')[0]
#Signal Pass jet masses cut and VAE
Mjj = (h5py.File(fin, "r")['jet_kinematics'][:,0])[(hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1)]
mj1 = (h5py.File(fin, "r")['jet_kinematics'][:,5])[(hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1)]
mj2 = (h5py.File(fin, "r")['jet_kinematics'][:,-5])[(hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1)]

#Signal Fail jet masses cut and VAE
FMjj = (h5py.File(fin, "r")['jet_kinematics'][:,0])[np.logical_not((hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1))]
Fmj1 = (h5py.File(fin, "r")['jet_kinematics'][:,5])[np.logical_not((hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1))]
Fmj2 = (h5py.File(fin, "r")['jet_kinematics'][:,-5])[np.logical_not((hbb_signal_2>0.90) & (sig_score>0.00005).reshape(-1))]

#Save Pass Masses arrays as csv
np.savetxt("Bkg/Pass/"+name+"_Mjj_BKG.csv", Mjj, delimiter=",")
np.savetxt("Bkg/Pass/"+name+"_mj1_BKG.csv", mj1, delimiter=",")
np.savetxt("Bkg/Pass/"+name+"_mj2_BKG.csv", mj2, delimiter=",")


#Save Fail Masses arrays as csv
np.savetxt("Bkg/Fail/"+name+"_Mjj_BKG.csv", FMjj, delimiter=",")
np.savetxt("Bkg/Fail/"+name+"_mj1_BKG.csv", Fmj1, delimiter=",")
np.savetxt("Bkg/Fail/"+name+"_mj2_BKG.csv", Fmj2, delimiter=",")


