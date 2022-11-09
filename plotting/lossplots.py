import sys
sys.path.append('..')
from utils.TrainingUtils import *
import h5py


fin = "../../CASEUtils/jet_images/jet_testfiles/X3000_Y700.h5" #signal we're testing
bkg = "../../CASEUtils/jet_images/jet_testfiles/QCD-HT1500to2000_2.h5" #background we're testing


plot_dir = "plotting/plots/"
model_name = "../../CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 

fsignal = h5py.File(fin, "r")
fbkg =h5py.File(bkg, "r")

model = tf.keras.models.load_model(model_name)
reco_signal = model.predict(fin, batch_size = 1)
reco_bkg = model.predict(fbkg, batch_size = 1)
sig_score =  np.mean(np.square(reco_signal - fsignal), axis=(1,2))
bkg_score = np.mean(np.square(reco_bkg - fbkg), axis=(1,2))


colors = ["g", "b", "r", "gray", "purple", "pink", "orange", "m", "skyblue", "yellow", "lightcoral", "gold","olive"]
hist_labels = ["Background", "Signal"]
hist_colors = ["b", "r"]

bkg_events = Y < 0.1
sig_events = Y > 0.9


