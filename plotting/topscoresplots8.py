import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
from numpy import genfromtxt
#these are the processes and their scalings
processes = {'MX2400_MY100':0.01426653227501796,'MX2400_MY250':0.014205190129868695 ,
'MX2400_MY350':0.01421193132321061,'MX1600_MY150': 0.014019414578046748,
'MX2000_MY250':0.014386354225470658,'MX3000_MY190':0.014098784748060589,
'MX3000_MY300':0.01426472634652829,'MX3000_MY400':0.0140920946695184,
'MX2800_MY100':0.014162788862260646,'MX2800_MY190':0.01455293357535315,
'MX2600_MY300':0.014146798101153794,'TTToHadronic':9.355392919618758}

for process in processes.keys():
    fin =  "../../CASEUtils/jet_images/analysis_note_jets/merged_run2_{}.h5".format(process)#signal we're testing
    plot_dir = "plotting/plots/"
    model_name = "../../CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 
    f = h5py.File(fin, "r")
    dest = "analysis_note_datasets_SR"

    sys_weights_map = {
        'nom_weight' : 0,
        'pdf_up' : 1,
        'pdf_down': 2,
        'prefire_up': 3,
        'prefire_down' : 4,
        'pileup_up' : 5 ,
        'pileup_down' : 6,
        'btag_up' : 7,
        'btag_down' : 8,
        'PS_ISR_up' : 9,
        'PS_ISR_down' : 10,
        'PS_FSR_up' : 11,
        'PS_FSR_down' : 12,
        'F_up' : 13,
        'F_down' : 14,
        'R_up' : 15,
        'R_down' : 16,
        'RF_up' : 17,
        'RF_down' : 18,
        'top_ptrw_up' : 19,
        'top_ptrw_down' : 20,
        }
    #if(os.path.exists("analysis_note_datasets/Pass_test/{}_nom_weight.csv".format(process))):
    #    print("skipping {}".format(process))
    #    continue
    hbb_signal_1 = f['jet1_extraInfo'][:,-2]
    hbb_signal_2 = f['jet2_extraInfo'][:,-2]

    toptagging_1 = f['jet1_extraInfo'][:,-1]
    toptagging_2 = f['jet2_extraInfo'][:,-1]
    mj1_higgscut = np.array(f['jet_kinematics'][:,5]).reshape(-1)
    mj2_higgscut = np.array(f['jet_kinematics'][:,-5]).reshape(-1)

    fsignal1 = f["j1_images"][()]
    fsignal2 = f["j2_images"][()]

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
    #### these vae cuts immediately below are for CR
    #vaecuts1 = np.logical_and((sig_score1>0.000025),(sig_score1<0.00004))  # these are the right vae cuts
    #vaecuts2 = np.logical_and((sig_score2>0.000025),(sig_score2<0.00004))   #right vae cuts
    #THESE SR CUTS FOR CR DO THE ONES ABOVE
    vaecuts1 = (sig_score1>0.00005)   # these are the right vae cuts
    vaecuts2 = (sig_score2>0.00005)  #right vae cuts

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
    higgs1cut = np.logical_and(mj1_higgscut>100, mj1_higgscut<150)
    higgs2cut = np.logical_and(mj2_higgscut>100, mj2_higgscut<150)
    keephiggs1 = np.logical_and(is_j1_moreHiggs,    higgs1cut)
    keephiggs2 = np.logical_and(is_j2_moreHiggs,    higgs2cut)
    keephiggs = np.logical_or(keephiggs1,keephiggs2)
    keepevent = np.logical_and(keepevent, keephiggs)
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

    #saving nominal weights out here first
    nom_weight_fail = f['sys_weights'][:,0].reshape(-1)[np.logical_and(fail_boolean, keepevent)]*processes[process]
    nom_weight_loose = f['sys_weights'][:,0].reshape(-1)[np.logical_and(loose_boolean, keepevent)]*processes[process]
    nom_weight_pass = f['sys_weights'][:,0].reshape(-1)[np.logical_and(pass_boolean, keepevent)]*processes[process]
    np.savetxt(dest+"/Pass_test/"+process +"_nom_weight.csv", nom_weight_pass, delimiter=",")
    np.savetxt(dest+"/Loose_test/"+process +"_nom_weight.csv",nom_weight_loose, delimiter=",")
    np.savetxt(dest+"/Fail_test/"+process +"_nom_weight.csv", nom_weight_fail, delimiter=",")

    #here we are saving the event weights up and down for every event and saving them to where the csv for masses are saved
    for key in sys_weights_map.keys():
        if process == 'nom_weight':
            continue


        if process != "TTToHadronic":
            if (key == 'top_ptrw_up') or (key == 'top_ptrw_down'):
                continue
        val = sys_weights_map[key]
        sys_weights_fail = f['sys_weights'][:,val].reshape(-1)[np.logical_and(fail_boolean, keepevent)]
        sys_weights_loose = f['sys_weights'][:,val].reshape(-1)[np.logical_and(loose_boolean, keepevent)]
        sys_weights_pass = f['sys_weights'][:,val].reshape(-1)[np.logical_and(pass_boolean, keepevent)]
        np.savetxt(dest+"/Pass_test/"+process +'_'+key+".csv", np.multiply(sys_weights_pass,nom_weight_pass), delimiter=",")
        np.savetxt(dest+"/Loose_test/"+process +'_'+key+".csv", np.multiply(sys_weights_loose,nom_weight_loose), delimiter=",")
        np.savetxt(dest+"/Fail_test/"+process +'_'+key+".csv", np.multiply(sys_weights_fail,nom_weight_fail), delimiter=",")
