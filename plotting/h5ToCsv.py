import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
from numpy import genfromtxt
import csv

#these are the processes and their scalings
#TO DO: Calculate this automatically!
lumi_scalings = {'MX2400_MY100':0.01426653227501796,'MX2400_MY250':0.014205190129868695 ,
'MX2400_MY350':0.01421193132321061,'MX1600_MY150': 0.014019414578046748,
'MX2000_MY250':0.014386354225470658,'MX3000_MY190':0.014098784748060589,
'MX3000_MY300':0.01426472634652829,'MX3000_MY400':0.0140920946695184,
'MX2800_MY100':0.014162788862260646,'MX2800_MY190':0.01455293357535315,
'MX2600_MY300':0.014146798101153794,'TTToHadronic':9.355392919618758}

def run_single_process(process,n_process=-1):
    fin =  "/uscms/home/roguljic/nobackup/Mufaro_backup/CASEUtils/jet_images/analysis_note_jets/merged_run2_{}.h5".format(process)
    plot_dir = "/uscms/home/roguljic/plotting/plots/"
    model_name = "/uscms/home/roguljic/nobackup/Mufaro_backup/CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 
    f = h5py.File(fin, "r")
    dest = "analysis_note_datasets_SR"
    #if(os.path.exists(dest+"/Pass/"+process+"_CRMjj_BKG.csv")):
     #   print("skipping"+ dest+"/Pass/"+process)
     #   continue

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

    hbb_signal_1 = f['jet1_extraInfo'][:n_process,-2]
    hbb_signal_2 = f['jet2_extraInfo'][:n_process,-2]

    toptagging_1 = f['jet1_extraInfo'][:n_process,-1]
    toptagging_2 = f['jet2_extraInfo'][:n_process,-1]

    mj1_higgscut = np.array(f['jet_kinematics'][:n_process,5]).reshape(-1)
    mj2_higgscut = np.array(f['jet_kinematics'][:n_process,-5]).reshape(-1)

    fsignal1 = f["j1_images"][:n_process,:,:]
    fsignal2 = f["j2_images"][:n_process,:,:]

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

    pass_boolean = np.logical_and(keepevent,np.logical_or(is_j1_higgs, is_j2_higgs))
    loose_boolean = np.logical_and(keepevent,np.logical_or(is_j1_loose, is_j2_loose))
    fail_boolean = np.logical_and(keepevent,np.logical_or(is_j1_fail, is_j2_fail))

    tot_mjj = np.array(h5py.File(fin, "r")['jet_kinematics'][:n_process,0]).reshape(-1)
    tot_mj1 = np.array(h5py.File(fin, "r")['jet_kinematics'][:n_process,5]).reshape(-1)
    tot_mj2 = np.array(h5py.File(fin, "r")['jet_kinematics'][:n_process,-5]).reshape(-1)

    weights = f['sys_weights'][:n_process]*lumi_scalings[process]

    #Let's try to organize calculations a bit better
    tot_mh = (np.where(is_j2_moreHiggs == True, tot_mj2, tot_mj1))
    tot_my = (np.where(is_j2_moreHiggs == True, tot_mj1, tot_mj2)) 

    region_mask_dict = {"Pass":pass_boolean,"Loose":loose_boolean,"Fail":fail_boolean}
    for region in region_mask_dict:
        file_name  = f"{dest}/{region}/{process}_CR.csv"
        store_csv(tot_mjj,tot_mh,tot_my,weights,region_mask_dict[region],file_name)

def store_csv(mjj,mh,my,weights,mask,file_name):
    #TO DO: Add column titles

    region_mjj = mjj[mask]
    region_mh  = mh[mask]
    region_my  = my[mask]
    weights    = weights[mask]

    evts_in_region = len(region_mjj)
    assert evts_in_region==len(region_mh)
    assert evts_in_region==len(region_my)
    assert evts_in_region==len(weights)

    print(f"Writing {evts_in_region} events to {file_name}")

    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f)
        for i in range(len(region_mjj)):
            content = [region_mjj[i], region_mh[i], region_my[i]]
            content.extend(weights[i])
            print(content)
            writer.writerow(content)


#processes = ['MX2400_MY100','MX2400_MY250','MX2400_MY350','MX1600_MY150','MX2000_MY250','MX3000_MY190','MX3000_MY300','MX3000_MY400','MX2800_MY100','MX2800_MY190','MX2600_MY300','TTToHadronic']
processes = ['MX2400_MY100']
for process in processes:
    n_process = 15 #Set to -1 to process all events
    run_single_process(process,n_process=n_process)