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



def mjj_from_4vecs(j1, j2):
    #assume j1 and j2 in (pt,eta,phi,m) format
    px1 = j1[:,0] * np.cos(j1[:,2])  
    py1 = j1[:,0] * np.sin(j1[:,2])  
    pz1 = j1[:,0] * np.sinh(j1[:,1]) 
    px2 = j2[:,0] * np.cos(j2[:,2])  
    py2 = j2[:,0] * np.sin(j2[:,2])  
    pz2 = j2[:,0] * np.sinh(j2[:,1]) 
    E1 = np.sqrt(px1**2 + py1**2 + pz1**2 + j1[:,3]**2)
    E2 = np.sqrt(px2**2 + py2**2 + pz2**2 + j2[:,3]**2)
    mjj = np.sqrt((E1 + E2)**2 - (px1 + px2)**2 - (py1 + py2)**2 - (pz1 + pz2)**2)
    return mjj

for process in processes.keys():
    fin =  "../../CASEUtils/jet_images/analysis_note_jets/merged_run2_{}.h5".format(process)#signal we're testing
    plot_dir = "plotting/plots/"
    model_name = "../../CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 
    f = h5py.File(fin, "r")
    dest = "analysis_note_datasets_JME_SR"


    JME_vars = ["JES_up", "JES_down", "JER_up","JER_down","JMS_up","JMS_down","JMR_up","JMR_down"]                   
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

    for JME in JME_vars:
        keepevent_temp = np.copy(keepevent)
        j1 = f["jet_kinematics"][:,2:6]
        j2 = f["jet_kinematics"][:,6:10]     
        if JME == "JES_up":
        #here we will calculate mjj and mj1 for every systematic
            pt_sys1 = f['jet1_JME_vars'][:,0].reshape(-1)
            pt_sys2 = f['jet2_JME_vars'][:,0].reshape(-1)
            mj1_corr = f['jet1_JME_vars'][:,1].reshape(-1)
            mj2_corr = f['jet2_JME_vars'][:,1].reshape(-1)
            keepevent_temp = np.logical_and((pt_sys1>300),np.logical_and(pt_sys2>300, keepevent_temp))   
            j1[:, 0] = pt_sys1
            j2[:, 0] = pt_sys2
            j1[:, 3] = mj1_corr
            j2[:, 3] = mj2_corr
        if JME == "JES_down":
        #here we will calculate mjj and mj1 for every systematic
            pt_sys1 = f['jet1_JME_vars'][:,2].reshape(-1)
            pt_sys2 = f['jet2_JME_vars'][:,2].reshape(-1)
            mj1_corr = f['jet1_JME_vars'][:,3].reshape(-1)
            mj2_corr = f['jet2_JME_vars'][:,3].reshape(-1)
            keepevent_temp = np.logical_and((pt_sys1>300),np.logical_and(pt_sys2>300, keepevent_temp))     
            j1[:, 0] = pt_sys1
            j2[:, 0] = pt_sys2
            j1[:, 3] = mj1_corr
            j2[:, 3] = mj2_corr

        if JME == "JER_up":
        #here we will calculate mjj and mj1 for every systematic
            pt_sys1 = f['jet1_JME_vars'][:,4].reshape(-1)
            pt_sys2 = f['jet2_JME_vars'][:,4].reshape(-1)
            mj1_corr = f['jet1_JME_vars'][:,5].reshape(-1)
            mj2_corr = f['jet2_JME_vars'][:,5].reshape(-1)
            keepevent_temp = np.logical_and((pt_sys1>300),np.logical_and(pt_sys2>300, keepevent_temp))      
            j1[:, 0] = pt_sys1
            j2[:, 0] = pt_sys2
            j1[:, 3] = mj1_corr
            j2[:, 3] = mj2_corr
        if JME == "JER_down":
        #here we will calculate mjj and mj1 for every systematic
            pt_sys1 = f['jet1_JME_vars'][:,6].reshape(-1)
            pt_sys2 = f['jet2_JME_vars'][:,6].reshape(-1)
            mj1_corr = f['jet1_JME_vars'][:,7].reshape(-1)
            mj2_corr = f['jet2_JME_vars'][:,7].reshape(-1)
            keepevent_temp = np.logical_and((pt_sys1>300),np.logical_and(pt_sys2>300, keepevent_temp))      
            j1[:, 0] = pt_sys1
            j2[:, 0] = pt_sys2
            j1[:, 3] = mj1_corr
            j2[:, 3] = mj2_corr
        if JME == "JMS_up":
        #here we will calculate mjj and mj1 for every systematic
            mj1_corr = f['jet1_JME_vars'][:,8].reshape(-1)
            mj2_corr = f['jet2_JME_vars'][:,8].reshape(-1)     
            j1[:, 3] = mj1_corr
            j2[:, 3] = mj2_corr
        if JME == "JMS_down":
        #here we will calculate mjj and mj1 for every systematic
            mj1_corr = f['jet1_JME_vars'][:,9].reshape(-1)
            mj2_corr = f['jet2_JME_vars'][:,9].reshape(-1)      
            j1[:, 3] = mj1_corr
            j2[:, 3] = mj2_corr
        if JME == "JMR_up":
        #here we will calculate mjj and mj1 for every systematic
            mj1_corr = f['jet1_JME_vars'][:,10].reshape(-1)
            mj2_corr = f['jet2_JME_vars'][:,10].reshape(-1)     
            j1[:, 3] = mj1_corr
            j2[:, 3] = mj2_corr
        if JME == "JMR_down":
        #here we will calculate mjj and mj1 for every systematic
            mj1_corr = f['jet1_JME_vars'][:,11].reshape(-1)
            mj2_corr = f['jet2_JME_vars'][:,11].reshape(-1)    
            j1[:, 3] = mj1_corr
            j2[:, 3] = mj2_corr

            
        tot_mjj = mjj_from_4vecs(j1, j2)
        tot_mj1 = j1[:,3] 
        tot_mj2 = j2[:,3]
        

        #saving nominal weights out here first


        pass_mH1_bool = np.logical_and(pass_boolean,is_j1_moreHiggs)
        pass_mH2_bool = np.logical_and(pass_boolean,is_j2_moreHiggs)
        pass_mH1 = (np.where(pass_mH2_bool == True, tot_mj2, tot_mj1))
        pass_mH = pass_mH1[np.logical_and(keepevent_temp, pass_boolean)] #where pass_mH1_bool == False put values in tot_mj2 on that position in tot_mj1
        pass_mY1_bool = np.logical_and(pass_boolean,is_j1_Y)
        pass_mY2_bool = np.logical_and(pass_boolean,is_j2_Y)
        pass_mY1 = (np.where(pass_mY2_bool == True, tot_mj2, tot_mj1)) 
        pass_mY = pass_mY1[np.logical_and(keepevent_temp, pass_boolean)]  
        pass_mjj = tot_mjj[np.logical_and(pass_boolean,keepevent_temp)]

        loose_mH1_bool = np.logical_and(loose_boolean,is_j1_moreHiggs)
        loose_mH2_bool = np.logical_and(loose_boolean,is_j2_moreHiggs)
        loose_mH1 = (np.where(loose_mH2_bool == True, tot_mj2, tot_mj1))
        loose_mH = loose_mH1[np.logical_and(keepevent_temp, loose_boolean)] #where loose_mH1_bool == False put values in tot_mj2 on that position in tot_mj1
        loose_mY1_bool = np.logical_and(loose_boolean,is_j1_Y)
        loose_mY2_bool = np.logical_and(loose_boolean,is_j2_Y)
        loose_mY1 = (np.where(loose_mY2_bool == True, tot_mj2, tot_mj1)) 
        loose_mY = loose_mY1[np.logical_and(keepevent_temp, loose_boolean)]  
        loose_mjj = tot_mjj[np.logical_and(loose_boolean,keepevent_temp)]

        fail_mH1_bool = np.logical_and(fail_boolean,is_j1_moreHiggs)
        fail_mH2_bool = np.logical_and(fail_boolean,is_j2_moreHiggs)
        fail_mH1 = (np.where(fail_mH2_bool == True, tot_mj2, tot_mj1))
        fail_mH = fail_mH1[np.logical_and(keepevent_temp, fail_boolean)] #where fail_mH1_bool == False put values in tot_mj2 on that position in tot_mj1
        fail_mY1_bool = np.logical_and(fail_boolean,is_j1_Y)
        fail_mY2_bool = np.logical_and(fail_boolean,is_j2_Y)
        fail_mY1 = (np.where(fail_mY2_bool == True, tot_mj2, tot_mj1)) 
        fail_mY = fail_mY1[np.logical_and(keepevent_temp, fail_boolean)]  
        fail_mjj = tot_mjj[np.logical_and(fail_boolean,keepevent_temp)]

        


        nom_weight_fail = f['sys_weights'][:,0].reshape(-1)[np.logical_and(fail_boolean, keepevent_temp)]*processes[process]
        nom_weight_loose = f['sys_weights'][:,0].reshape(-1)[np.logical_and(loose_boolean, keepevent_temp)]*processes[process]
        nom_weight_pass = f['sys_weights'][:,0].reshape(-1)[np.logical_and(pass_boolean, keepevent_temp)]*processes[process]
        np.savetxt("{}/Pass/{}_{}_nom_weight.csv".format(dest,process,JME), nom_weight_pass, delimiter=",")
        np.savetxt("{}/Loose/{}_{}_nom_weight.csv".format(dest,process,JME),nom_weight_loose, delimiter=",")
        np.savetxt("{}/Fail/{}_{}_nom_weight.csv".format(dest,process,JME), nom_weight_fail, delimiter=",")

        #Save Pass Masses arrays as csv
        np.savetxt("{}/Pass/{}_{}_Mjj_BKG.csv".format(dest,process,JME), pass_mjj, delimiter=",")
        np.savetxt("{}/Pass/{}_{}_mY_BKG.csv".format(dest,process,JME), pass_mY, delimiter=",")
        np.savetxt("{}/Pass/{}_{}_mH_BKG.csv".format(dest,process,JME), pass_mH, delimiter=",")


        #Save Loose Masses arrays as csv
        np.savetxt("{}/Loose/{}_{}_Mjj_BKG.csv".format(dest,process,JME), loose_mjj, delimiter=",")
        np.savetxt("{}/Loose/{}_{}_mY_BKG.csv".format(dest,process,JME), loose_mY, delimiter=",")
        np.savetxt("{}/Loose/{}_{}_mH_BKG.csv".format(dest,process,JME), loose_mH, delimiter=",")


        #Save Fail Masses arrays as csv
        np.savetxt("{}/Fail/{}_{}_Mjj_BKG.csv".format(dest,process,JME), fail_mjj, delimiter=",")
        np.savetxt("{}/Fail/{}_{}_mY_BKG.csv".format(dest,process,JME), fail_mY, delimiter=",")
        np.savetxt("{}/Fail/{}_{}_mH_BKG.csv".format(dest,process,JME), fail_mH, delimiter=",")

        #here we are saving the event weights up and down for every event and saving them to where the csv for masses are saved


