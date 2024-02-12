import sys
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
from numpy import genfromtxt
import csv
import subprocess
from math import ceil
import os

eosls = 'eos root://cmseos.fnal.gov ls'
xrdfsls = "xrdfs root://cmseos.fnal.gov ls"
#TO DO: Calculate weights automatically
#TO DO: Figure out year from input file instead of from evtInfo
#TO DO: Include data flags


#these are the processes and their scalings
lumi_scalings = {'MX2400_MY100':0.01426653227501796,'MX2400_MY250':0.014205190129868695 ,
'MX2400_MY350':0.01421193132321061,'MX1600_MY150': 0.014019414578046748,
'MX2000_MY250':0.014386354225470658,'MX3000_MY190':0.014098784748060589,
'MX3000_MY300':0.01426472634652829,'MX3000_MY400':0.0140920946695184,
'MX2800_MY100':0.014162788862260646,'MX2800_MY190':0.01455293357535315,
'MX2600_MY300':0.014146798101153794,'TTToHadronic':9.355392919618758}#TTToHadronic is based on an incorrect xsec (687.1 pb) and total 2016 lumi
xsecs = {"signal":0.005, 'TTToHadronic':377.96}
int_lumi = {"2016":16800,"2016APV":19500,"2017":41500,"2018":59800}


def run_single_process(process,year):

    h5_dir  = f"/store/user/roguljic/H5_output/{year}/{process}/"
    if not "jetht" in process.lower():
        xrdcp_cmd = f"xrdcp root://cmseos.fnal.gov/{h5_dir}/merged.h5 merged.h5"
        fin =  "merged.h5"
        subprocess.call(xrdcp_cmd,shell=True)
        process_file(fin,process,year,"SR")
        process_file(fin,process,year,"CR")
        subprocess.call("rm merged.h5",shell=True)
    else:
        fNames   = subprocess.check_output(['{} {}'.format(xrdfsls,h5_dir)],shell=True,text=True).split('\n')
        n_files  = len(fNames)
        for i,fName in enumerate(fNames):
            print(f"{i}/{n_files}")
            if not "nano" in fName:
                continue
            short_name = fName.split("/")[-1]
            xrdcp_cmd = f"xrdcp root://cmseos.fnal.gov/{h5_dir}/{short_name} {short_name}"
            print(xrdcp_cmd)
            subprocess.call(xrdcp_cmd,shell=True)
            process_file(short_name,process,year,"SR")
            process_file(short_name,process,year,"CR")
            subprocess.call(f"rm {short_name}",shell=True) 

def process_file(fin,process,year,region):
    #model_name = "/uscms_data/d3/roguljic/XHanomalous/CMSSW_11_3_4/src/TagNTrain/plotting/jrand_autoencoder_m2500.h5" #Have to give absolute path here ._.
    model_name = os.getcwd()+"/jrand_autoencoder_m2500.h5" #Have to give absolute path here ._.
    #model_name = "/uscms_data/d3/roguljic/XHanomalous/CMSSW_11_3_4/src/TagNTrain/plotting/jrand_autoencoder_m2500.h5" #Have to give absolute path here ._.
    f = h5py.File(fin, "r")

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

    n_presel     = len(f['event_info'])
    presel_eff   = f['preselection_eff'][0]
    n_gen        = int(n_presel/presel_eff)
    print(f"Number of preselected events: {n_presel}")
    print(f"Preselection efficiency {presel_eff:.4f}")
    print(f"Number of generated events: {n_gen}")

    batch_size = 50000 #Gets rid of "exceedes 10% of memory" earning
    n_batches  = ceil(float(n_presel)/batch_size)
    for n_batch in range(n_batches):
        print(f"Batch {n_batch}/{n_batches}")
        start_evt = n_batch*batch_size
        stop_evt  = start_evt+batch_size
        if(stop_evt>n_presel):
            stop_evt=-1

        hbb_signal_1 = f['jet1_extraInfo'][start_evt:stop_evt,-2]
        hbb_signal_2 = f['jet2_extraInfo'][start_evt:stop_evt,-2]

        toptagging_1 = f['jet1_extraInfo'][start_evt:stop_evt,-1]
        toptagging_2 = f['jet2_extraInfo'][start_evt:stop_evt,-1]

        mj1_higgscut = np.array(f['jet_kinematics'][start_evt:stop_evt,5]).reshape(-1)
        mj2_higgscut = np.array(f['jet_kinematics'][start_evt:stop_evt,-5]).reshape(-1)

        fsignal1 = f["j1_images"][start_evt:stop_evt,:,:]
        fsignal2 = f["j2_images"][start_evt:stop_evt,:,:]

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
        if region=="SR":
            vaecuts1 = (sig_score1>0.00005) 
            vaecuts2 = (sig_score2>0.00005)
        else:
            vaecuts1 = np.logical_and((sig_score1>0.000025),(sig_score1<0.00004))
            vaecuts2 = np.logical_and((sig_score2>0.000025),(sig_score2<0.00004))
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

        tot_mjj = np.array(h5py.File(fin, "r")['jet_kinematics'][start_evt:stop_evt,0]).reshape(-1)
        tot_mj1 = np.array(h5py.File(fin, "r")['jet_kinematics'][start_evt:stop_evt,5]).reshape(-1)
        tot_mj2 = np.array(h5py.File(fin, "r")['jet_kinematics'][start_evt:stop_evt,-5]).reshape(-1)

        if "jetht" in process.lower():
            weights = []
            data_flag = True
        else:
            weights = f['sys_weights'][start_evt:stop_evt]*lumi_scalings[process]
            data_flag = False

        #Let's try to organize calculations a bit better
        tot_mh = (np.where(is_j2_moreHiggs == True, tot_mj2, tot_mj1))
        tot_my = (np.where(is_j2_moreHiggs == True, tot_mj1, tot_mj2)) 

        tagging_dict = {"Pass":pass_boolean,"Loose":loose_boolean,"Fail":fail_boolean}
        for tag_region in tagging_dict:
            file_name  = f"output/{process}_{year}_{region}_{tag_region}.csv"
            store_csv(tot_mjj,tot_mh,tot_my,weights,tagging_dict[tag_region],file_name,data_flag)

def store_csv(mjj,mh,my,weights,mask,file_name,data_flag):
    region_mjj = mjj[mask]
    region_mh  = mh[mask]
    region_my  = my[mask]
    if not data_flag:
        weights    = weights[mask]

    evts_in_region = len(region_mjj)
    assert evts_in_region==len(region_mh)
    assert evts_in_region==len(region_my)

    print(f"Writing {evts_in_region} events to {file_name}")

    with open(file_name, "a+", newline='') as f:
        writer = csv.writer(f)
        for i in range(len(region_mjj)):
            content = [region_mjj[i], region_mh[i], region_my[i]]
            if not data_flag:
                content.extend(weights[i])
            writer.writerow(content)

#python h5ToCsv_condor.py TTToHadronic 2017
process = sys.argv[1]
year    = sys.argv[2]
run_single_process(process,year)