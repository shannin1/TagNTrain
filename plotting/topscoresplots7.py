import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
from numpy import genfromtxt


fin = "../../CASEUtils/jet_images/analysis_note_jets/merged_run2_data_2016.h5" #signal we're testing
plot_dir = "plotting/plots/"
model_name = "../../CASEUtils/jet_images/AEmodels/AEs/jrand_autoencoder_m2500.h5" 
f = h5py.File(fin, "r")

model = tf.keras.models.load_model(model_name)




pass_mH1 = []
pass_mY1 = []
pass_mH2 = []
pass_mY2 = []
pass_mjj = []


loose_mH1 = []
loose_mY1 = []
loose_mH2 = []
loose_mY2 = []
loose_mjj = []

fail_mH1 = []
fail_mY1 = []
fail_mH2 = []
fail_mY2 = []
fail_mjj = []






batch_size = 20000
num_batches = int(np.ceil(f['j1_images'].shape[0] / batch_size )) 
sig_score1 = []
sig_score2 = []
for i in range(num_batches):
    hbb_signal_1 = f['jet1_extraInfo'][:,-2][i*batch_size: (i+1) * batch_size]
    hbb_signal_2 = f['jet2_extraInfo'][:,-2][i*batch_size: (i+1) * batch_size]

    toptagging_1 = f['jet1_extraInfo'][:,-1][i*batch_size: (i+1) * batch_size]
    toptagging_2 = f['jet2_extraInfo'][:,-1][i*batch_size: (i+1) * batch_size]
    mj1_higgscut = np.array(f['jet_kinematics'][:,5]).reshape(-1)[i*batch_size: (i+1) * batch_size]
    mj2_higgscut = np.array(f['jet_kinematics'][:,-5]).reshape(-1)[i*batch_size: (i+1) * batch_size]
    imgs = f['j1_images'][i*batch_size: (i+1) * batch_size]
    imgs2 = f['j2_images'][i*batch_size: (i+1) * batch_size]
    imgs = np.expand_dims(imgs, axis = -1)
    imgs2 = np.expand_dims(imgs2, axis = -1)
    reco_signal1 = model.predict(imgs, batch_size = 1)
    reco_signal2 = model.predict(imgs2, batch_size = 1)
    sig_score1 =  np.mean(np.square(reco_signal1 - imgs), axis=(1,2)).reshape(-1)
    sig_score2 =  np.mean(np.square(reco_signal2 - imgs2), axis=(1,2)).reshape(-1)
    #sig_score1 = np.append(sig_score1,temp_sig_score1)
    #sig_score2 = np.append(sig_score2,temp_sig_score2)
    print("currently predicting batch {}".format(i))
    #if i==1:
    #   break
    #print("here")

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
    
    vaecuts1 = np.logical_and((sig_score1>0.000025),(sig_score1<0.00004))  # these are the right vae cuts
    vaecuts2 = np.logical_and((sig_score2>0.000025),(sig_score2<0.00004))   #right vae cuts
    
    #THESE SR CUTS FOR CR DO THE ONES ABOVE
    #vaecuts1 = (sig_score1>0.00005)   # these are the right vae cuts
    #vaecuts2 = (sig_score2>0.00005)  #right vae cuts
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
    #now the HIGGS mass cut, we keep all events with 100<MH<150
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

    tot_mjj = np.array(f['jet_kinematics'][:,0]).reshape(-1)[i*batch_size: (i+1) * batch_size]
    tot_mj1 = np.array(f['jet_kinematics'][:,5]).reshape(-1)[i*batch_size: (i+1) * batch_size]
    tot_mj2 = np.array(f['jet_kinematics'][:,-5]).reshape(-1)[i*batch_size: (i+1) * batch_size]






    name = fin.split('/')[-1].split('.')[0].split('run2_')[-1]




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
    pass_mH1 = np.append(pass_mH1, tot_mj1[np.logical_and(np.logical_and(pass_boolean,is_j1_moreHiggs),keepevent)])
    pass_mY1 = np.append(pass_mY1, tot_mj1[np.logical_and(np.logical_and(pass_boolean,is_j1_Y),keepevent)])
    pass_mH2 = np.append(pass_mH2, tot_mj2[np.logical_and(np.logical_and(pass_boolean,is_j2_moreHiggs),keepevent)])
    pass_mY2 = np.append(pass_mY2,tot_mj2[np.logical_and(np.logical_and(pass_boolean,is_j2_Y),keepevent)])
    pass_mjj = np.append(pass_mjj,tot_mjj[np.logical_and(pass_boolean,keepevent)])


    loose_mH1 = np.append(loose_mH1, tot_mj1[np.logical_and(np.logical_and(loose_boolean,is_j1_moreHiggs),keepevent)])
    loose_mY1 = np.append(loose_mY1, tot_mj1[np.logical_and(np.logical_and(loose_boolean,is_j1_Y),keepevent)])
    loose_mH2 = np.append(loose_mH2, tot_mj2[np.logical_and(np.logical_and(loose_boolean,is_j2_moreHiggs),keepevent)])
    loose_mY2 = np.append(loose_mY2,tot_mj2[np.logical_and(np.logical_and(loose_boolean,is_j2_Y),keepevent)])
    loose_mjj = np.append(loose_mjj, tot_mjj[np.logical_and(loose_boolean,keepevent)])

    fail_mH1 = np.append(fail_mH1, tot_mj1[np.logical_and(np.logical_and(fail_boolean,is_j1_moreHiggs),keepevent)])
    fail_mY1 = np.append(fail_mY1, tot_mj1[np.logical_and(np.logical_and(fail_boolean,is_j1_Y),keepevent)])
    fail_mH2 = np.append(fail_mH2,tot_mj2[np.logical_and(np.logical_and(fail_boolean,is_j2_moreHiggs),keepevent)])
    fail_mY2 = np.append(fail_mY2,tot_mj2[np.logical_and(np.logical_and(fail_boolean,is_j2_Y),keepevent)])
    fail_mjj = np.append(fail_mjj, tot_mjj[np.logical_and(fail_boolean,keepevent)])

#Save Pass Masses arrays as csv
np.savetxt("analysis_note_datasets/Pass/"+name+"_VAECR_Mjj_BKG.csv", pass_mjj, delimiter=",")
np.savetxt("analysis_note_datasets/Pass/"+name+"_VAECR_mY_BKG.csv", np.append(pass_mY1, pass_mY2), delimiter=",")
np.savetxt("analysis_note_datasets/Pass/"+name+"_VAECR_mH_BKG.csv", np.append(pass_mH1, pass_mH2), delimiter=",")


#Save Loose Masses arrays as csv
np.savetxt("analysis_note_datasets/Loose/"+name+"_VAECR_Mjj_BKG.csv", loose_mjj, delimiter=",")
np.savetxt("analysis_note_datasets/Loose/"+name+"_VAECR_mY_BKG.csv", np.append(loose_mY1, loose_mY2), delimiter=",")
np.savetxt("analysis_note_datasets/Loose/"+name+"_VAECR_mH_BKG.csv", np.append(loose_mH1, loose_mH2), delimiter=",")


#Save Fail Masses arrays as csv
np.savetxt("analysis_note_datasets/Fail/"+name+"_VAECR_Mjj_BKG.csv", fail_mjj, delimiter=",")
np.savetxt("analysis_note_datasets/Fail/"+name+"_VAECR_mY_BKG.csv", np.append(fail_mY1, fail_mY2), delimiter=",")
np.savetxt("analysis_note_datasets/Fail/"+name+"_VAECR_mH_BKG.csv", np.append(fail_mH1, fail_mH2), delimiter=",")
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