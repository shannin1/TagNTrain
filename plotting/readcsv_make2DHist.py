# http://scikit-hep.org/root_numpy/reference/generated/root_numpy.array2tree.html

from root_numpy import array2tree
import numpy as np
import ROOT
from numpy import genfromtxt

processes = ['MX2400_MY100','MX2400_MY250','MX2400_MY350','MX1600_MY150','MX2000_MY250','MX3000_MY190','MX3000_MY300','MX3000_MY400','MX2800_MY100','MX2800_MY190','MX2600_MY300','TTToHadronic']
for process in processes:

    Mjj = genfromtxt("analysis_note_datasets/Pass/{}_CRMjj_BKG.csv".format(process), delimiter=",")
    mY = genfromtxt("analysis_note_datasets/Pass/{}_CRmY_BKG.csv".format(process), delimiter=",")

    LMjj = genfromtxt("analysis_note_datasets/Loose/{}_CRMjj_BKG.csv".format(process), delimiter=",")
    LmY = genfromtxt("analysis_note_datasets/Loose/{}_CRmY_BKG.csv".format(process), delimiter=",")

    FMjj = genfromtxt("analysis_note_datasets/Fail/{}CRMjj_BKG.csv".format(process), delimiter=",")
    FmY = genfromtxt("analysis_note_datasets/Fail/{}CRmY_BKG.csv".format(process), delimiter=",")



    # create a list to store (mX, mY) tuples
    l = []
    l2 = [] #fail
    l3 = []

    # loop over all the number of jets in Pass
    for i in range(len(Mjj)):
        # create the (mX, mY) tuple and append it to the list
        l.append((Mjj[i],mY[i]))
    # loop over all the number of jets in Pass
    for i in range(len(FMjj)):
        # create the (mX, mY) tuple and append it to the list
        l2.append((FMjj[i],FmY[i]))
    for i in range(len(LMjj)):
        # create the (mX, mY) tuple and append it to the list
        l3.append((LMjj[i],LmY[i]))  
    # create the 
    a =  np.array(l, dtype=[('Mjj',np.float64),('mY',np.float64)])
    b =  np.array(l2, dtype=[('FMjj',np.float64),('FmY',np.float64)]) #fail
    c =  np.array(l3, dtype=[('LMjj',np.float64),('LmY',np.float64)])

    # print, just to show the 10 row by 1 column shape (where the 1 column is a tuple of 2 values, mX and mY)
    print('Array has shape {}'.format(a.shape))
    print('Array has shape {}'.format(b.shape))
    print('Array has shape {}'.format(c.shape))

    # now create the ROOT TTree from the mX and mY data
    t = array2tree(a)
    t2 = array2tree(b)
    t3 = array2tree(c)

    # run TTree.Scan() just to show the structure of the tree
    #t.Scan()

    # create a dataframe from the tree
    df = ROOT.RDataFrame(t)
    df2 = ROOT.RDataFrame(t2)
    df3 = ROOT.RDataFrame(t3)
    # create a histogram from the dataframe
    h3 = df3.Histo2D(('Loose','LM_{jj} vs M_{Y}',50,1000,4000,50,0,500),'LMjj','LmY')
    h = df.Histo2D(('Pass','M_{jj} vs M_{Y}',50,1000,4000,50,0,500),'Mjj','mY')
    h2 = df2.Histo2D(('Fail','FM_{jj} vs M_{Y}',50,1000,4000,50,0,500),'FMjj','FmY')


    # save to new file (this is where you'd store the 2D histogram for the bkg estimate)
    f = ROOT.TFile('analysis_note_datasets/2dhistos_unscaled/{}_CR.root'.format(process),'recreate')
    f.cd()
    h.Write()
    h2.Write()
    h3.Write()
    f.Close()








