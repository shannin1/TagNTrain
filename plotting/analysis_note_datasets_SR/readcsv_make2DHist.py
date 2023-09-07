# http://scikit-hep.org/root_numpy/reference/generated/root_numpy.array2tree.html

from root_numpy import array2tree
import numpy as np
import ROOT
from numpy import genfromtxt

processes = ['MX2400_MY100','MX2400_MY250','MX2400_MY350','MX1600_MY150','MX2000_MY250','MX3000_MY190','MX3000_MY300','MX3000_MY400','MX2800_MY100','MX2800_MY190','MX2600_MY300','TTToHadronic']
for process in processes:
        f = ROOT.TFile('2dhistos_scaled/{}_SR.root'.format(process),'recreate')
        f.cd()
        sys_weights_map = ['nom_weight','pdf_up','pdf_down', 'prefire_up','prefire_down' , 'pileup_up'  ,'pileup_down' , 'btag_up' ,'btag_down' ,'PS_ISR_up' , 'PS_ISR_down' ,
                            'PS_FSR_up' ,'PS_FSR_down', 'F_up' ,'F_down' , 'R_up' ,'R_down' ,'RF_up' , 'RF_down','top_ptrw_up' ,'top_ptrw_down']
        for sys_weight in sys_weights_map:
                if process != 'TTToHadronic':
                    if ((sys_weight == 'top_ptrw_up') or (sys_weight=='top_ptrw_down')):
                         continue
                weightname = sys_weight
                if weightname == 'nom_weight':
                    weightname == 'nominal'

                passname = 'PmjjvsPmY_SR_pass_{}'.format(weightname)
                loosename = 'LmjjvsLmY_SR_loose_{}'.format(weightname)
                failname = 'FmjjvsFmY_SR_fail_{}'.format(weightname)        

                Mjj = genfromtxt("Pass/{}_CRMjj_BKG.csv".format(process), delimiter=",")
                mY = genfromtxt("Pass/{}_CRmY_BKG.csv".format(process), delimiter=",")
                passweights = genfromtxt("Pass_test/{}_{}.csv".format(process, sys_weight), delimiter=",") 

                LMjj = genfromtxt("Loose/{}_CRMjj_BKG.csv".format(process), delimiter=",")
                LmY = genfromtxt("Loose/{}_CRmY_BKG.csv".format(process), delimiter=",")
                looseweights = genfromtxt("Loose_test/{}_{}.csv".format(process, sys_weight), delimiter=",") 

                FMjj = genfromtxt("Fail/{}CRMjj_BKG.csv".format(process), delimiter=",")
                FmY = genfromtxt("Fail/{}CRmY_BKG.csv".format(process), delimiter=",")
                failweights = genfromtxt("Fail_test/{}_{}.csv".format(process, sys_weight), delimiter=",") 



                # create a list to store (mX, mY) tuples
                l = []
                l2 = [] #fail
                l3 = []

                # loop over all the number of jets in Pass
                for i in range(len(Mjj)):
                    # create the (mX, mY) tuple and append it to the list
                    l.append((Mjj[i],mY[i],passweights[i]))
                # loop over all the number of jets in Pass
                for i in range(len(FMjj)):
                    # create the (mX, mY) tuple and append it to the list
                    l2.append((FMjj[i],FmY[i],failweights[i]))
                for i in range(len(LMjj)):
                    # create the (mX, mY) tuple and append it to the list
                    l3.append((LMjj[i],LmY[i],looseweights[i]))  
                # create the 
                a =  np.array(l, dtype=[('Mjj',np.float64),('mY',np.float64),('passweights',np.float64)])
                b =  np.array(l2, dtype=[('FMjj',np.float64),('FmY',np.float64),('failweights',np.float64)]) #fail
                c =  np.array(l3, dtype=[('LMjj',np.float64),('LmY',np.float64),('looseweights',np.float64)])

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
                print(len(looseweights))
                print(len(LMjj))
                h2 = df2.Histo2D((failname,'FM_{jj} vs M_{Y}',50,1000,4000,50,0,500),'FMjj','FmY','failweights')
                h3 = df3.Histo2D((loosename,'LM_{jj} vs M_{Y}',50,1000,4000,50,0,500),'LMjj','LmY', 'looseweights')
                h = df.Histo2D((passname,'PM_{jj} vs M_{Y}',50,1000,4000,50,0,500),'Mjj','mY','passweights')
                


                # save to new file (this is where you'd store the 2D histogram for the bkg estimate)
               
                h.Write()
                h2.Write()
                h3.Write()
        f.Close()








