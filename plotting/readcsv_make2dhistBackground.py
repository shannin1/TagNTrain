from root_numpy import array2tree
import numpy as np
import ROOT
from numpy import genfromtxt


Mjj = genfromtxt("Sig/Pass/MX3000_MY300_merged_files_Mjj_BKG.csv", delimiter=",")
mj1 = genfromtxt("Sig/Pass/MX3000_MY300_merged_files_mj1_BKG.csv", delimiter=",")

LMjj = genfromtxt("Sig/Loose/MX3000_MY300_merged_files_Mjj_BKG.csv", delimiter=",")
Lmj1 = genfromtxt("Sig/Loose/MX3000_MY300_merged_files_mj1_BKG.csv", delimiter=",")

FMjj = genfromtxt("Sig/Fail/MX3000_MY300_merged_filesMjj_BKG.csv", delimiter=",")
Fmj1 = genfromtxt("Sig/Fail/MX3000_MY300_merged_filesmj1_BKG.csv", delimiter=",")



# create a list to store (mX, mY) tuples
l = []
l2 = [] #fail
l3 = []

# loop over all the number of jets in Pass
for i in range(len(Mjj)):
    # create the (mX, mY) tuple and append it to the list
    l.append((Mjj[i],mj1[i]))
# loop over all the number of jets in Pass
for i in range(len(FMjj)):
    # create the (mX, mY) tuple and append it to the list
    l2.append((FMjj[i],Fmj1[i]))
for i in range(len(LMjj)):
    # create the (mX, mY) tuple and append it to the list
    l3.append((LMjj[i],Lmj1[i]))  
# create the 
a =  np.array(l, dtype=[('Mjj',np.float64),('mj1',np.float64)])
b =  np.array(l2, dtype=[('FMjj',np.float64),('Fmj1',np.float64)]) #fail
c =  np.array(l3, dtype=[('LMjj',np.float64),('Lmj1',np.float64)])

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
h = df.Histo2D(('Pass','M_{jj} vs M_{j1}',50,1000,4000,50,0,500),'Mjj','mj1')
h2 = df2.Histo2D(('Fail','FM_{jj} vs M_{j1}',50,1000,4000,50,0,500),'FMjj','Fmj1')
h3 = df3.Histo2D(('Loose','LM_{jj} vs M_{j1}',50,1000,4000,50,0,500),'LMjj','Lmj1')

# save to new file (this is where you'd store the 2D histogram for the bkg estimate)
f = ROOT.TFile('2dhist_3wayMX3000_MY300.root','recreate')
f.cd()
h.Write()
h2.Write()
h3.Write()
f.Close()