from root_numpy import array2tree
import numpy as np
import ROOT
from numpy import genfromtxt

'''#QCD to start'''
Mjj = genfromtxt("Bkg/Pass/QCD2016APV_Mjj_BKG.csv", delimiter=",")
mj1 = genfromtxt("Bkg/Pass/QCD2016APV_mj1_BKG.csv", delimiter=",")

FMjj = genfromtxt("Bkg/Fail/QCD2016APV_Mjj_BKG.csv", delimiter=",")
Fmj1 = genfromtxt("Bkg/Fail/QCD2016APV_mj1_BKG.csv", delimiter=",")


Mjj2 = genfromtxt("Bkg/Pass/arrTTToHadronic2016APV_Mjj_BKG.csv", delimiter=",")
mj12 = genfromtxt("Bkg/Pass/arrTTToHadronic2016APV_mj1_BKG.csv", delimiter=",")

FMjj2 = genfromtxt("Bkg/Fail/arrTTToHadronic2016APVMjj_BKG.csv", delimiter=",")
Fmj12 = genfromtxt("Bkg/Fail/arrTTToHadronic2016APVmj1_BKG.csv", delimiter=",")

print(len(Mjj))
print(len(Mjj))
print(len(FMjj))
print(len(FMjj2))

Mjj = np.append(Mjj,Mjj2)
mj1 = np.append(mj1,mj12)
FMjj = np.append(FMjj,FMjj2)
Fmj1 = np.append(Fmj1,Fmj12)

# create a list to store (mX, mY) tuples
l = []
l2 = [] #fail
# loop over all the number of jets in Pass
for i in range(len(Mjj)):
    # create the (mX, mY) tuple and append it to the list
    l.append((Mjj[i],mj1[i]))
# loop over all the number of jets in Pass
for i in range(len(FMjj)):
    # create the (mX, mY) tuple and append it to the list
    l2.append((FMjj[i],Fmj1[i]))
# create the 
a = np.array(l, dtype=[('Mjj',np.float64),('mj1',np.float64)])
b =  np.array(l2, dtype=[('FMjj',np.float64),('Fmj1',np.float64)]) #fail

# print, just to show the 10 row by 1 column shape (where the 1 column is a tuple of 2 values, mX and mY)
print('Array has shape {}'.format(a.shape))
print('Array has shape {}'.format(b.shape))

# now create the ROOT TTree from the mX and mY data
t = array2tree(a)
t2 = array2tree(b)

# run TTree.Scan() just to show the structure of the tree
#t.Scan()

# create a dataframe from the tree
df = ROOT.RDataFrame(t)
df2 = ROOT.RDataFrame(t2)

# create a histogram from the dataframe
h = df.Histo2D(('Pass','M_{jj} vs M_{j1}',50,1000,4000,50,0,500),'Mjj','mj1')
h2 = df2.Histo2D(('Fail','FM_{jj} vs M_{j1}',50,1000,4000,50,0,500),'FMjj','Fmj1')
#Saving
f = ROOT.TFile('2dhist_data2016APV.root','recreate')
f.cd()
h.Write()
h2.Write()
f.Close()


#TTToHadronic
Mjj = genfromtxt("Bkg/Pass/arrTTToHadronic2016APV_Mjj_BKG.csv", delimiter=",")
mj1 = genfromtxt("Bkg/Pass/arrTTToHadronic2016APV_mj1_BKG.csv", delimiter=",")

FMjj = genfromtxt("Bkg/Fail/arrTTToHadronic2016APVMjj_BKG.csv", delimiter=",")
Fmj1 = genfromtxt("Bkg/Fail/arrTTToHadronic2016APVmj1_BKG.csv", delimiter=",")

# create a list to store (mX, mY) tuples
l = []
l2 = [] #fail
# loop over all the number of jets in Pass
for i in range(len(Mjj)):
    # create the (mX, mY) tuple and append it to the list
    l.append((Mjj[i],mj1[i]))
# loop over all the number of jets in Pass
for i in range(len(FMjj)):
    # create the (mX, mY) tuple and append it to the list
    l2.append((FMjj[i],Fmj1[i]))
# create the 
a = np.array(l, dtype=[('Mjj',np.float64),('mj1',np.float64)])
b =  np.array(l2, dtype=[('FMjj',np.float64),('Fmj1',np.float64)]) #fail

# print, just to show the 10 row by 1 column shape (where the 1 column is a tuple of 2 values, mX and mY)
print('Array has shape {}'.format(a.shape))
print('Array has shape {}'.format(b.shape))

# now create the ROOT TTree from the mX and mY data
t = array2tree(a)
t2 = array2tree(b)

# run TTree.Scan() just to show the structure of the tree
#t.Scan()

# create a dataframe from the tree
df = ROOT.RDataFrame(t)
df2 = ROOT.RDataFrame(t2)

# create a histogram from the dataframe
h7 = df.Histo2D(('Pass','M_{jj} vs M_{j1}',50,1000,4000,50,0,500),'Mjj','mj1')
h8 = df2.Histo2D(('Fail','FM_{jj} vs M_{j1}',50,1000,4000,50,0,500),'FMjj','Fmj1')

#Saving
f = ROOT.TFile('2dhist_TTToHadronic2016APV.root','recreate')
f.cd()
h7.Write()
h8.Write()
f.Close()
