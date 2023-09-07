# http://scikit-hep.org/root_numpy/reference/generated/root_numpy.array2tree.html

from root_numpy import array2tree
import numpy as np
import ROOT
from numpy import genfromtxt
#every bkg is from 2016APV
'''#QCD to start'''
Mjj = genfromtxt("Bkg/Pass/QCD2016APV_Mjj_BKG.csv", delimiter=",")
mj1 = genfromtxt("Bkg/Pass/QCD2016APV_mj1_BKG.csv", delimiter=",")

FMjj = genfromtxt("Bkg/Fail/QCD2016APVMjj_BKG.csv", delimiter=",")
Fmj1 = genfromtxt("Bkg/Fail/QCD2016APVmj1_BKG.csv", delimiter=",")

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
f = ROOT.TFile('2dhist_QCD2016APV.root','recreate')
f.cd()
h.Write()
h2.Write()
f.Close()



 #TTTo2L2Nu 2nd 
Mjj = genfromtxt("Bkg/Pass/arrTTTo2L2Nu2016APV_Mjj_BKG.csv", delimiter=",")
mj1 = genfromtxt("Bkg/Pass/arrTTTo2L2Nu2016APV_mj1_BKG.csv", delimiter=",")

FMjj = genfromtxt("Bkg/Fail/arrTTTo2L2Nu2016APVMjj_BKG.csv", delimiter=",")
Fmj1 = genfromtxt("Bkg/Fail/arrTTTo2L2Nu2016APVmj1_BKG.csv", delimiter=",")

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
h3 = df.Histo2D(('Pass','M_{jj} vs M_{j1}',50,1000,4000,50,0,500),'Mjj','mj1')
h4 = df2.Histo2D(('Fail','FM_{jj} vs M_{j1}',50,1000,4000,50,0,500),'FMjj','Fmj1')
#Saving
f = ROOT.TFile('2dhist_TTTo2L2Nu2016APV.root','recreate')
f.cd()
h3.Write()
h4.Write()
f.Close()

#TTToSemiLeptonic
Mjj = genfromtxt("Bkg/Pass/arrTTToSemiLeptonic2016APV_Mjj_BKG.csv", delimiter=",")
mj1 = genfromtxt("Bkg/Pass/arrTTToSemiLeptonic2016APV_mj1_BKG.csv", delimiter=",")

FMjj = genfromtxt("Bkg/Fail/arrTTToSemiLeptonic2016APVMjj_BKG.csv", delimiter=",")
Fmj1 = genfromtxt("Bkg/Fail/arrTTToSemiLeptonic2016APVmj1_BKG.csv", delimiter=",")

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
h5 = df.Histo2D(('Pass','M_{jj} vs M_{j1}',50,1000,4000,50,0,500),'Mjj','mj1')
h6 = df2.Histo2D(('Fail','FM_{jj} vs M_{j1}',50,1000,4000,50,0,500),'FMjj','Fmj1')
#Saving
f = ROOT.TFile('2dhist_TTToSemiLeptonic2016APV.root','recreate')
f.cd()
h5.Write()
h6.Write()
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

#WToQQ
Mjj = genfromtxt("Bkg/Pass/arrWJetsToQQ2016APV_Mjj_BKG.csv", delimiter=",")
mj1 = genfromtxt("Bkg/Pass/arrWJetsToQQ2016APV_mj1_BKG.csv", delimiter=",")

FMjj = genfromtxt("Bkg/Fail/arrWJetsToQQ2016APVMjj_BKG.csv", delimiter=",")
Fmj1 = genfromtxt("Bkg/Fail/arrWJetsToQQ2016APVmj1_BKG.csv", delimiter=",")

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
h9 = df.Histo2D(('Pass','M_{jj} vs M_{j1}',50,1000,4000,50,0,500),'Mjj','mj1')
h10 = df2.Histo2D(('Fail','FM_{jj} vs M_{j1}',50,1000,4000,50,0,500),'FMjj','Fmj1')

#Saving
f = ROOT.TFile('2dhist_WJetsToQQ2016APV.root','recreate')
f.cd()
h9.Write()
h10.Write()
f.Close()





