import os, subprocess
from itertools import compress
from numpy import genfromtxt
import numpy as np


def merge(locations):
    csvarrays = []
    for loc in locations:
        jet_mass = genfromtxt(loc, delimiter=",")
        csvarrays = np.append(csvarrays, jet_mass)
    return csvarrays



failpath = 'analysis_TEST/Fail/'
loosepath = 'analysis_TEST/Loose/'
passpath = 'analysis_TEST/Pass/'
passfiles = os.listdir(passpath)
loosefiles = os.listdir(loosepath)
failfiles = os.listdir(failpath)
mY_fail= []
mY_pass = []
mY_loose = []
mjj_fail= []
mjj_pass = []
mjj_loose = []
for f in passfiles:
    mY_pass.append("mY_BKG" in f)
    mjj_pass.append("Mjj_BKG" in f)
data_mY_pass = list(compress(passfiles, mY_pass))
data_mjj_pass = list(compress(passfiles, mjj_pass))


for f in loosefiles:
    mY_loose.append("mY_BKG" in f)
    mjj_loose.append("Mjj_BKG" in f)
data_mY_loose = list(compress(loosefiles, mY_loose))
data_mjj_loose = list(compress(loosefiles, mjj_loose))


for f in failfiles:
    mY_fail.append("mY_BKG" in f)
    mjj_fail.append("Mjj_BKG" in f)
data_mY_fail = list(compress(failfiles, mY_fail))
data_mjj_fail = list(compress(failfiles, mjj_fail))









#making arrays into paths by appending the path to their location
data_mY_pass = [passpath + f for f in data_mY_pass]
data_mjj_pass = [passpath + f for f in data_mjj_pass]

data_mY_fail = [failpath + f for f in data_mY_fail]
data_mjj_fail = [failpath + f for f in data_mjj_fail]

data_mY_loose = [loosepath + f for f in data_mY_loose]
data_mjj_loose = [loosepath + f for f in data_mjj_loose]


#actual merged data
data_mY_pass = merge(data_mY_pass)
data_mjj_pass = merge(data_mjj_pass)
data_mY_fail = merge(data_mY_fail)
data_mjj_fail = merge(data_mjj_fail)
data_mY_loose = merge(data_mY_loose)
data_mjj_loose = merge(data_mjj_loose)

print(len(data_mY_fail))
print(len(data_mY_loose))
print(len(data_mY_pass))

np.savetxt("analysis_TEST/JetHT_merged_CR_Mjj_Pass.csv", data_mjj_pass, delimiter=",")
np.savetxt("analysis_TEST/JetHT_merged_CR_mY_Pass.csv", data_mY_pass, delimiter=",")

np.savetxt("analysis_TEST/JetHT_merged_CR_mY_Loose.csv", data_mY_loose, delimiter=",")
np.savetxt("analysis_TEST/JetHT_merged_CR_Mjj_Loose.csv", data_mjj_loose, delimiter=",")

np.savetxt("analysis_TEST/JetHT_merged_CR_Mjj_Fail.csv", data_mjj_fail, delimiter=",")
np.savetxt("analysis_TEST/JetHT_merged_CR_mY_Fail.csv", data_mY_fail, delimiter=",")





