import sys
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from utils.TrainingUtils import *
import h5py
fin = "../../CASEUtils/H5_maker/QCD-HT1000to1500_nano-mc2016post-11_2016.h5"
toptag = h5py.File(fin, "r")['jet1_extraInfo'][:,-1]
plt.hist(toptag,bins =40)
plt.savefig("testingtop.png")
plt.show()