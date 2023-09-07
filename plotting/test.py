
import numpy as np
import matplotlib.pyplot as plt
 
# Data in numpy array
exp_data = np.array([12, 15, 13, 20, 19, 20, 11, 19, 11, 12, 19, 13, 
                    12, 10, 6, 19, 3, 1, 1, 0, 4, 4, 6, 5, 3, 7, 
                    12, 7, 9, 8, 12, 11, 11, 18, 19, 18, 19, 3, 6, 
                    5, 6, 9, 11, 10, 14, 14, 16, 17, 17, 19, 0, 2, 
                    0, 3, 1, 4, 6, 6, 8, 7, 7, 6, 7, 11, 11, 10, 
                    11, 10, 13, 13, 15, 18, 20, 19, 1, 10, 8, 16, 
                    19, 19, 17, 16, 11, 1, 10, 13, 15, 3, 8, 6, 9, 
                    10, 15, 19, 2, 4, 5, 6, 9, 11, 10, 9, 10, 9, 
                    15, 16, 18, 13])
exp_data2 = np.array([2,2,2,2,2,2,2,2,23,3,4,4,5,5,3,5,5,5,5,5,6,6,6,7,7,7,8,8,9,8,8,8,8,8,8,8,8,8,8,8,8,18])
hist_scores = [exp_data,exp_data2]
colors = ['purple', 'b']

labels = ["Background", "Signal"]

ns, bins, patches = plt.hist(hist_scores, bins=15, range=None, color=colors, alpha=0.6,label=labels, density = True, histtype='stepfilled')
plt.yticks([])
plt.tick_params(axis='y', labelsize=16)
plt.legend(loc='upper right', fontsize = 16) 
plt.savefig("test.png")
