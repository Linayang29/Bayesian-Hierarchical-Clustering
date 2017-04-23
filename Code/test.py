import BHC as bh
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# read in data
dat = np.genfromtxt('Data/Simulated/dat1.csv', delimiter=',', skip_header=1)
dat.shape
dat

# scatter plot
plt.scatter(dat[:,0], dat[:,1])
plt.show()

# dendrogram (for future use)
lk = bh.bhc(dat, family = niw, alpha = 1, r = 0.01)
dendrogram(lk)
