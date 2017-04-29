import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# read in data
dat = np.genfromtxt('Data/Simulated/dat1.csv', delimiter=',', skip_header=1)
dat.shape
dat

# scatter plot
plt.scatter(dat[:,0], dat[:,1])
for i in range(dat.shape[0]):
    plt.annotate(i, (dat[i,0], dat[i,1]))
plt.show()

# dendrogram (for future use)
Z = bhc(dat, family = "niw", alpha = 1, r = 0.1)
Z1 = np.array(Z)
Z1[:,2] = 1/Z1[:,2]
maxw = max(Z1[:,2])
Z1[Z1[:,2] < 0,2] = 2*maxw
for i in range(Z1.shape[0]):
    if Z1[i, 0] > (N-1):
        Z1[i, 2] += Z1[Z1[i, 0].astype("int")-N, 2]
    if Z1[i,1] > (N-1):
        Z1[i,2] += Z1[Z1[i,1].astype("int")-N, 2]

dendrogram(Z1)
plt.show()
