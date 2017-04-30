import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# multivariate data
dat = np.genfromtxt('Data/Simulated/dat1.csv', delimiter=',', skip_header=1)
dat.shape
dat

# scatter plot
plt.scatter(dat[:,0], dat[:,1])
for i in range(dat.shape[0]):
    plt.annotate(i, (dat[i,0], dat[i,1]))
plt.show()

# dendrogram
Z, color = bhc(dat, family = "multivariate", alpha = 1, r = 0.001)
dendrogram(Z, link_color_func=lambda k : color[k])
plt.show()


# bernoulli data
bdat = np.genfromtxt('~/Desktop/bindat.csv', delimiter=',')
bdat.shape
X=bdat
dat = bdat

dt2 = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

# dendrogram
dt2 = np.array([[1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 1, 0]])
pd.DataFrame(dt2)
dat = dt2
Zb, colorb = bhc(dt2, family = "bernoulli", alpha = 0.001)
dendrogram(Zb, link_color_func=lambda k : colorb[k])
plt.show()
