import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

mean0 = [ 0, 5]
cov0  = [[3, 0], [0, 3]]
mean1 = [ 5,0]
cov1  = [[3, 0], [0, 3]]

points = [[0, 0], [0.49566229433460823, 0.019241543168938648], [0.531419388151616, 0.11770260261399335], [0.6753979462694049, 0.17270179292570187], [0.9519761899854436, 0.2550923138757115], [1.440049064877885, 0.2731241019997851], [1.6427140094396377, 0.28690298680112936], [1.8938519138387566, 0.3675971293987615], [2.111150795513188, 0.3804666256356315]]

x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)
Z0 = np.random.random((len(x),len(y)))
Z1 = np.random.random((len(x),len(y)))
Z2 = np.random.random((len(x),len(y)))

def pdf0(arg1,arg2):
    return (stats.multivariate_normal.pdf((arg1,arg2), mean0, cov0))
def pdf1(arg1,arg2):
    return (stats.multivariate_normal.pdf((arg1,arg2), mean1, cov1))

for i in range (0, len(x)):
    for j in range(0,len(y)):
        Z0[i,j] = pdf0(x[i],y[j])
        Z1[i,j] = pdf1(x[i],y[j])

Z0=Z0.T
Z1=Z1.T        
Z2=Z2.T

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.contour(X,Y,Z0)
ax3.contour(X,Y,Z1)
points = np.array(points)
plt.plot(points[:,0],points[:,1],"*g",markersize=10)
plt.show()