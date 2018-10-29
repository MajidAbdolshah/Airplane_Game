from __future__ import division
import GPy
from scipy.stats import mvn
import GPyOpt
import numpy as np 
import os.path
from termcolor import colored
from numpy import linalg as LA
import numpy as np
import sys
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import math
from scipy.stats import norm
import GPy
import pprint 
from scipy.spatial import ConvexHull
from tqdm import *
import time
from sklearn.preprocessing import scale
from scipy.stats import multivariate_normal
import random
import sys
from george.kernels import ExpSquaredKernel

INPUT_DIM = 2
INITIAL = 5
OUTPUT_DIM = 2
MAX_ITER = 200
EPSILON = 10**-6
REF = [10,10]
LEN_SCALE = 0.1
EST_RANGE = 256

NUM = 1000

def f(x,fNum):
    def firstfun(x):
        return multivariate_normal.pdf(x,mean=np.array([5,0]),cov=np.array([[1,0],[0,1]]))
    def secondfun(x):
        return multivariate_normal.pdf(x,mean=np.array([0,5]),cov=np.array([[1,0],[0,1]]))
    options = {0 : firstfun,
               1 : secondfun,}
    return options[fNum](x)

def function(Xp):
    tmp_1 = f(Xp,0)
    tmp_2 = f(Xp,1)
    res_ = np.vstack((tmp_1,tmp_2))
    return res_.T

X = np.random.uniform(0.,5.,(NUM,2))
Y = function(X)# + np.random.randn(NUM,1)*0.005

plt.plot(Y[:,0],Y[:,1],'*b')
plt.show()
'''
kernel = GPy.kern.RBF(input_dim=1, variance=10, lengthscale=0.5)
m = GPy.models.GPRegression(X, Y,kernel)
#m.optimize_restarts(num_restarts = 10)
print(m.kern)
#m.optimize(max_f_eval = 1)

print(m.predict(np.array([[0.5]]),kern=kernel))
#plt.plot(X,Y,'.-r')
m.plot()
plt.show()
### Needs some plotting
'''
'''
from george import kernels
import numpy as np
import matplotlib.pyplot as pl
import george

x = 10 * np.sort(np.random.rand(15))
yerr = 0.01 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.xlim(0, 10)
pl.ylim(-1.45, 1.45)
pl.xlabel("x")
pl.ylabel("y");


kernel = np.var(y) * kernels.ExpSquaredKernel(0.5)
gp = george.GP(kernel)
gp.compute(x, yerr)

x_pred = np.linspace(0, 10, 500)
pred, pred_var = gp.predict(y, x_pred, return_var=True)

pl.fill_between(x_pred, pred - np.sqrt(pred_var), pred + np.sqrt(pred_var),
                color="k", alpha=0.2)
pl.plot(x_pred, pred, "k", lw=1.5, alpha=0.5)
pl.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
pl.plot(x_pred, np.sin(x_pred), "--g")
pl.xlim(0, 10)
pl.ylim(-1.45, 1.45)
pl.xlabel("x")
pl.ylabel("y");
pl.show()
'''