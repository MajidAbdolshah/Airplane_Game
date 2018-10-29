from scipy.stats import multivariate_normal
import numpy as np

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