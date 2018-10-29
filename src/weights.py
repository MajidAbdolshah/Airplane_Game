import numpy as np
from src.gpstuff import *
from src.datacomplex import *

def cell_point_dom(points,cell):
    res_ = []
    for i in range(0,len(points)):
        if ((cell[0,0] >= points[i,0]) and (cell[0,1] >= points[i,1])):
            res_.append(i)
    return res_

def Expected_HVI(points,weights,Not_Jgrid_):
    if (len(points) != len(weights)):
        sys.exit("WARNING WEIGHTs AND POINTS?")
    w_area = 0
    j_area = 0
    wPoints = {}
    Weightsdic = {}
    for val in Not_Jgrid_:
        wPoints[repr(val)] = cell_point_dom(points,val)
        if (len(wPoints[repr(val)])):
            exWeights = 1 - np.prod(np.take(1-weights,wPoints[repr(val)]))
        else:
            exWeights = 1
        area_ = (val[0,3] - val[0,1])*(val[0,2] - val[0,0])
        w_area += exWeights*area_
        j_area += area_
        Weightsdic[repr(val)] = exWeights
    return w_area



def findWeight(Mu,Sigma,deepness,breadth,points_):
    mvnMu = np.array([Mu[0][0,0], Mu[1][0,0]])
    mvnSigma = np.array([[Sigma[0][0,0],0],[0,Sigma[1][0,0]]])
    Rotationdict = createRot()
    sumProb = 0
    log = []
    for key in Rotationdict:
        mvnMuRnine = np.dot(Rotationdict[key],mvnMu)
        mvnSigmaRnine = np.dot(Rotationdict[key].T,np.dot(mvnSigma,Rotationdict[key]))
        sumProb += np.sum(multivariate_normal.pdf(points_,mvnMuRnine,mvnSigmaRnine))
        log.append(sumProb)
    return sumProb/(deepness*breadth)

def createRot():
    Rotationdict = {}
    Rotationdict[0] = np.array([[0,-1],[1,0]])
    Rotationdict[1] = np.array([[0,1],[-1,0]])
    return Rotationdict

def Derivatives(x,xPareto,dataset,Kernels,Kinv_0,Kinv_1):
    Mu = {}
    Sigma = {}
    Mu[0] = (dvt_mu(x,dataset.data,Kernels,dataset.outputs[:,0],'ker0',Kinv_0))
    Mu[1] = (dvt_mu(x,dataset.data,Kernels,dataset.outputs[:,1],'ker1',Kinv_1))
    Sigma[0] = (improved_dvt_var(x,dataset.data,Kernels,'ker0',Kinv_0))
    Sigma[1] = (improved_dvt_var(x,dataset.data,Kernels,'ker1',Kinv_1))
    return Mu,Sigma

def sampleXs(X,Y,bound):
    ind0_0 = np.where(Y[0,:] >= bound[0])[0]
    ind1_0 = np.where(Y[1,:] >= bound[1])[0]
    fIndx_0 = list(set(ind0_0) & set(ind1_0))

    ind0_1 = np.where(Y[0,:] <= bound[2])[0]
    ind1_1 = np.where(Y[1,:] <= bound[3])[0]
    fIndx_1 = list(set(ind0_1) & set(ind1_1))

    finalInd = list(set(fIndx_0) & set(fIndx_1))
    return X.T[finalInd],Y.T[finalInd]

def WeightPoints(xPareto,dataset,Kernels,Kinv_0,Kinv_1,points_):
    Wlog = []
    RPareto = []
    for val in xPareto:
        x = np.array([val])
        Mu,Sigma = Derivatives(x,xPareto,dataset,Kernels,Kinv_0,Kinv_1)
        tmp_ = findWeight(Mu,Sigma,DEEP,BREAD,points_)
        Wlog.append(tmp_)
        RPareto.append([function(x)[0,0],function(x)[0,1]])
    Weights = np.array(Wlog)
    RPareto = np.array(RPareto)
    return Weights,RPareto