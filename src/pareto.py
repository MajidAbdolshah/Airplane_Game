import numpy as np
INPUT_DIM = 2
OUTPUT_DIM = 2
REF = [10,10]

def mPareto(y):
    pFlag = 0
    pSet = np.empty((0,y.shape[1]))
    sortedx = y[y[:,0].argsort()]
    uSortedx = np.empty((0,y.shape[1]))
    tMin = float('inf')

    if np.unique(sortedx[:,0]).shape[0] != sortedx[:,0].shape[0]:
        #print('Some points in a row...\nHandling that....\n')
        pFlag = 1
    if pFlag:
        U = np.unique(sortedx[:,0])
        for val in U:
            uSortedx = np.append(uSortedx,np.array([np.min(sortedx[sortedx[:,0] == val],axis=0)]),axis=0)
        for val in uSortedx:
            if (val[1] <= tMin):
                pSet = np.append(pSet,np.array([val]),axis=0)
                tMin = val[1]
        return pSet
    else:
        for val in sortedx:
            if (val[1] <= tMin):
                pSet = np.append(pSet,np.array([val]),axis=0)
                tMin = val[1]
        return pSet

def parY_X(y,ypar):

    size_y = y.shape[0]
    size_yp = ypar.shape[0]
    indx = []
    
    for i in range(size_yp):
        for j in range(size_y):
            if (y[j,0] == ypar[i,0] and y[j,1] == ypar[i,1]):
                indx.append(j)
    return indx

def findXpareto(xData,yData,yPareto):
    indx = parY_X(yData,yPareto)
    Xres = np.empty([INPUT_DIM,])
    for val in indx:
        Xres = np.vstack((Xres,xData[val,:]))
        

    return Xres[1:]

def samplePareto(par):
    infoDic = {}
    uPartsX = np.sort(par[:,0])
    uMapX = np.concatenate([[0],uPartsX,[REF[0]]])
    uPartsY = np.sort(par[:,1])
    uMapY = np.concatenate([[0],uPartsY,[REF[1]]])

    cells_ = np.empty((0,OUTPUT_DIM*2))
    Jparet_ = np.empty((0,OUTPUT_DIM*2))
    not_Jparet_ = np.empty((0,OUTPUT_DIM*2))
    for i in range(0,len(uMapX)-1):
        for j in range(0,len(uMapY)-1):

            pos_st = np.array([uMapX[i],uMapY[j]])
            pos_en = np.array([uMapX[i+1],uMapY[j+1]])

            pos_ = np.matrix(np.append(pos_st,pos_en,axis=0))
            mid_point = np.array([(pos_[0,0]+pos_[0,2])/2,
                                  (pos_[0,1]+pos_[0,3])/2])
            cells_ = np.vstack([cells_,pos_])
            infoDic[repr(pos_)] = ruPareto(mid_point,par)
            if ruPareto(mid_point,par):
                Jparet_ = np.vstack([Jparet_,pos_])
            else:
                not_Jparet_ = np.vstack([not_Jparet_,pos_])


    return cells_,Jparet_,not_Jparet_,infoDic


def ruPareto(x,par):
    for val in par:
        if(x[0]>val[0] and x[1]>val[1]):
            return False
    return True