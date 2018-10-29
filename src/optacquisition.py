import sys
import time 
import random
import numpy as np
from src.pareto import *
from function import *
from src.gpstuff import *
from src.weights import *
from copy import deepcopy

def progress(count, total, status=''):
    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s --->  %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

def Generate_bounded(x1,x2,y1,y2,num):
    first_ = np.random.uniform(low = x1, high = x2, size=(num,))
    second_ = np.random.uniform(low = y1, high = y2, size=(num,))
    return np.vstack((first_,second_)).T

def AQFunc(X,dataset,points_,cnt):
    
    ################# TRAIN THE MODEL
    start_time = time.time()
    mod1,Kernels['ker0'] = trainModel(dataset.data,np.matrix(dataset.outputs[:,0]).T,'ker0',40)
    mod2,Kernels['ker1'] = trainModel(dataset.data,np.matrix(dataset.outputs[:,1]).T,'ker1',40)
    Kinv_0 = np.linalg.pinv(Kernels['ker0'].K(dataset.data,dataset.data))
    Kinv_1 = np.linalg.pinv(Kernels['ker1'].K(dataset.data,dataset.data))
    print("_____________________________")
    cprint("GP trained in %s seconds; OK!\n" % round(time.time() - start_time,5),"blue")

    #################  FIND THE PARETO
    start_time = time.time()
    yPareto = mPareto(dataset.outputs)
    xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
    if (len(xPareto)!=len(yPareto)):
        sys.exit("Size of X pareto is not same as Y pareto!")
    #print("_____________________________")
    #print("Found the Pareto in %s seconds; OK!\n" % round(time.time() - start_time,5))

    #################  READY TO LUNCH THE LOOP
    start_time = time.time()
    #copyPareto = deepcopy(yPareto)
    copyxPareto = deepcopy(xPareto)
    SlidingY = np.empty(shape = (0,OUTPUT_DIM))
    grid_,Jgrid_,Ngridholder_,pMap_ = samplePareto(yPareto)
    for valg in Jgrid_:
        if (valg[0,0]>valg[0,1]) and (valg[0,2]>valg[0,3]):
            yBatch = Generate_bounded(valg[0,0],valg[0,2],valg[0,1],valg[0,3],2)
            SlidingY = np.vstack((SlidingY,yBatch))

    if (SlidingY.shape[0] > 20):
        indices_ = [random.randint(0, SlidingY.shape[0]-1) for p in range(0, 20)]
        SlidingY = SlidingY[indices_]
        #plt.plot(SlidingY[:,0],SlidingY[:,1],"*g")
        #plt.show()
        
    Faster = {}
    for k in (range(SlidingY.shape[0])):
        AddY = np.vstack((yPareto,np.array([SlidingY[k,0],SlidingY[k,1]])))
        ParAddY = mPareto(AddY)
        Fast_1,Fast_2,Fast_3,Fast_4 = samplePareto(ParAddY)
        Faster[k] = Fast_3    
    cprint("Grids_ to Handle_: %s" % SlidingY.shape[0],"red")
    x_log = []
    imp_log = []
    indices_ = [random.randint(0, X.shape[0]-1) for p in range(0, 100)]
    optimizerX = X[indices_]
    #print("_____________________________")
    #print("Data prep. for main loop launched in %s seconds; OK!\n" % round(time.time() - start_time,5))
    
    stat_ = "Optimizing round " + str(cnt)
    for i in (range(len(optimizerX))):
        progress(i, len(optimizerX), status = stat_)
        x_log.append([optimizerX[i,:]])
        Total_HVI_diff = 0
        y1,Sigy1 = testModel(mod1,np.array([optimizerX[i,:]]))
        y2,Sigy2 = testModel(mod2,np.array([optimizerX[i,:]]))

        temp_x_pareto = np.vstack((copyxPareto,np.array([optimizerX[i,:]])))
        New_Weights_,New_Paretos_ = WeightPoints(temp_x_pareto,dataset,Kernels,Kinv_0,Kinv_1,points_)
        slide_size = len(SlidingY)
        for j in (range(slide_size)):
            temp_pareto = np.vstack((New_Paretos_[:-1],np.array([SlidingY[j,0],SlidingY[j,1]])))
            found_temp_pareto = mPareto(temp_pareto)
            usef_weights = New_Weights_[parY_X(temp_pareto,found_temp_pareto)]
            Probs_dim1 = norm(y1, Sigy1).pdf(SlidingY[j,0])
            Probs_dim2 = norm(y2, Sigy2).pdf(SlidingY[j,1])
            EHVI_New = Expected_HVI(found_temp_pareto,usef_weights,Faster[j])*Probs_dim1*Probs_dim2
            Total_HVI_diff += EHVI_New
        imp_log.append(Total_HVI_diff)
    indx = imp_log.index(max(imp_log))
    Best_x = x_log[indx][0]
    Best_y = function(np.array([Best_x]))[0,0],function(np.array([Best_x]))[0,1]
    xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
    return Best_x,Best_y,yPareto,xPareto