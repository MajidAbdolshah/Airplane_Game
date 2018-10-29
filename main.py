from src.datacomplex import *
from src.pareto import *
from function import *
from src.gpstuff import *
from src.optacquisition import *

if __name__ == '__main__':

    ################# PARAMETERS
    ctn = 0
    ancnt = 0
    bounds = dict()
    bounds  = {'min': [0,0],'max':[5,5]}
    depth_ = 0.5
    depth__ = 0.5
    observation = 4
    COUNTER_ = 36

    ################# GENERATE DATASET
    start_point = [0,0]
    log_start_point = []
    log_start_point.append(start_point)
    
    while ancnt < COUNTER_:
        print("Get ready for reset...")
        ctn = 0
        print(log_start_point)
        time.sleep(3)
        X = Generate_bounded(start_point[0],start_point[0]+depth_,start_point[1]
                            +0.07,start_point[1]+0.1,1000)
        Y = function(X)
        indices_ = [random.randint(0, X.shape[0]-1) for p in range(0, observation)]
        obs_X = X[indices_]
        obs_Y = Y[indices_]
        
        start_time = time.time()
        xData = obs_X
        yData = obs_Y
        
        #xData,yData = initvals_(bounds,INITIAL)
        dataset = DataComplex(xData,yData)
        yPareto = mPareto(dataset.outputs)
        xPareto = findXpareto(dataset.data,dataset.outputs,yPareto)
        initial_pareto = yPareto
        points_ = createPoints(DEEP,BREAD)
        #X,Y = initvals_(bounds,MAXSAMPLE)
        #X,Y = X.T,Y.T
        print("Data Initialized in %s seconds; OK!\n" % round(time.time() - start_time,5))
        
        while ctn < COUNTER:
    
            cprint("\n_____________ INFO ________________ "+str(ancnt),'green')
            info(dataset.data,dataset.outputs,yPareto,xPareto)
    
            Best_x,Best_y,yPareto,xPareto = AQFunc(X,dataset,points_,ctn)       
            if Best_x not in dataset.data:
                dataset.newData(Best_x)
                dataset.newOut(Best_y)
            else:
                #sys.exit("Couldn't go on, almost the same point! Why?! Check the parameters")
                pass
            
            np.savetxt('yPareto.txt', yPareto, fmt='%.100f')
            np.savetxt('initial_pareto.txt', initial_pareto, fmt='%.100f')
            #printRes(yPareto,initial_pareto)
            ctn += 1
        ind_ = random.randint(0, len(xPareto)-1)
        start_point  = list(xPareto[ind_])
        log_start_point.append(start_point)
        ancnt += 1
        