import numpy as np
import time
import sys
import random
from termcolor import *
from function import *
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import math
from termcolor import colored

INPUT_DIM = 2
OUTPUT_DIM = 2
INITIAL = 16
DEEP = 100
BREAD = 100
MAXSAMPLE = 10**6
COUNTER = 16

class DataComplex:
    data = np.empty((0,INPUT_DIM))
    outputs = np.empty((0,OUTPUT_DIM))
    def __init__(self, iData,iOut):
        self.data = iData
        self.outputs = iOut
    def newData(self,newPoint):
        self.data = np.append(self.data,[newPoint],axis=0)
    def newOut(self,newPoint):
        self.outputs = np.append(self.outputs,[newPoint],axis=0)

def print_dots(string):
    sys.stdout.write(string)
    sys.stdout.flush()

def print_fancy(string,val):
    print_dots(string)
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    time.sleep(val)
    print_dots('.')
    time.sleep(val)
    print()

'''
def createPoints(deepness,breadth):
    points_ = {}
    for i in range(1,deepness+1):
        #temp_x = np.arange(0,i/3,i/(breadth*3))
        #temp_x = np.arange(0,i,i/(breadth))
        temp_x = np.linspace(0, i/10, num=breadth)
        temp_y = np.zeros(temp_x.shape)+i
        temp_merge = np.vstack((temp_x,temp_y))
        points_[i] = temp_merge

    for key in points_:
        if key!=1:
            points_[1] = np.hstack((points_[1],points_[key]))
    return points_[1].T
'''

def createPoints(deepness,breadth):
    points_ = {}
    for i in range(1,deepness+1):
        #temp_x = np.arange(0,i/3,i/(breadth*3))
        #temp_x = np.arange(0,i,i/(breadth))
        temp_x = np.linspace(0, i/5, num=breadth)
        temp_y = np.zeros(temp_x.shape)+i
        temp_merge = np.vstack((temp_x,temp_y))
        points_[i] = temp_merge

    for key in points_:
        if key!=1:
            points_[1] = np.hstack((points_[1],points_[key]))
    return points_[1].T

def info(*arg):
    for i in range(len(arg)):
        cprint("Shape of "+str(i)+" is "+str(arg[i].shape),"green")

def initvals_(bounds,INITIAL):
    str_ = "Initializing " + str(INITIAL) + " Data"
    print_fancy(str_,0.1)
    gData = np.zeros([INITIAL,INPUT_DIM])
    for i in range(0,INITIAL):
        for j in range(0,INPUT_DIM):
            gData[i,j] = random.uniform(bounds['min'][j], bounds['max'][j])
    gDataY = function(gData)
    return gData,gDataY

def printRes(yPareto,initial_pareto):
    fig = plt.figure()
    plt.plot(yPareto[:,0],yPareto[:,1],"or",markersize = 8,label='solutions we found')
    plt.plot(initial_pareto[:,0],initial_pareto[:,1],"*b",markersize = 15,label='initial observation')
    plt.legend('Model length', 'Data length',loc='upper center', shadow=True)
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.legend()
    plt.title('r2>(2)*r1')
    #plt.savefig('2_32_lONG_times.pdf', format='pdf', dpi=1000)
    plt.show()
    time.sleep(1)
    plt.close('all')