#a class for handeling the CA in the simulation
import pycxsimulator
from pylab import *
import numpy as np
import random



def voting(worldWith,method):

    if method == 'equal_split':
        l = int(len(worldWith)) #assumes the worldWith is odd
        sumHead = np.array(worldWith[0,l]).sum() #TO-DO, check if this gives correct output 
        sumTail = np.array(worldWith[l,]).sum()
        
        if sumHead == sumTail:
            return int(random.uniform(0,1)) #randomly gives 0 or 1 of there is a tie
        if sumHead > sumTail:
            return 0
        else:
            return 1

    #takes in the window
    result= 0
    return result



