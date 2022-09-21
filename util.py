# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:48:05 2022

@author: Pierre Boniface
"""
import numpy as np


def initialize(L):

    rul = []
    res = [0] * 2**L
    n=0

    for _ in range(2**L):
        rul.append(format(n, ('0' + str(L) + 'b')))
        n = n+1

    dic = dict(zip(rul,res))

    print(dic)
initialize(4)



def initalize_window(worldWith, angel):
    return format(int(np.interp(angel,[-0.2095,0.2095],[0,2**worldWith])), ('0' + str(worldWith) + 'b'))

