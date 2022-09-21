# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:48:05 2022

@author: Pierre Boniface
"""

import numpy as np




def initialize_rules(windowLength):

    binaryWindow = 2 ** windowLength
    conditionList = []                                                         # Empty list to be filled with sequential pinary numbers
    responseList = [0]                                                         # WIP genome to be generated by evo
    n = 0

    for _ in range(binaryWindow):

        conditionList.append(format(n, ('0' + str(windowLength) + 'b')))       # Appends n in binary format to list of rules, example format; '03b', where 0 indicates leading zeroes, 3 is L (window length) in string form, and b indicates binary
        n = n + 1

    rules = dict(zip(conditionList, responseList))                             # Merges the list of rules with the list of outcomes

    print(rules)

    return rules



def initalize_window(worldWidth, angel):

    return format(int(np.interp(angel,[-0.2095,0.2095],[0,2**worldWidth])), ('0' + str(worldWidth) + 'b'))


