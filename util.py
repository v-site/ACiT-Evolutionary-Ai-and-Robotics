# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:48:05 2022

@author: Pierre Boniface
"""
import matplotlib.pyplot as plt
import pycxsimulator
from pylab import *
import numpy as np
import random

def get_action(worldWidth, angel, windowLength, votingMethod, genome):

    worldMap = initialize_window(worldWidth, angel)
    rules = initialize_rules(windowLength,genome)
    processedMap = apply_rules(worldMap,rules,windowLength)
    action = voting(processedMap,votingMethod)

    return action



def initialize_rules(windowLength,genome):

    conditionList = []
    responseList = []
    binaryString = format(genome, ('0' + str(2**windowLength) + 'b'))
    n = 0

    for _ in range(2**windowLength):
        # WIP genome to be generated by evo
        responseList.append(int(binaryString[n]))
        n += 1

    n = 0

    for n in range(2**windowLength):
        # Appends n in binary format to list of conditions
        conditionList.append(format(n, ('0' + str(windowLength) + 'b')))
        n += 1

    # Merges the list of conditions with the list of responses
    return dict(zip(conditionList, responseList))



def initialize_window(worldWidth, angel):

    minAngle = -0.2095
    maxAngle =  0.2095
    worldMap = []
    n = 0

    # Maps the angle to a value between zero and the maximum binary worldwidth and converts to binary string
    binaryString = format(int(np.interp(angel, [minAngle, maxAngle], [0,2**worldWidth])), ('0' + str(worldWidth) + 'b'))

    for _ in range(worldWidth):

        # Appends the each character in the binary string as an int to the worldMap array
        worldMap.append(int(binaryString[n]))
        n += 1

    return worldMap



def apply_rules(worldMap,rules,windowLength):

    edgeWidth = int((windowLength-1)/2)
    tempMap = [0]*edgeWidth + worldMap + [0]*edgeWidth
    plotMap = worldMap

    for _ in range(len(worldMap)):
        processedMap = []
        n = edgeWidth
        for _ in range(len(worldMap)):
            processedMap.append(rules[''.join(map(str, tempMap[n-edgeWidth:n+edgeWidth+1]))])
            n += 1
        tempMap = [0]*edgeWidth + processedMap + [0]*edgeWidth
        plotMap = np.vstack((plotMap, processedMap))

    #print(plotMap)

    return processedMap



def voting(processedMap,votingMethod):

    if votingMethod == 'equal_split':
        l = int(len(processedMap)/2) #assumes the worldWith is even
        sumHead = sum(processedMap[0:l]) #TO-DO, check if this gives correct output,
        sumTail = sum(processedMap[l:])

        if sumHead == sumTail:
            return int(random.randint(0,1)) #randomly gives 0 or 1 of there is a tie
        if sumHead > sumTail:
            return 0
        else:
            return 1

    return 0 #WIP



#takes in a list parents as [genome,fitness]
def evolve(parents, cutSize, breedType, operator, crossoverRatio):

    parents = list(dict(list(parents.items())[int(len(parents)*(1-cutSize)):]).keys())
    print('winners \n', parents, '\n')

    offspring = [] #initiate offspring list
    crossoverParents = []
    mutationParents = []

    if operator == 'deterministically':
        crossoverParents = parents[:int((len(parents)*crossoverRatio))]
        mutationParents = parents[(int(len(parents)*crossoverRatio)):]

        print('crosParent \n', crossoverParents, '\n')
        print('mutParent \n', mutationParents, '\n')

    i = 0
    n = 0

    for i in range(len(mutationParents)):

        parentGenome = list(mutationParents[i])

        for n in range(len(parentGenome)):

            if  random.random() <= 0.01:

                if parentGenome[n] == '1':

                    parentGenome[n] = '0'
                else:
                    parentGenome[n] = '1'

            n += 1

        offspring.append(''.join(parentGenome))
        i += 1

    i = 0

    if breedType == 'one-point-crossover': #parent genome split in two and added together

        for i in range(int(len(crossoverParents)/2)):

            p1 = list(crossoverParents[i])
            p2 = list(crossoverParents[i+1])

            c = p1[:int(len(p1)/2)]+p2[int(len(p2)/2):]

            offspring.append(''.join(c))

            i += 2

    i = 0

    if breedType == 'two-point-crossover': #parent genome split in three and added together

        for i in range(int(len(crossoverParents)/2)):

            p1 = list(crossoverParents[i])
            p2 = list(crossoverParents[i+1])

            #add variable to control the split
            c = p1[:int(len(p1)*0.25)]+p2[int(len(p2)*0.25):int(len(p2)*0.75)]+p1[int(len(p1)*0.75):]

            offspring.append(''.join(c))

            i += 2

    i = 0

    if breedType == 'end-cross-over': #parent genome split in two and added together

        for i in range(int(len(crossoverParents)/2)):

            p1 = list(crossoverParents[i])
            p2 = list(crossoverParents[i+1])

            c = None

            offspring.append(''.join(c))

            i += 2

    return offspring