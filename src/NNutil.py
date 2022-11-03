from fileinput import filename
import numpy as np
import datetime
import random
from operator import itemgetter
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt
import os
import csv




def get_config():

    with open('config.yml') as f:

        data = yaml.load(f, Loader = SafeLoader)

        return data



config = get_config()



def generate_initial_batch(n):

    return np.random.randn(n,4)



def get_action(observation, genome):

    minVals = [ -2.4, -4, -0.2095, -4]
    maxVals = [  2.4,  4,  0.2095,  4]

    interpValues = []

    for i in range(len(observation)):
        interpValues.append(np.interp(observation[i], [minVals[i], maxVals[i]], [-1,1]))

    if ( np.sum(np.array(interpValues) * genome) < 0 ):
        return 0
    else:
        return 1



def mutate(candidates):

    offspring = []

    for i in range(len(candidates)):

        c = candidates[i]

        for n in range(len(candidates[0])):

            if random.random() <= config['mutationRatio']:

                c[n] *= (1 + (random.random() / 10))

        offspring.append(list(c))

    return offspring



def breed(candidates):

    Pn = len(candidates) #number of elites
    Pl = len(list(candidates[0])) #length of genomes
    offspring = []

    if config['breedType'] == 'one-point': #parent genome split in two and added together

        for i in range(Pn):

            p1 = candidates[i]
            p2 = candidates[-1-i]

            c = p1[:int(Pl/2)] + p2[int(Pl/2):]

            offspring.append(c)

    if config['breedType'] == 'two-point': #parent genome split in three and added together

        for i in range(Pn):

            p1 = candidates[i]
            p2 = candidates[-1-i]

            c = p1[:int(Pl*0.25)] + p2[int(Pl*0.25):int(Pl*0.75)] + p1[int(Pl*0.75):]

            offspring.append(c)

    if config['breedType'] == 'uniform': #randomly insert genom-element from each of the parents

        for i in range(Pn):

            p1 = candidates[random.randint(0, Pn-1)]
            p2 = candidates[random.randint(0, Pn-1)]
            c = []

            for n in range(Pl):

                c.append(p1[n]) if  random.random() < 0.5 else c.append(p2[n])

            offspring.append(c)


    return offspring



def evolve(parents):

    #print(len(parents))
    
    Pn = len(parents) #number of elites
    elites =[]
    offspring = []
    elitesOffspring = []
    if config['elitRatio']>0:
        elites = list(map(itemgetter(0), parents))[int(Pn*(1-config['elitRatio'])):]

        #print(len(elites))

        elitesOffspring += breed(elites[int(len(elites)*0.2):]) #elits will breed

        #print(len(elitesOffspring))                           #gives 16 elites

        elitesOffspring += mutate(elites[:int(len(elites)*0.2)]) #some elites will spontaneous mutate       #gives 4

        #print(len(elitesOffspring))

    midleClass = parents[:int(Pn*(1-config['elitRatio']))]

        #print(len(midleClass))

    #run tournament for the rest
    Mn = len(midleClass)
    for _ in range(int((Pn*config['midleClassRatio'])-len(elitesOffspring))):
        rivals  = []
        for _ in range(random.randint(2, Mn-1)):
            rivals.append(midleClass[random.randint(0, Mn-1)])

        #select the two best of the rivals
        rivals = sorted(rivals, key=itemgetter(1))[-2:]

        #breed the two,
        l = list(map(itemgetter(0), rivals))

        child = list(breed(l))
        #print(f"{child} \n")
        offspring.append(child[0])



    #print(f"Elite offsprings:           {len(elitesOffspring)}")
    #print(f"Midle class offspring:      {len(offspring)}")
    #print(f"Elite passed to next gen:   {len(elitesOffspring)}")
    #returns 90% of the population, add 10% random later
    #print(f"Passed to CA: {len(offspring + elitesOffspring + elites)}")

    return offspring + elitesOffspring + elites



def plot(maxReward, avgReward, allReward, gCounter):

    top25 = []
    polyX = []

    for i in range(gCounter):

        polyX.append(i)

        allReward[i].sort()

        top25.append(np.average(allReward[i][int(len(allReward[i])*0.75):]))



    line = np.linspace(0, len(polyX)-1, 100)

    topModel = np.poly1d(np.polyfit(polyX, top25    , config['polyFactor']))
    maxModel = np.poly1d(np.polyfit(polyX, maxReward, config['polyFactor']))
    avgModel = np.poly1d(np.polyfit(polyX, avgReward, config['polyFactor']))



    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1.6, 0.9])

    ax.plot(top25    , color='green' , label='^25')
    ax.plot(maxReward, color='red'   , label="Max")
    ax.plot(avgReward, color='orange', label="Avg")

    ax.plot(line, topModel(line), color='green' , linestyle='dashed')
    ax.plot(line, maxModel(line), color='red'   , linestyle='dashed')
    ax.plot(line, avgModel(line), color='orange', linestyle='dashed')

    for i in range(gCounter):

        ax.scatter([i]*config['populationSize'], allReward[i], color='blue', s=1)



    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()



    if (gCounter == config['generations']):
        fileName = get_filename()
        fig.savefig('plots/' + fileName + '.png', dpi=300, bbox_inches='tight')

def get_filename():
    fileName = (str(config['seed']) + '_' +
                    str(config['worldWidth']) + '_' +
                    str(config['windowLength']) + '_' +
                    str(config['windowSpacing']) + '_' +
                    str(config['generations']) + '_' +
                    str(config['maxSteps']) + '_' +
                    str(config['populationSize']) + '_' +
                    str(config['maxAttempts']) + '_' +
                    str(config['iterations']) + '_' +
                    str(config['breedType']) + '_' +
                    str(config['votingMethod']) + '_' +
                    str(config['mutationRatio']) + '_' +
                    str(config['elitRatio']) + '_' +
                    str(config['midleClassRatio']) + '_' +
                    datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    return fileName

def write_logs(fileName, logEntry):
    f = open(fileName , 'a', newline='')
    # create the csv writer
    writer = csv.writer(f,delimiter=',')

    # write a row to the csv file
    writer.writerow(logEntry)

    # close the file
    f.close()