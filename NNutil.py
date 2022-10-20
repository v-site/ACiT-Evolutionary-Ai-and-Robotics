import numpy as np
import datetime
import random
from operator import itemgetter
from timeit import default_timer as timer
import yaml
from yaml.loader import SafeLoader
import matplotlib.pyplot as plt



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



def evolve(parents):

    parents = list(map(itemgetter(0), parents))[int(len(parents)*(1-config['cutSize'])):]

    Pn = len(parents) #number of parents
    Pl = len(list(parents[0])) #length of genomes

    offspring = []

    for i in range(Pn):

        c = parents[i]

        for n in range(Pl):

            if random.random() <= config['mutationRatio']:

                c[n] *= (1 + (random.random() / 10))

        offspring.append(c)

    if config['breedType'] == 'one-point': #parent genome split in two and added together

        for i in range(Pn):

            p1 = parents[i]
            p2 = parents[-1-i]

            c = p1[:int(Pl/2)] + p2[int(Pl/2):]

            offspring.append(c)

    if config['breedType'] == 'two-point': #parent genome split in three and added together

        for i in range(Pn):

            p1 = parents[i]
            p2 = parents[-1-i]

            c = p1[:int(Pl*0.25)] + p2[int(Pl*0.25):int(Pl*0.75)] + p1[int(Pl*0.75):]

            offspring.append(c)

    if config['breedType'] == 'uniform': #randomly insert genom-element from each of the parents

        for i in range(Pn):

            p1 = parents[random.randint(0, Pn-1)]
            p2 = parents[random.randint(0, Pn-1)]
            c = []

            for n in range(Pl):

                c.append(p1[n]) if  random.random() < 0.5 else c.append(p2[n])

            offspring.append(c)

    offspring += parents # Live to fight another day

    return offspring



def plot(maxReward, avgReward, generationList):

    top25 = []
    polyX = []

    for i in range(len(generationList)):

        polyX.append(i)

        generationList[i].sort()

        top25.append(np.average(generationList[i][int(len(generationList[i])*0.8):]))



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



    for i in range(len(generationList)):

        ax.scatter([i]*len(generationList[i]), generationList[i], color='blue', s=1)



    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()



    if (len(generationList) == config['generations']):
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
                    str(config['cutSize']) + '_' +
                    datetime.datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))

        fig.savefig('plots/' + fileName + '.png', dpi=300, bbox_inches='tight')
