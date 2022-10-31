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

        return yaml.load(f, Loader = SafeLoader)



config = get_config()



def generate_initial_batch(batchSize):

    parentGenomes = []

    for _ in range(batchSize):

        genome = random.randint(0, 2 ** (2**config['windowLength'])-1)

        parentGenomes.append(format(genome, ('0' + str(2**config['windowLength']) + 'b')))

    return parentGenomes



def set_condition_list():

    conditionList = []

    for n in range(2**config['windowLength']):

        conditionList.append(format(n, ('0' + str(config['windowLength']) + 'b')))

    return conditionList



def initialize_rules(genome):

    if isinstance(genome, int):

        genome = format(genome, ('0' + str(2**config['windowLength']) + 'b'))

    return list(map(int, genome))


def initialize_world(observation):

    #https://www.gymlibrary.dev/environments/classic_control/cart_pole/

    #Observation values: [CartPos, CartVel, PoleAngle, PoleAngleVel]

    minVals = [ -2.4, -4, -0.2095, -4]
    maxVals = [  2.4,  4,  0.2095,  4]

    worldMap = []

    for i in range(len(observation)):

        flipper = int(np.interp(observation[i], [minVals[i], maxVals[i]], [0,config['worldWidth']]))

        for n in range(config['worldWidth']):

            worldMap.append(1) if n == flipper else worldMap.append(0)

        if i == len(observation)-1:
            continue

        for n in range(config['windowSpacing']):

            worldMap.append(0)

    return worldMap



def apply_rules(worldMap, rules):

    edgeWidth = int((config['windowLength']-1)/2)
    tempMap = [0]*edgeWidth + worldMap + [0]*edgeWidth

    for _ in range(config['iterations']):

        processedMap = []
        n = edgeWidth

        for _ in range(len(worldMap)):

            processedMap.append(rules[''.join(map(str, tempMap[n-edgeWidth:n+edgeWidth+1]))])
            n += 1

        tempMap = [0]*edgeWidth + processedMap + [0]*edgeWidth

    return processedMap



def voting(processedMap):

    if config['votingMethod'] == 'equal_split':

        l = int(len(processedMap)/2)
        b = len(processedMap) % 2
        print(b)
        sumHead = sum(processedMap[0:l+b])
        sumTail = sum(processedMap[l:])

        print(processedMap)
        print(processedMap[0:l+b])
        print(processedMap[l:])

        if sumHead == sumTail:

            return int(random.randint(0,1))

        if sumHead > sumTail:

            return 0

    elif config['votingMethod'] == 'majority':

        if processedMap.count('1') <= processedMap.count('0'):

            return 0

    return 1



def get_action(observation, rules):

    worldMap = initialize_world(observation)

    processedMap = apply_rules(worldMap, rules)

    action = voting(processedMap)

    return action



def evolve(parents):

    parents = list(map(itemgetter(0), parents))[int(len(parents)*(1-config['cutSize'])):]

    Pn = len(parents) #number of parents
    Pl = len(list(parents[0])) #length of genomes

    offspring = []

    for i in range(Pn):

        parentGenome = list(parents[i])

        for n in range(Pl):

            if random.random() <= config['mutationRatio']:

                parentGenome[n] = '0' if parentGenome[n] == '1' else '1'

        offspring.append(''.join(parentGenome))

    if config['breedType'] == 'one-point': #parent genome split in two and added together

        for i in range(int(Pn)):

            p1 = list(parents[i])
            p2 = list(parents[-1-i])

            c = p1[:int(Pl/2)] + p2[int(Pl/2):]


            offspring.append(''.join(c))

    if config['breedType'] == 'two-point': #parent genome split in three and added together

        for i in range(int(Pn)):

            p1 = list(parents[i])
            p2 = list(parents[-1-i])

            c = p1[:int(Pl*0.25)] + p2[int(Pl*0.25):int(Pl*0.75)] + p1[int(Pl*0.75):]

            offspring.append(''.join(c))

    if config['breedType'] == 'uniform': #randomly insert genom-element from each of the parents

        for i in range(int(Pn)):

            p1 = list(parents[random.randint(0, Pn-1)])
            p2 = list(parents[random.randint(0, Pn-1)])
            c = []

            for n in range(Pl):

                c.append(p1[n]) if  random.random() < 0.5 else c.append(p2[n])

            offspring.append(''.join(c))

    offspring += parents # Live to fight another day

    return offspring



def plot(maxReward, avgReward, generationList):

    top25 = []
    polyX = []

    for i in range(len(generationList)):

        polyX.append(i)

        generationList[i].sort()

        top25.append(np.average(generationList[i][int(len(generationList[i])*0.75):]))



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
