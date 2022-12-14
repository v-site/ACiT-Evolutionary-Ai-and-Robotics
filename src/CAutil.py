import csv
import yaml
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from yaml.loader import SafeLoader



# opens the config file and returns a dictionary of all parameters
def get_config():

    with open('config.yaml') as f:

        return yaml.load(f, Loader = SafeLoader)



# stores the dictionary globaly
config = get_config()



# generates random genes
def generate_initial_batch(batchSize):

    parentGenomes = []

    for _ in range(batchSize):

        genome = random.randint(0, 2 ** (2 ** config['windowLength']) - 1)

        parentGenomes.append(format(genome, ('0' + str(2 ** config['windowLength']) + 'b')))

    return parentGenomes



# constructs the list of all possible neighborhoods with the current windowLength 
def set_condition_list():

    conditionList = []

    for n in range(2 ** config['windowLength']):

        conditionList.append(format(n, ('0' + str(config['windowLength']) + 'b')))

    return conditionList



# converts the genome into a list of bits to be zipped to the condittionList
def initialize_rules(genome):

    if isinstance(genome, int):

        genome = format(genome, ('0' + str(2 ** config['windowLength']) + 'b'))

    return list(map(int, genome))



# construts the world map by interpolating observation within the available bits in worldWidth
def initialize_world(observation):

    # https://www.gymlibrary.dev/environments/classic_control/cart_pole/

    # Observation values: [CartPos, CartVel, PoleAngle, PoleAngleVel]

    minVals = [-2.4, -4, -0.2095, -4]
    maxVals = [ 2.4,  4,  0.2095,  4]

    worldMap = []

    for i in range(len(observation)):

        flipper = int(np.interp(observation[i], [minVals[i], maxVals[i]], [0, config['worldWidth']]))

        for n in range(config['worldWidth']):

            worldMap.append(1) if n == flipper else worldMap.append(0)

        # makes sure there is no spacing after the last observation
        if i == len(observation) - 1:
            continue

        for n in range(config['windowSpacing']):

            worldMap.append(0)

    return worldMap



# appends fixed boundries and applies the ruleset to the world map n times eqal to iterations
def apply_rules(worldMap, rules):

    edgeWidth = int((config['windowLength'] - 1) / 2)
    tempMap = [0] * edgeWidth + worldMap + [0] * edgeWidth

    for _ in range(config['iterations']):

        processedMap = []
        n = edgeWidth

        for _ in range(len(worldMap)):

            processedMap.append(rules[''.join(map(str, tempMap[n - edgeWidth : n + edgeWidth + 1]))])
            n += 1

        tempMap = [0] * edgeWidth + processedMap + [0] * edgeWidth

    return processedMap



# decides the action taken based on votingMethod 
def voting(processedMap):

    # finds the side with most instances of '1'
    if config['votingMethod'] == 'equal_split':

        l = int(len(processedMap) / 2)
        b = len(processedMap) % 2

        sumHead = sum(processedMap[0 : l + b])
        sumTail = sum(processedMap[l : ])

        if sumHead == sumTail:

            return int(random.randint(0, 1))

        if sumHead > sumTail:

            return 0
    
    # determines if there are more occurances of '1' or '0
    elif config['votingMethod'] == 'majority':

        if processedMap.count('1') <= processedMap.count('0'):

            return 0

    return 1



# concatinates all required functions needed to aquire an action
def get_action(observation, rules):

    worldMap = initialize_world(observation)

    processedMap = apply_rules(worldMap, rules)

    action = voting(processedMap)

    return action



# mutates (bitflip) candidates based on mutationRatio
def mutate(candidates):

    offspring = []

    Pl = len(list(candidates[0]))

    for i in range(len(candidates)):

        parentGenome = list(candidates[i])

        for n in range(Pl):

            if random.random() <= config['mutationRatio']:

                parentGenome[n] = '0' if parentGenome[n] == '1' else '1'

        offspring.append(''.join(parentGenome))

    return offspring



# crossbreeds candidates based on breedType
def breed(candidates):

    offspring = []

    Pn = len(candidates)
    Pl = len(list(candidates[0]))

    # parent genome split in two and added together
    if config['breedType'] == 'one-point':

        for i in range(Pn):

            p1 = list(candidates[i])
            p2 = list(candidates[-1 - i])

            c = p1[: int(Pl / 2)] + p2[int(Pl / 2) :]


            offspring.append(''.join(c))

    # parent genome split in three and added together
    if config['breedType'] == 'two-point': 

        for i in range(Pn):

            p1 = list(candidates[i])
            p2 = list(candidates[-1 - i])

            c = p1[: int(Pl * 0.25)] + p2[int(Pl * 0.25) : int(Pl * 0.75)] + p1[int(Pl * 0.75) :]

            offspring.append(''.join(c))
    
    # randomly insert genom-element from each of the candidates
    if config['breedType'] == 'uniform': 

        for i in range(Pn):

            p1 = list(candidates[random.randint(0, Pn - 1)])
            p2 = list(candidates[random.randint(0, Pn - 1)])
            c = []

            for n in range(Pl):

                c.append(p1[n]) if  random.random() < 0.5 else c.append(p2[n])

            offspring.append(''.join(c))

    return offspring



# evolve genomes from last generation 
def evolve(parents):

    elites =[]
    offspring = []
    elitesOffspring = []

    Pn = len(parents)

    # separates out the elites if required
    if config['elitRatio'] > 0:

        elites = list(map(itemgetter(0), parents))[int(Pn * (1 - config['elitRatio'] / 2)) :]

        elitesOffspring += breed(elites[int(len(elites) * 0.2) :])

        elitesOffspring += mutate(elites[: int(len(elites) * 0.2)])

    midleClass = parents[: int(Pn * (1 - config['elitRatio']))]

    Mn = len(midleClass)
    
    # turnament seletion for the privelage of breeding
    for _ in range(int((Pn * config['midleClassRatio']))):

        rivals  = []

        for _ in range(random.randint(2, Mn - 1)):

            rivals.append(midleClass[random.randint(0, Mn - 1)])

        rivals = sorted(rivals, key = itemgetter(1))[-2 :]

        l = list(map(itemgetter(0), rivals))

        child = list(breed(l))

        offspring.append(child[0])

    return offspring + elitesOffspring + elites



# plots all data gathered so far
def plot(maxReward, avgReward, allReward, gCounter):

    top25 = []
    polyX = []
    
    # creates x axis values and stores the average of te top 25 percent
    for i in range(gCounter):

        polyX.append(i)

        allReward[i].sort()

        top25.append(np.average(allReward[i][int(len(allReward[i]) * 0.75) :]))

    # initializes highly sbdivided line
    line = np.linspace(0, len(polyX) - 1, 100)

    # applyes the y walues from top, max, and avg to the highres line
    topModel = np.poly1d(np.polyfit(polyX, top25    , config['polyFactor']))
    maxModel = np.poly1d(np.polyfit(polyX, maxReward, config['polyFactor']))
    avgModel = np.poly1d(np.polyfit(polyX, avgReward, config['polyFactor']))

    # creates new plot and initializes axis
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1.6, 0.9])

    # plots line plots
    ax.plot(top25    , color='green' , label='^25')
    ax.plot(maxReward, color='red'   , label="Max")
    ax.plot(avgReward, color='orange', label="Avg")

    # plots smoothed line plots
    ax.plot(line, topModel(line), color='green' , linestyle='dashed')
    ax.plot(line, maxModel(line), color='red'   , linestyle='dashed')
    ax.plot(line, avgModel(line), color='orange', linestyle='dashed')

    # plots scatter plot per generation
    for i in range(gCounter):

        ax.scatter([i] * config['populationSize'], allReward[i], color = 'blue', s = 1)

    # adds labels and displays the plot
    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()

    # evaluates if a plot should be stored as image
    if (gCounter == config['generations']):

        fileName = get_filename()

        fig.savefig('plots/' + fileName + '.png', dpi = 300, bbox_inches = 'tight')



# creates a string based on parameters used as filename for plot ad log
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



# writes the logfile
def write_logs(fileName, logEntry):

    # creates or opens file
    f = open(fileName, 'a', newline = '')
    
    # create the csv writer
    writer = csv.writer(f, delimiter = ',')

    # write a row to the csv file
    writer.writerow(logEntry)

    # close the file
    f.close()
