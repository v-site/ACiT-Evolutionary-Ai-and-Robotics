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

    return np.random.randn(batchSize, 4)


# interpolates observations and multiplys by the weights to determin action taken
def get_action(observation, genome):

    minVals = [-2.4, -4, -0.2095, -4]
    maxVals = [ 2.4,  4,  0.2095,  4]

    interpValues = []

    for i in range(len(observation)):

        interpValues.append(np.interp(observation[i], [minVals[i], maxVals[i]], [-1, 1]))

    if (np.sum(np.array(interpValues) * genome) < 0 ):

        return 0

    else:

        return 1



# mutates (random multiplication) candidates based on mutationRatio
def mutate(candidates):

    offspring = []

    for i in range(len(candidates)):

        c = candidates[i]

        for n in range(len(candidates[0])):

            if random.random() <= config['mutationRatio']:

                c[n] *= (1 + (random.random() / 10))

        offspring.append(list(c))

    return offspring



# crossbreeds candidates based on breedType
def breed(candidates):

    offspring = []

    Pn = len(candidates)
    Pl = len(list(candidates[0]))

    # parent genome split in two and added together
    if config['breedType'] == 'one-point':

        for i in range(Pn):

            p1 = candidates[i]
            p2 = candidates[-1 - i]

            c = p1[: int(Pl / 2)] + p2[int(Pl / 2) :]

            offspring.append(c)

    # parent genome split in three and added together
    if config['breedType'] == 'two-point': 

        for i in range(Pn):

            p1 = candidates[i]
            p2 = candidates[-1 - i]

            c = p1[: int(Pl * 0.25)] + p2[int(Pl * 0.25) : int(Pl * 0.75)] + p1[int(Pl * 0.75) :]

            offspring.append(c)

    # randomly insert genom-element from each of the parents
    if config['breedType'] == 'uniform': 

        for i in range(Pn):

            p1 = candidates[random.randint(0, Pn - 1)]
            p2 = candidates[random.randint(0, Pn - 1)]
            c = []

            for n in range(Pl):

                c.append(p1[n]) if  random.random() < 0.5 else c.append(p2[n])

            offspring.append(c)

    return offspring



# evolve genomes from last generation 
def evolve(parents):

    elites =[]
    offspring = []
    elitesOffspring = []

    Pn = len(parents)

    # separates out the elites if required
    if config['elitRatio'] > 0:

        elites = list(map(itemgetter(0), parents))[int(Pn * (1 - config['elitRatio'])) :]

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
