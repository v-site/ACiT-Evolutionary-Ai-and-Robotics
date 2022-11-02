from ctypes import util
import gym
import CAutil
import numpy as np
from operator import itemgetter
from timeit import default_timer as timer
import csv
import os

config = CAutil.get_config()

parentGenomes = CAutil.generate_initial_batch(config['populationSize'])
conditionList = CAutil.set_condition_list()

allReward = [[] for _ in range(config['generations'])]
maxReward = []
avgReward = []
simTime = []

env = gym.make("CartPole-v1")
observation, info = env.reset(seed= config['seed'])

startTime = timer()

#initiate log
fileName = os.path.join('logs',  CAutil.get_filename() +'.txt')
header = ['gen', 'maxR', 'avgR' , 'dT', 'aT(ms)', 'best_genome_over_450']
CAutil.write_logs(fileName = fileName, logEntry = header)

for gCounter in range(config['generations']):

    parents = []

    t = timer()

    for n in range(config['populationSize']):

        rules = dict(zip(conditionList, CAutil.initialize_rules(parentGenomes[n])))

        totReward = 0

        for _ in range(config['maxAttempts']):

            for i in range(config['maxSteps']):
                
                observation, reward, terminated, truncated, info = env.step(CAutil.get_action(observation, rules))

                totReward += reward - abs(observation[0])/2.4

                if terminated or truncated:

                    observation, info = env.reset(seed = (config['seed'] + gCounter + 1 + i))

                    break

        avgGenomeReward = round((totReward/config['maxAttempts']), 1)

        allReward[gCounter].append(avgGenomeReward)

        parents.append([parentGenomes[n], allReward[gCounter][n]])



    simTime.append(round((timer()-t)*1000/config['populationSize'], 1))

    parents = sorted(parents, key=itemgetter(1))

    parentGenomes = CAutil.evolve(parents)

    parentGenomes += CAutil.generate_initial_batch(config['populationSize']-len(parentGenomes)) #something fishy with this when it gets a negative number!

    maxReward.append(list(map(itemgetter(1), parents))[-1])

    avgReward.append(round(np.average(allReward[gCounter]), 1))


    

    if maxReward[gCounter] > 450:

        logEntry = [str(gCounter+1).zfill(3), maxReward[gCounter], avgReward[gCounter] , round(timer()-startTime, 1), simTime[gCounter], parents[-1][0]]
        CAutil.write_logs(fileName=fileName,logEntry=logEntry)
        print(f"Gen: {str(gCounter+1).zfill(3)} maxR: {maxReward[gCounter]} avgR {avgReward[gCounter]} dT: {round(timer()-startTime, 1)} aT: {simTime[gCounter]}ms Genome: {parents[-1][0]}")

    logEntry = [str(gCounter+1).zfill(3), maxReward[gCounter], avgReward[gCounter] , round(timer()-startTime, 1), simTime[gCounter], 'null']
    print(f"Gen: {str(gCounter+1).zfill(3)} maxR: {maxReward[gCounter]} avgR {avgReward[gCounter]} dT: {round(timer()-startTime, 1)} aT: {simTime[gCounter]}ms")
    CAutil.write_logs(fileName=fileName,logEntry=logEntry)


    if (gCounter == config['generations']-1 or (gCounter+1) % config['plotFrequency'] == 0):

        CAutil.plot(maxReward, avgReward, allReward, gCounter+1)



print(parents[-1][0])
