import os
import gym
import numpy as np
import src.CAutil as CAutil
from operator import itemgetter
from timeit import default_timer as timer

# load parameters
config = CAutil.get_config()

# initialize population
parentGenomes = CAutil.generate_initial_batch(config['populationSize'])
conditionList = CAutil.set_condition_list()

# initialize statistics lists
allReward = [[] for _ in range(config['generations'])]
maxReward = []
avgReward = []
simTime = []

#initiate log
fileName = os.path.join('logs', CAutil.get_filename() + '.txt')
header = ['gen', 'maxR', 'avgR', 'dT', 'aT(ms)', 'best_genome_over_450']
CAutil.write_logs(fileName = fileName, logEntry = header)

#initiate simulation
env = gym.make("CartPole-v1")
observation, info = env.reset(seed = config['seed'])

startTime = timer()

for gCounter in range(config['generations']):

    parents = []

    t = timer()

    for n in range(config['populationSize']):

        #creates rule-to-result dictionary
        rules = dict(zip(conditionList, CAutil.initialize_rules(parentGenomes[n])))

        totReward = 0

        for _ in range(config['maxAttempts']):

            for i in range(config['maxSteps']):
                
                # gets action based on environment
                observation, reward, terminated, truncated, info = env.step(CAutil.get_action(observation, rules))

                # penalty equal to absolut divergence rom center of environment
                totReward += reward - abs(observation[0]) / 2.4

                if terminated or truncated:

                    # Checks if seed is used
                    observation, info = env.reset(seed = (config['seed'] + gCounter + i)) if config['seed'] else env.reset()
                    break


        # calculates genome specific statistics
        avgGenomeReward = round((totReward/config['maxAttempts']), 1)

        allReward[gCounter].append(avgGenomeReward)

        parents.append([parentGenomes[n], allReward[gCounter][n]])


    # calculate generational statistics
    simTime.append(round((timer() - t) * 1000 / config['populationSize'], 1))

    parents = sorted(parents, key = itemgetter(1))

    parentGenomes = CAutil.evolve(parents)

    parentGenomes += CAutil.generate_initial_batch(config['populationSize'] - len(parentGenomes))

    maxReward.append(list(map(itemgetter(1), parents))[-1])

    avgReward.append(round(np.average(allReward[gCounter]), 1))


    # print and log generational statistics plus top genome if its reward is above 450
    if maxReward[gCounter] > 450:

        logEntry = [str(gCounter + 1 ).zfill(3), maxReward[gCounter], avgReward[gCounter], round(timer() - startTime, 1), simTime[gCounter], parents[-1][0]]

        print(f"Gen: {str(gCounter+1).zfill(3)} maxR: {maxReward[gCounter]} avgR {avgReward[gCounter]} dT: {round(timer()-startTime, 1)} aT: {simTime[gCounter]}ms Genome: {parents[-1][0]}")

    # print and log generational statistics
    else:

        logEntry = [str(gCounter + 1).zfill(3), maxReward[gCounter], avgReward[gCounter], round(timer() - startTime, 1), simTime[gCounter], 'null']

        print(f"Gen: {str(gCounter+1).zfill(3)} maxR: {maxReward[gCounter]} avgR {avgReward[gCounter]} dT: {round(timer()-startTime, 1)} aT: {simTime[gCounter]}ms")


    # save logfile
    CAutil.write_logs(fileName = fileName,logEntry = logEntry)


    # plots all stored results if generation cap reached or plot frequency fullfilled
    if (gCounter == config['generations'] - 1 or (gCounter + 1) % config['plotFrequency'] == 0):

        CAutil.plot(maxReward, avgReward, allReward, gCounter + 1)
