import gym
import NNutil
import numpy as np
from operator import itemgetter
from timeit import default_timer as timer

config = NNutil.get_config()

parentGenomes = NNutil.generate_initial_batch(config['populationSize'])

allReward = [[] for _ in range(config['generations'])]
maxReward = []
avgReward = []
simTime = []

env = gym.make("CartPole-v1")
observation, info = env.reset()

startTime = timer()

for gCounter in range(config['generations']):

    parents = []

    t = timer()

    for n in range(config['populationSize']):

        genome = parentGenomes[n]

        totReward = 0

        for _ in range(config['maxAttempts']):

            for _ in range(config['maxSteps']):

                observation, reward, terminated, truncated, info = env.step(NNutil.get_action(observation, genome))

                totReward += reward - abs(observation[0])/2.4

                if terminated or truncated:

                    observation, info = env.reset()

                    break

        avgGenomeReward = round((totReward/config['maxAttempts']), 1)

        allReward[gCounter].append(avgGenomeReward)

        parents.append([parentGenomes[n], allReward[gCounter][n]])



    simTime.append(round((timer()-t)*1000/config['populationSize'], 1))

    parents = sorted(parents, key=itemgetter(1))

    parentGenomes = NNutil.evolve(parents)

    parentGenomes += NNutil.generate_initial_batch(config['populationSize']-len(parentGenomes))

    maxReward.append(list(map(itemgetter(1), parents))[-1])

    avgReward.append(round(np.average(allReward[gCounter]), 1))



    print(f"Gen: {str(gCounter+1).zfill(3)} maxR: {maxReward[gCounter]} avgR {avgReward[gCounter]} dT: {round(timer()-startTime, 1)} aT: {simTime[gCounter]}ms")

    if maxReward[gCounter] > 450:

        print(parents[-1][0])

    if (gCounter == config['generations']-1 or (gCounter+1) % config['plotFrequency'] == 0):

        NNutil.plot(maxReward, avgReward, allReward, gCounter+1)



print(parents[-1][0])
