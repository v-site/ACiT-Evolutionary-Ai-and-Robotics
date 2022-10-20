import gym
import NNutil
import numpy as np
from operator import itemgetter
from timeit import default_timer as timer


#initate

config = NNutil.get_config()

env = gym.make("CartPole-v1")  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False #0.05
patience = 10



parentGenomes = NNutil.generate_initial_batch(config['populationSize'])

generationList = []

startTime = timer()
simTime = []

maxReward = []
avgReward = []



for gCounter in range(config['generations']):

    parentResults = []
    parents = []

    t = timer()

    for n in range(config['populationSize']):

        genome = parentGenomes[n] #shape(4,)..?

        totReward = 0

        for _ in range(config['maxAttempts']):

            for _ in range(config['maxSteps']):

                observation, reward, terminated, truncated, info = env.step(NNutil.get_action(observation, genome)) # 0.015-0.02ms (can't realy do anything about this one)

                totReward += reward - abs(observation[0])/2.4 # må nok endres litt

                if terminated or truncated:

                    observation, info = env.reset()

                    break

        avgGenomeReward = round((totReward/config['maxAttempts']), 1)

        parentResults.append(avgGenomeReward)

        parents.append([parentGenomes[n], parentResults[n]])



    simTime.append(round((timer()-t)*1000/config['populationSize'], 1))

    parents = sorted(parents, key=itemgetter(1))

    parentGenomes = NNutil.evolve(parents)

    parentGenomes = np.vstack((parentGenomes, NNutil.generate_initial_batch(config['populationSize']-len(parentGenomes))))

    maxReward.append(list(map(itemgetter(1), parents))[-1])

    avgReward.append(round(np.average(parentResults), 1))

    generationList.append(parentResults)



    print(f"Gen: {str(gCounter+1).zfill(3)} maxR: {maxReward[gCounter]} avgR {avgReward[gCounter]} dT: {round(timer()-startTime, 1)} aT: {simTime[gCounter]}ms")

    if maxReward[gCounter] > 450:

        print(parents[-1])

    if (gCounter > patience-1) and learningTreshold:

        avgLearningRate = np.average(np.diff(parentResults[-patience:]))

        print(f"learning th: {learningTreshold}, based on {parentResults[-patience:]}, avg learning rate: {avgLearningRate}")

        if abs(avgLearningRate) < abs(learningTreshold):

            break

    if (gCounter == config['generations']-1 or config['livePlot'] == 'true'):

        NNutil.plot(maxReward, avgReward, generationList)


print(parents[-1][0])
