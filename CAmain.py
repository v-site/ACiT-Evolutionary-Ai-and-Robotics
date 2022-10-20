import gym
import CAutil
import numpy as np
from operator import itemgetter
from timeit import default_timer as timer



config = CAutil.get_config()

env = gym.make("CartPole-v1")  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False #0.05
patience = 10



parentGenomes = CAutil.generate_initial_batch(config['populationSize']) #0.13 ms
conditionList = CAutil.set_condition_list()

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

        rules = dict(zip(conditionList, CAutil.initialize_rules(parentGenomes[n]))) #0.01-0.02 ms

        totReward = 0

        for _ in range(config['maxAttempts']):

            for _ in range(config['maxSteps']):

                observation, reward, terminated, truncated, info = env.step(CAutil.get_action(observation, rules))

                totReward += reward - abs(observation[0])/2.4

                if terminated or truncated:

                    observation, info = env.reset()

                    break

        avgGenomeReward = round((totReward/config['maxAttempts']), 1)

        parentResults.append(avgGenomeReward)

        parents.append([parentGenomes[n], parentResults[n]])



    simTime.append(round((timer()-t)*1000/config['populationSize'], 1))

    parents = sorted(parents, key=itemgetter(1))

    parentGenomes = CAutil.evolve(parents)

    parentGenomes += CAutil.generate_initial_batch(config['populationSize']-len(parentGenomes)) #add random genoms to satisfy batch size

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

        CAutil.plot(maxReward, avgReward, generationList)


print(parents[-1][0])
