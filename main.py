import gym
import util
import threading
import numpy as np
from operator import itemgetter
from timeit import default_timer as timer # Supposedly more better :))

#initate

config = util.get_config()

env = gym.make("CartPole-v1")  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False #0.05
patience = 10

#initiate CA world
worldWidth   = config['worldWith'] #should be even number
windowLength = config['windowLength'] #must be odd (3 gives a genome length of 8 bit (2^3), 5; 32, 7; 128, 9; 256, 12; 1024)
votingMethod = config['votingMethod']
iterations   = config['iterations']
maxSteps     = config['maxSteps'] #this allows the genom to respawn, if the simulation is terminated
popSize      = config['populationSize'] #should be divisible by crossover ratio
generations  = config['generations'] # defines the ammounts of generations foconfig[]he evolutionary algorithm

parentGenomes = util.generate_initial_batch(popSize, windowLength) #0.13 ms
conditionList = util.set_condition_list(windowLength) #0.022 ms

generationList = []

startTime = timer()
avgSimTime = []

config = util.get_config()

maxReward = []
avgReward = []

#observer

eCounter = 0

for _ in range(generations):

    eCounter += 1
    n = 0

    parentResults = []
    parents = []

    t = timer()

    for _ in range(popSize):

        genomeEpisodes = 1 #number of episodes within the maxSteps

        rules = dict(zip(conditionList, util.initialize_rules(windowLength,parentGenomes[n]))) #0.01-0.02 ms
        totReward = 0

        for _ in range(config['maxAttempts']):

            for _ in range(config['maxSteps']):

                action = util.get_action(worldWidth, observation, config['windowSpacing'], windowLength, votingMethod, rules, iterations) # 0.16-0.22 ms (this is by far the longest runner, and it gets performed 200 times at a time)
                observation, reward, terminated, truncated, info = env.step(action) # 0.015-0.02ms (can't realy do anything about this one)
                totReward += reward - abs(observation[0])/2.4 # må nokk endres litt

                if terminated or truncated:
                    observation, info = env.reset()
                    break

        n += 1
        avgGenomeReward = round((totReward/config['maxAttempts']),1)
        parentResults.append(avgGenomeReward)

################################ VVV only run once per epoch, don't 5care (0.2ms) VVV #########################################################

    avgSimTime.append(round((timer()-t)*1000/popSize, 1))

    for n in range(popSize):

        parents.append([parentGenomes[n], parentResults[n]])

    parents = sorted(parents, key=itemgetter(1))

    env.close()

    parentGenomes = util.evolve(parents=parents, cutSize=config['cutSize'], breedType=config['breedType'], mutationRatio=config['mutationRatio'])

    parentGenomes += (util.generate_initial_batch(popSize-len(parentGenomes), windowLength)) #add random genoms to satisfy batch size

    maxReward.append(list(map(itemgetter(1), parents))[-1])

    avgReward.append(round(np.average(parentResults), 1))

    generationList.append(parentResults)

    if maxReward[eCounter-1] > 400:

        print(f"Generation: {eCounter} maxR: {maxReward[eCounter-1]} avgR {avgReward[eCounter-1]} dT: {round(timer()-startTime, 1)} aT: {avgSimTime[eCounter-1]}ms")
        print(parents[-1])

    #early stop based on average performance?
    if (eCounter > patience) and learningTreshold:

        avgLearningRate = np.average(np.diff(parentResults[-patience:]))
        print(f"learning th: {learningTreshold}, based on {parentResults[-patience:]}, avg learning rate: {avgLearningRate}")

        if abs(avgLearningRate) < abs(learningTreshold):

            break

    print(f"Gen: {str(eCounter).zfill(3)} maxR: {maxReward[eCounter-1]} avgR {avgReward[eCounter-1]} dT: {round(timer()-startTime, 1)} aT: {avgSimTime[eCounter-1]}ms")

    ploting = threading.Thread(target=util.plot(maxReward, avgReward, generationList))
    ploting.start()

print('\n', 'Average time per genome ', round(np.average(avgSimTime), 2), 'ms')
