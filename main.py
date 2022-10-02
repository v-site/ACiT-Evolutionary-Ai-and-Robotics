import gym
import util
import numpy as np
from timeit import default_timer as timer # Supposedly more better :))
from operator import itemgetter

#initate
env = gym.make("CartPole-v1")  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = 0.05
patience = 5

#initiate CA world
worldWidth = 16 #should be even number
windowLength = 5 #must be odd (3 gives a genome length of 8 bit (2^3), 5; 32, 7; 128, 9; 256, 12; 1024)
votingMethod = 'equal_split'
iterations = 5
maxSteps = 200 #this allows the genom to respawn, if the simulation is terminated
batchSize = 100 #should be divisible by crossover ratio

epochs = 100 # defines the ammounts of epochs for the evolutionary algorithm

parentGenomes = util.generate_initial_batch(batchSize, windowLength) #0.13 ms
conditionList = util.set_condition_list(windowLength) #0.022 ms

epochPerformance = []
startTime = timer()
avgSimTime = []


#observer
eCounter = 0
for _ in range(epochs):

    eCounter += 1
    n = 0

    parentResults = np.array()
    parents = np.array()


    t = timer()


    for _ in range(batchSize):

        genomeEpisodes = 1 #number of episodes within the maxSteps
        genomeReward = 0 #accumulative reward over maxSteps

        rules = dict(zip(conditionList, util.initialize_rules(windowLength,parentGenomes[n]))) #0.01-0.02 ms

        for _ in range(maxSteps):

            action = util.get_action(worldWidth, observation[2], windowLength, votingMethod, rules, iterations) # 0.16-0.22 ms (this is by far the longest runner, and it gets performed 200 times at a time)
            observation, reward, terminated, truncated, info = env.step(action) # 0.015-0.02ms (can't realy do anything about this one)
            genomeReward += 1

            if terminated or truncated:
                observation, info = env.reset()
                genomeEpisodes += 1

        n += 1

        parentResults.append(round(genomeReward/genomeEpisodes, 1))

################################ VVV only run once per epoch, don't care (0.2ms) VVV #########################################################

    avgSimTime.append(round((timer()-t)*1000/batchSize, 1))

    for n in range(batchSize):
        parents.append([parentGenomes[n], parentResults[n]])

    parents = sorted(parents, key=itemgetter(1))

    env.close()

    parentGenomes = util.evolve(parents, 0.2, 'uniform')

    parentGenomes += (util.generate_initial_batch(batchSize-len(parentGenomes), windowLength)) #add random genoms to satisfy batch size
    
    epochPerformance.append(round(np.average(parentResults), 2))
    maxReward = list(map(itemgetter(1), parents))[-1]
    print(f"Generation: {eCounter} maxR: {list(map(itemgetter(1), parents))[-1]} avgR {round(np.average(parentResults), 2)} dT: {round(time.time()-startTime, 2)}")
    if maxReward > 99:
        print(f"Generation: {eCounter} maxR: {list(map(itemgetter(1), parents))[-1]} avgR {round(np.average(parentResults), 2)} dT: {round(time.time()-startTime, 2)}")
        print(parents[-1])
        
    #early stop based on average performance?
    if epochs > patience:
        avgLearningRate = np.average(np.diff(parents[:patience]))
        if avgLearningRate < learningTreshold:
            break


    epochPerformance.append(round(np.average(parentResults), 1))

    #maxReward = list(map(itemgetter(1), parents))[-1]

    print(f"Gen: {str(eCounter).zfill(3)} maxR: {list(map(itemgetter(1), parents))[-1]} avgR {round(np.average(parentResults), 1)} dT: {round(timer()-startTime, 1)} aT: {avgSimTime[eCounter-1]}ms")

    #if maxReward > 99:
        #print(parents[-1])

print(parents)
print('average time per genome ', np.average(avgSimTime), 'ms')