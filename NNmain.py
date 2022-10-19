import gym
import NNutil
import threading
import numpy as np
from operator import itemgetter
from timeit import default_timer as timer # Supposedly more better :))


#initate

config = NNutil.get_config()

env = gym.make("CartPole-v1")  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False #0.05
patience = 10
avgSimTime = []
maxReward = []
avgReward = []
generationList = []

startTime = timer()

population = NNutil.generate_initial_population(config['populationSize'])

gCounter = 0

for _ in range(config['generations']):

    gCounter += 1
    n = 0

    parentResults = []
    parents = []

    t = timer()

    for _ in range(config['populationSize']):
        genome = population[n] #shape(4,)..?
        genomeEpisodes = 1 #number of episodes within the maxSteps
        totReward = 0

        for _ in range(config['maxAttempts']):

            for _ in range(config['maxSteps']):

                action = NNutil.get_action(observation, genome)

                observation, reward, terminated, truncated, info = env.step(action) # 0.015-0.02ms (can't realy do anything about this one)
                
                totReward += reward - abs(observation[0])/2.4 # må nok endres litt

                if terminated or truncated:
                    observation, info = env.reset()
                    break

        n += 1
        avgGenomeReward = round((totReward/config['maxAttempts']),1)
        parentResults.append(avgGenomeReward)

################################ VVV only run once per epoch, don't 5care (0.2ms) VVV #########################################################

    avgSimTime.append(round((timer()-t)*1000/config['populationSize'], 1))

    for n in range(config['populationSize']):

        parents.append([population[n], parentResults[n]])

    parents = sorted(parents, key=itemgetter(1))

    env.close()

    population = NNutil.evolve(parents=parents)

    fill = (NNutil.generate_initial_population(config['populationSize']-len(population))) #add random genoms to satisfy batch size
    population = np.vstack((population,fill))

    maxReward.append(list(map(itemgetter(1), parents))[-1])

    avgReward.append(round(np.average(parentResults), 1))

    generationList.append(parentResults)

    if maxReward[gCounter-1] > 400:

        print(f"Generation: {gCounter} maxR: {maxReward[gCounter-1]} avgR {avgReward[gCounter-1]} dT: {round(timer()-startTime, 1)} aT: {avgSimTime[gCounter-1]}ms")
        print(parents[-1])

    #early stop based on average performance?
    if (gCounter > patience) and learningTreshold:

        avgLearningRate = np.average(np.diff(parentResults[-patience:]))
        print(f"learning th: {learningTreshold}, based on {parentResults[-patience:]}, avg learning rate: {avgLearningRate}")

        if abs(avgLearningRate) < abs(learningTreshold):

            break

    print(f"Gen: {str(gCounter).zfill(3)} maxR: {maxReward[gCounter-1]} avgR {avgReward[gCounter-1]} dT: {round(timer()-startTime, 1)} aT: {avgSimTime[gCounter-1]}ms")

    ploting = threading.Thread(target=NNutil.plot(maxReward, avgReward, generationList))
    ploting.start()

print('\n', 'Average time per genome ', round(np.average(avgSimTime), 2), 'ms')
