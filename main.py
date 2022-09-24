import gym
import util
import random

#initate
env = gym.make("CartPole-v1")  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability

#initiate CA world
worldWidth = 16 #must be even number
windowLength = 5 #must be odd (3 gives a genome length of 8 bit, 5; 32, 7, 125, 9, 256, 12, 1024)
votingMethod = 'equal_split'
maxSteps = 200 #this allows the genom to respawn, if the simulation is terminated
batchSize = 100

parentGenomes = []
parentResults = []

#observer
for _ in range(batchSize):

    genomeEpisodes = 0 #number of episodes within the maxSteps
    genomeReward = 0 #accumulative reward over maxSteps

    genome = random.randint(0, 2**(2**windowLength)-1) #must be less or equal to 2**(2**windowLength)-1 (for windowlength of 3, genome = (0,255))
    parentGenomes.append(format(genome, ('0' + str(2**windowLength) + 'b')))

    for _ in range(maxSteps):
        action = util.get_action(worldWidth, observation[2], windowLength, votingMethod, genome)
        #env.action_space.sample() can be used to randomize the action
        observation, reward, terminated, truncated, info = env.step(action)
        genomeReward += 1

        if terminated or truncated:
            observation, info = env.reset()
            genomeEpisodes += 1

    parentResults.append(round(genomeReward/genomeEpisodes, 2))

parents = {k: v for k, v in sorted(dict(zip(parentGenomes, parentResults)).items(), key=lambda item: item[1])}

env.close()

print(parents, '\n')

print('offspring \n', util.evolve(parents, 0.2, 'one-point-crossover', 'deterministically', 0.8))


#print (f"Episodes: {genomeEpisodes}")
#print (f"Genome accumulated reward over {maxSteps} steps: {genomeReward}")
#print (f"Reward / episodes: {round(genomeReward/genomeEpisodes, 2)}")

