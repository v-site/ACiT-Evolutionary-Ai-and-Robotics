import gym
import util
import random

#initate
env = gym.make("CartPole-v1", render_mode="human")  #render_mode can either be none (headless) or human (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability

#initiate CA world
worldWidth = 8 #must be even number
windowLength = 3 #must be odd number
votingMethod = 'equal_split'
maxSteps = 100 #this allows the genom to respawn, if the simulation is terminated
batchSize = 10

parentGenomes = []
parentResults = []

#observer
for _ in range(batchSize):

    genomeEpisodes = 0 #number of episodes within the maxSteps
    genomeReward = 0 #accumulative reward over maxSteps

    genome = random.randint(0,255) #must be less or equal to 2**2**windowlength (for windowlength of 3, genome = (0,255))
    parentGenomes.append(format(genome, ('0' + str(2 ** windowLength) + 'b')))

    for _ in range(maxSteps):
        action = util.get_action(worldWidth, observation[2], windowLength, votingMethod, genome)
        print(action)
        #env.action_space.sample() can be used to randomize the action
        observation, reward, terminated, truncated, info = env.step(action)
        genomeReward +=1

        if terminated or truncated:
            observation, info = env.reset()
            genomeEpisodes +=1

    parentResults.append(round(genomeReward/genomeEpisodes, 2))

env.close()

parents = dict(zip(parentGenomes, parentResults))

print(parents)
#print (f"Episodes: {genomeEpisodes}")
#print (f"Genome accumulated reward over {maxSteps} steps: {genomeReward}")
#print (f"Reward / episodes: {round(genomeReward/genomeEpisodes, 2)}")

