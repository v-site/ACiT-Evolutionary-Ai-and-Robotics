import gym
import util

#initate
env = gym.make("CartPole-v1", render_mode="human")  #render_mode can either be none (headless) or human (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability

#initiate CA world
worldWidth = 8
windowLength = 3
votingMethod = 'equal_split'
genome = 56 #must be less or equal to 2**2**windowlength (for windowlength of 3, genome = (0,256))
maxSteps = 100 #this allows the genom to respawn, if the simulation is terminated
batchSize = 10

genomeEpisodes = 0 #nmber of episodes within the maxSteps
genomeReward = 0 #accumulative reward over maxSteps

#observer
for _ in range(maxSteps):
    action = util.get_action(worldWidth, observation[2], windowLength, votingMethod, genome)
    print(action)
    #env.action_space.sample() can be used to randomize the action
    observation, reward, terminated, truncated, info = env.step(action)
    genomeReward +=1

    if terminated or truncated:
        observation, info = env.reset()
        genomeEpisodes +=1
env.close()
print (f"Episodes: {genomeEpisodes}")
print (f"Genome accumulated reward over {maxSteps} steps: {genomeReward}")
print (f"Reward / episodes: {round(genomeReward/genomeEpisodes, 2)}")