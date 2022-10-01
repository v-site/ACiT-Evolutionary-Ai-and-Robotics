import gym
import util

env = gym.make("CartPole-v1", render_mode = 'human')
observation, info = env.reset()

genome = '11110000100111000100100111000000'

worldWidth = 16
windowLength = len(list(format(len(list(genome)),'b')))-1
votingMethod = 'equal_split'
maxSteps = 500

genomeReward = 0
genomeEpisodes = 0

for _ in range(maxSteps):
    action = util.get_action(worldWidth, observation[2], windowLength, votingMethod, genome,iterations=5)
    observation, reward, terminated, truncated, info = env.step(action)
    genomeReward += 1

    if terminated or truncated:
        observation, info = env.reset()
        genomeEpisodes += 1

env.close()

print(genomeEpisodes,'\n',genomeReward/genomeEpisodes)