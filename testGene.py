import gym
import CAutil

env = gym.make("CartPole-v1", render_mode = 'human')
observation, info = env.reset()


genome = '11110111011001011010101011000001' #'11110000010000000101000100100100'

worldWidth = conf
windowLength = len(list(format(len(list(genome)),'b')))-1
votingMethod = 'equal_split'
maxSteps = 500

genomeReward = 0
genomeEpisodes = 0

rules = dict(zip(CAutil.set_condition_list(windowLength), CAutil.initialize_rules(windowLength=windowLength, genome=genome)))
print(rules)

for _ in range(maxSteps):
    action = CAutil.get_action(worldWidth, observation[2], windowLength, votingMethod, rules, iterations=5)
    observation, reward, terminated, truncated, info = env.step(action)
    genomeReward += 1

    if terminated or truncated:
        observation, info = env.reset()
        genomeEpisodes += 1

env.close()

print(genomeEpisodes,'\n',genomeReward/genomeEpisodes)