import gym
import util

genome = '10101011101011110111101000001011011110101011100001110110011101000100001101101000000000001101100000111101111100110011011000010101'
#10101011101011110111101000001011011110101011100001110110011101000100001101101000000000001101100000111101111100110011011000010101
#10101011101011110101101000001011011110101011100001111110011101000101001101101000000000000101100000111101111100110010011000010101
#00110111000010011000000010010111010011001011101000000010011001001010110111010100101010101110011011000011011001110101111111010011

config = util.get_config()
#initate
env = gym.make("CartPole-v1", render_mode='human')  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False #0.05
patience = 10

config = util.get_config()

conditionList = util.set_condition_list(config['windowLength']) #0.022 ms

rules = dict(zip(conditionList, util.initialize_rules(config['windowLength'],genome))) #0.01-0.02 ms
totReward = 0

for _ in range(config['maxAttempts']):
    #print('New attempts')
    for _ in range(config['maxSteps']):

        action = util.get_action(config['worldWith'], observation, config['windowSpacing'], config['windowLength'], config['votingMethod'], rules, config['iterations']) # 0.16-0.22 ms (this is by far the longest runner, and it gets performed 200 times at a time)
        observation, reward, terminated, truncated, info = env.step(action) # 0.015-0.02ms (can't realy do anything about this one)
        totReward += reward
        #print(f"cart velocity: {observation[1]}, pole vel: {observation[3]}")
        if terminated or truncated:
            observation, info = env.reset()
env.close()
