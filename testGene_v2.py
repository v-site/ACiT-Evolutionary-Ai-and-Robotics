import gym
import util

genome = '00100111110100111010001000000010001011001110011000001101101100101101000010100111110010111110101011111011110100101111101100100011'
#10101011101011110111101000001011011110101011100001110110011101000100001101101000000000001101100000111101111100110011011000010101
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
