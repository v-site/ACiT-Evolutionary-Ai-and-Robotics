import gym
import CAutil

genome = '10001110100001000000011011101100100001100110010101010110101111101000101011101000010011111110111011110011110101011110000110111110'

config = CAutil.get_config()
#initate
env = gym.make("CartPole-v1", render_mode='human')  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False
patience = 10

rules = dict(zip(CAutil.set_condition_list(), CAutil.initialize_rules(genome)))

for _ in range(config['maxAttempts']):

    for _ in range(config['maxSteps']):

        observation, reward, terminated, truncated, info = env.step(CAutil.get_action(observation, rules))

        if terminated or truncated:
            observation, info = env.reset()

env.close()
