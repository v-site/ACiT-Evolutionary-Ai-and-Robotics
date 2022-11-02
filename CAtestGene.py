import gym
import CAutil

genome = '11000000001010111001110110010001100001001100011001100101100011101111001110110110110010110110100110010101110100010100100011110110' #2,7,1,3
#Interesting:10001111100011101001000100100010110010101100011110000010001101101101001000100101100111011111001011101011101111010011001001110100
#494.7: 11100101101110100010001001010001100101011000000110000000101110001100110010011010000001010111111010100110110100010110001101110101

config = CAutil.get_config()
env = gym.make("CartPole-v1", render_mode='human')
observation, info = env.reset() #seed=config['seed']

rules = dict(zip(CAutil.set_condition_list(), CAutil.initialize_rules(genome)))

totReward = 0

for _ in range(1000):

    observation, reward, terminated, truncated, info = env.step(CAutil.get_action(observation, rules))

    totReward += 1

    if terminated:
        break

env.close()

print(totReward)
