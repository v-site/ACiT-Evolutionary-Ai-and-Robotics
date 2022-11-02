import gym
import CAutil

genome = '11011110100000111110001111001100000011111101010100010111001111100101011101101110010101001001011011000010111110000111111101011010'
#Interesting:10001111100011101001000100100010110010101100011110000010001101101101001000100101100111011111001011101011101111010011001001110100
#494.7: 11100101101110100010001001010001100101011000000110000000101110001100110010011010000001010111111010100110110100010110001101110101

config = CAutil.get_config()
env = gym.make("CartPole-v1", render_mode='human')
observation, info = env.reset()

rules = dict(zip(CAutil.set_condition_list(), CAutil.initialize_rules(genome)))


for _ in range(config['maxAttempts']):

    totReward = 0

    for _ in range(5000):

        observation, reward, terminated, truncated, info = env.step(CAutil.get_action(observation, rules))

        totReward += 1

        if terminated:
            observation, info = env.reset()
            break

    print(totReward)


env.close()
