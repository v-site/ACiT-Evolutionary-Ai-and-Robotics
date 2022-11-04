import gym
import src.CAutil as CAutil

genome = '11100110011011000010101101100001110011010111001010001000000101111011101110001000010010010100011111100101110001001101001111011010' #15,7,5,6

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
