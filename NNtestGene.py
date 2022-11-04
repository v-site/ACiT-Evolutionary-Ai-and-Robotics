import gym
import src.NNutil as NNutil

genome = [0.83151852849641, 0.5667395410331488, 1.0377943781978949, 0.9111872106718677]

config = NNutil.get_config()
env = gym.make("CartPole-v1", render_mode='human')
observation, info = env.reset()

for _ in range(config['maxAttempts']):

    totReward = 0

    for _ in range(5000):

        observation, reward, terminated, truncated, info = env.step(NNutil.get_action(observation, genome))

        totReward += 1

        if terminated:

            observation, info = env.reset()

            break

    print(totReward)


env.close()
