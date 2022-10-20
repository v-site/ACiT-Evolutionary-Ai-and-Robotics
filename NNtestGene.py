import gym
import NNutil

genome = [1.84858023, 0.36366978, 0.50911942, 0.68024333]

config = NNutil.get_config()

env = gym.make("CartPole-v1", render_mode='human')  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False
patience = 10

for _ in range(config['maxAttempts']):

    for _ in range(config['maxSteps']):

        observation, reward, terminated, truncated, info = env.step(NNutil.get_action(observation, genome))

        if terminated or truncated:

            observation, info = env.reset()

env.close()
