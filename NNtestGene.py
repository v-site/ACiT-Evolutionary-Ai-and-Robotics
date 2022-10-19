import gym
import NNutil

genome = [1.20910192, 1.69631326, 0.40255236, 1.36242095]
config = NNutil.get_config()
#initate
env = gym.make("CartPole-v1", render_mode='human')  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False #0.05
patience = 10

config = NNutil.get_config()

totReward = 0

for _ in range(config['maxAttempts']):
    #print('New attempts')
    for _ in range(config['maxSteps']):

        action = NNutil.get_action(observation, genome)
        observation, reward, terminated, truncated, info = env.step(action) # 0.015-0.02ms (can't realy do anything about this one)
        totReward += reward
        #print(f"cart velocity: {observation[1]}, pole vel: {observation[3]}")
        if terminated or truncated:
            observation, info = env.reset()
env.close()
