import gym
from lightgbm import early_stopping
import util
import numpy as np
from timeit import default_timer as timer # Supposedly more better :))
from operator import itemgetter
import time

genome = '00101111000010011101010011010001'
config = util.get_config()
#initate
env = gym.make("CartPole-v1", render_mode='human')  #render_mode = 'human' (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability
learningTreshold = False #0.05
patience = 10

#initiate CA world
worldWidth = config['worldWith'] #should be even number
windowLength = config['windowLength'] #must be odd (3 gives a genome length of 8 bit (2^3), 5; 32, 7; 128, 9; 256, 12; 1024)
votingMethod = config['votingMethod']
iterations = config['iterations']
maxSteps = config['maxSteps'] #this allows the genom to respawn, if the simulation is terminated

conditionList = util.set_condition_list(windowLength) #0.022 ms

epochPerformance = []
startTime = timer()
avgSimTime = []

config = util.get_config()

#observer
 

rules = dict(zip(conditionList, util.initialize_rules(windowLength,genome))) #0.01-0.02 ms
totReward = 0 

for _ in range(config['maxAttempts']):
    #print('New attempts')
    for _ in range(config['maxSteps']):
        
        action = util.get_action(worldWidth, observation, config['windowSpacing'], windowLength, votingMethod, rules, iterations) # 0.16-0.22 ms (this is by far the longest runner, and it gets performed 200 times at a time)
        observation, reward, terminated, truncated, info = env.step(action) # 0.015-0.02ms (can't realy do anything about this one)
        totReward += reward
        print(observation[0])
        #print(f"cart velocity: {observation[1]}, pole vel: {observation[3]}")
        if terminated or truncated:
            observation, info = env.reset()
            #print("you loose!")
            break
################################ VVV only run once per epoch, don't care (0.2ms) VVV #########################################################
