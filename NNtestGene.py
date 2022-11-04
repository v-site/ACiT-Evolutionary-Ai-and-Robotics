import gym
import src.NNutil as NNutil

# predefine genome for dtesting
genome = [0.83151852849641, 0.5667395410331488, 1.0377943781978949, 0.9111872106718677]

# initialize simulation
config = NNutil.get_config()
env = gym.make("CartPole-v1", render_mode='human')
observation, info = env.reset()

# give the genome n chances
for _ in range(config['maxAttempts']):
    
    # counts timesteps
    steps = 0

    # maximum number of time steps befor termination
    for _ in range(5000):
        
        # get action based on environment state
        observation, reward, terminated, truncated, info = env.step(NNutil.get_action(observation, genome))

        # increment step counter
        steps += 1

        # terminate environment if genome fails
        if terminated:
            
            # reset environment
            observation, info = env.reset()
            
            # exit current run
            break
    
    # print time step reached
    print(steps)

# close environment
env.close()
