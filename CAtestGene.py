import gym
import src.CAutil as CAutil

# predefine genome for dtesting
genome = '11100110011011000010101101100001110011010111001010001000000101111011101110001000010010010100011111100101110001001101001111011010' #15,7,5,6

# initialize simulation
config = CAutil.get_config()
env = gym.make("CartPole-v1", render_mode='human')
observation, info = env.reset()

rules = dict(zip(CAutil.set_condition_list(), CAutil.initialize_rules(genome)))

# give the genome n chances
for _ in range(config['maxAttempts']):
    
    # counts timesteps
    steps = 0

    # maximum number of time steps befor termination
    for _ in range(5000):
        
        # get action based on environment state
        observation, reward, terminated, truncated, info = env.step(CAutil.get_action(observation, rules))

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
