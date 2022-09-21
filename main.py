import gym
import util
import CAmanager


#initate
env = gym.make("CartPole-v1", render_mode="human")  #render_mode can either be none (headless) or human (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability

#initiate CA world
poleAngle = observation[2] #get the initial pole angle
worldWith = 8

util.initalize_window(worldWith,poleAngle)

#run CA ruleset (update) for n times

action = CAmanager.voting()
#update - sends actions to OpenAIGym

action = env.action_space[action] #the action provided from the CA


for _ in range(100):
    action = env.action_space.sample() #env.action_space.sample() can be used to randomize the action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

