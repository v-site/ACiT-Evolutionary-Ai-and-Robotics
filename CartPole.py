# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:41:41 2022

@author: Pierre Boniface
"""

import gym
env = gym.make("CartPole-v1", render_mode="human")  #render_mode can either be none (headless) or human (graphical)
observation, info = env.reset() #(seed=42) If sample() is to be used to randomize the actionspace, env.reset needs to be seeded for repeatability

for _ in range(100):
    action = env.action_space.sample() #env.action_space.sample() can be used to randomize the action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()