import csv
import gym
import numpy as np

my_env = gym.make("gym_helicopter.envs:helicopter-v2")
# my_env = gym.make("CustomEnv-v0")
done = False
initial_state = my_env.initial_states.copy()
for i in range(3):
    for j in range(3):
        for k in range(3):
            my_env.initial_states[9] =  i - 1
            my_env.initial_states[10] =  j - 1
            my_env.initial_states[11] =  k - 1
            observation = my_env.reset()
            done = False
            while not done:
                observation, b, done, _ = my_env.step(np.array((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.float64))
