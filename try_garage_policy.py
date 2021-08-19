# Load the policy
from garage.experiment import Snapshotter
import tensorflow as tf  # optional, only for TensorFlow as we need a tf.Session
import gym
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
snapshotter = Snapshotter()
data = snapshotter.load("data/local/experiment/sac_helicopter_resume_7_final")
policy = data["algo"].policy

# You can also access other components of the experiment
try_env = data["env"]
tot_reward = 0
steps, max_steps = 0, 500000
for i in range(10):
    done = False
    obs = try_env.reset()[0]  # The initial observation
    policy.reset()
    last_act = policy.get_action(obs)[1]["mean"]
    while steps < max_steps and not done:
        try:
            all_data = try_env.step(policy.get_action(obs)[1]["mean"] * 0.1 + last_act * 0.9)
            last_act = policy.get_action(obs)[1]["mean"]
            obs = all_data.observation
            done = all_data.terminal
            rew = all_data.reward
            # env.render()  # Render the environment to see what's going on (optional)
            steps += 1
            print(rew)
            tot_reward += rew
        except RuntimeError:
            done = True
