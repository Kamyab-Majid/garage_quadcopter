# Load the policy
from garage.experiment import Snapshotter
import tensorflow as tf  # optional, only for TensorFlow as we need a tf.Session
import gym
snapshotter = Snapshotter()
with tf.compat.v1.Session():  # optional, only for TensorFlow
    data = snapshotter.load("data/local/experiment/trpo_quadcopter_2")
policy = data["algo"].policy

# You can also access other components of the experiment
env = data["env"]
env = gym.make("CustomEnv-v0")
steps, max_steps = 0, 500000
done = False
obs = env.reset()  # The initial observation
policy.reset()

while steps < max_steps and not done:
    obs, rew, done, _ = env.step(policy.get_action(obs)[0])
    # env.render()  # Render the environment to see what's going on (optional)
    steps += 1
    tot_reward += rew

env.close()