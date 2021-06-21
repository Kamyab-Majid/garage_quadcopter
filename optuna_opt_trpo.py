#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
import gym
import envs
import optuna
import shutil
from garage.experiment import Snapshotter
import tensorflow as tf  # optional, only for TensorFlow as we need a tf.Session
from garage.torch.optimizers import ConjugateGradientOptimizer, OptimizerWrapper
import numpy as np

@wrap_experiment(archive_launch_repo=False, snapshot_mode='none')
def trpo_quadcopter(
    ctxt=None,
    seed=1,
    env=None,
    batch_size=512,
    n_steps=2048,
    gamma=0.9999,
    max_constraint_value=0.01,
    arch_size_policy=400,
    arch_size_value=400,
    arch_hid_lay_policy=2,
    arch_hid_lay_value=2,
    center_adv=False,
    gae_lambda=0.98,
    entropy_method='no_entropy'
):
    
    dir_path = "data"
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))
    set_seed(seed)
    trainer = Trainer(ctxt)
    policy = GaussianMLPPolicy(
        env.spec,
        hidden_sizes=[arch_size_policy for i in range(arch_hid_lay_policy)],
        hidden_nonlinearity=torch.relu,
        output_nonlinearity=torch.tanh,
    )

    value_function = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=[arch_size_value for i in range(arch_hid_lay_value)],
        hidden_nonlinearity=torch.relu,
        output_nonlinearity=torch.tanh,
    )

    sampler = LocalSampler(agents=policy, envs=env, max_episode_length=n_steps)

    algo = TRPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        gae_lambda=gae_lambda,
        sampler=sampler,
        discount=gamma,
        center_adv=center_adv,
        entropy_method=entropy_method,
        policy_optimizer=OptimizerWrapper(
            (ConjugateGradientOptimizer, dict(max_constraint_value=max_constraint_value)), policy
        ),
    )

    trainer.setup(algo, env)
    #     trainer.restore('data/local/experiment/trpo_quadcopter_2')
    trainer.train(n_epochs=10, batch_size=batch_size)
    return policy


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    arch_size_policy = trial.suggest_categorical("arch_size_policy", [100, 200, 400, 800])
    arch_hid_lay_policy = trial.suggest_categorical("arch_hid_lay_policy", [1, 2, 3])
    arch_hid_lay_value = trial.suggest_categorical("arch_hid_lay_value", [1, 2, 3])
    arch_size_value = trial.suggest_categorical("arch_size_value", [100, 200, 400, 800])
    max_constraint_value = trial.suggest_categorical("max_constraint_value", [0.005, 0.01, 0.1, 0.5, 1])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    entropy_method = trial.suggest_categorical("net_arch", ["regularized", "no_entropy"])
    center_adv = trial.suggest_categorical("center_adv", [False, True])
    env = GymEnv("CustomEnv-v0", max_episode_length=n_steps)
    try_env = gym.make("CustomEnv-v0")
    policy = trpo_quadcopter(
        seed=1,
        env=env,
        batch_size=batch_size,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_constraint_value=max_constraint_value,
        arch_size_policy=arch_size_policy,
        arch_hid_lay_policy=arch_hid_lay_policy,
        arch_size_value=arch_size_value,
        arch_hid_lay_value=arch_hid_lay_value,
        center_adv=center_adv,
        entropy_method=entropy_method

    )
#     snapshotter = Snapshotter()
#     with tf.compat.v1.Session():  # optional, only for TensorFlow
#         data = snapshotter.load("data/local/experiment/trpo_quadcopter")
#     policy = data["algo"].policy
    # You can also access other components of the experiment
    tot_reward = 0
    steps, max_steps = 0, 500000
    for i in range(10):
        done = False
        obs = try_env.reset()  # The initial observation
        policy.reset()
        while steps < max_steps and not done:
            obs, rew, done, _ = try_env.step(policy.get_action(obs)[0])
            # env.render()  # Render the environment to see what's going on (optional)
            steps += 1
            tot_reward += rew
    return tot_reward


# snapshotter = Snapshotter()
# # fake_env = gym.make("CustomEnv-v0")
# # env = GymEnv("CustomEnv-v0", max_episode_length=fake_env.numTimeStep)
# fake_env = gym.make("CustomEnv-v0")
# env = GymEnv("CustomEnv-v0", max_episode_length=fake_env.numTimeStep)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)
study.trials_dataframe().to_csv(path_or_buf='optuna_results.csv')
# trpo_quadcopter(seed=1, fake_env=fake_env, env=env)
