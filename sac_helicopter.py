#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import optuna
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, RaySampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer
import csv
import logging
import sys


@wrap_experiment(archive_launch_repo=False,snapshot_gap=500,snapshot_mode="gap")
def sac_helicopter(
    ctxt=None,
    seed=1,
    gamma=0.99,
    gradient_steps_per_itr=100,
    max_episode_length=1000,
    batch_size=100,
    net_arch=[256, 256],
    min_std=-20,
    max_std=1,
    buffer_size=1e6,
    min_buffer_size=1e5,
    tau=5e-3,
    steps_per_epoch=1,
    normalization=0,
):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    if normalization == 1:
        env = normalize(GymEnv("gym_helicopter.envs:helicopter-v2", max_episode_length=max_episode_length))
    else:
        env = GymEnv("gym_helicopter.envs:helicopter-v2", max_episode_length=max_episode_length)

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=net_arch,
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=nn.Tanh,
        min_std=np.exp(min_std),
        max_std=np.exp(max_std),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=net_arch, hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=net_arch, hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(buffer_size))

    sampler = RaySampler(
        agents=policy, envs=env, max_episode_length=max_episode_length, worker_class=FragmentWorker
    )

    sac = SAC(
        env_spec=env.spec,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        sampler=sampler,
        gradient_steps_per_itr=gradient_steps_per_itr,
        max_episode_length_eval=max_episode_length,
        replay_buffer=replay_buffer,
        min_buffer_size=min_buffer_size,
        target_update_tau=tau,
        discount=gamma,
        buffer_batch_size=batch_size,
        reward_scale=1.0,
        steps_per_epoch=steps_per_epoch,
        eval_env=env,
        initial_log_entropy = 1
    )

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=5000, batch_size=batch_size)
    return policy, env


policy, try_env = sac_helicopter(
    seed=521,
    gamma=0.999,
    gradient_steps_per_itr=2,
    max_episode_length=4000,
    batch_size=8,
    net_arch=[1000, 800, 400],
    min_std=-20,
    max_std=-1,
    buffer_size=1_000_000,
    min_buffer_size=100_000,
    tau=5e-3,
    steps_per_epoch=16,
    normalization=0,
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
    obs = try_env.reset()[0]  # The initial observation
    policy.reset()
    while steps < max_steps and not done:
        try:
            all_data = try_env.step(policy.get_action(obs)[1]['mean'])
            obs = all_data.observation
            done = all_data.terminal
            rew = all_data.reward
            # env.render()  # Render the environment to see what's going on (optional)
            steps += 1
            tot_reward += rew
        except RuntimeError:
            done = True
