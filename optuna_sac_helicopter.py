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


@wrap_experiment(archive_launch_repo=False, snapshot_mode="none")
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
        env = normalize(GymEnv('gym_helicopter.envs:helicopter-v2', max_episode_length=max_episode_length))
    else:
        env = GymEnv('gym_helicopter.envs:helicopter-v2', max_episode_length=max_episode_length)

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
        agents=policy, envs=env, max_episode_length=env.spec.max_episode_length, worker_class=FragmentWorker
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
    )

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    trainer.setup(algo=sac, env=env)
    trainer.train(n_epochs=10, batch_size=batch_size)
    return policy, env


def objective(trial):
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size1", [int(3000), int(4000), int(5000), int(10000)])
    min_buffer_size = trial.suggest_categorical("learning_starts", [1, 10, 100, 200, 2000])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    min_std = trial.suggest_categorical("min_std", [-1, -5, -10, -20, -40])
    max_std = trial.suggest_categorical("max_std", [-1, 5, 10, 20, 40])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big", "verybig"])
    gradient_steps_per_itr = trial.suggest_categorical("gradient_steps_per_itr", [1, 2, 5])
    normalization = trial.suggest_categorical("normalization", [0, 1])
    net_arch = {"small": [256, 256], "medium": [400, 300], "big": [256, 256, 256], "verybig": [512, 512, 512]}[net_arch]
    s = np.random.randint(0, 1000)
    policy, try_env = sac_helicopter(
        seed=521,
        gamma=gamma,
        gradient_steps_per_itr=gradient_steps_per_itr,
        max_episode_length=100000,
        batch_size=batch_size,
        net_arch=net_arch,
        min_std=min_std,
        max_std=max_std,
        buffer_size=buffer_size,
        min_buffer_size=min_buffer_size,
        tau=tau,
        steps_per_epoch=train_freq,
        normalization=normalization,
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
                all_data = try_env.step(policy.get_action(obs)[0])
                obs = all_data.observation
                done = all_data.terminal
                rew = all_data.reward
                # env.render()  # Render the environment to see what's going on (optional)
                steps += 1
                tot_reward += rew
            except RuntimeError:
                done = True
    return tot_reward


# def objective(trial):
#     x = trial.suggest_uniform('x', -100, 100)
#     return (x - 2) ** 2

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=100)
optuna.logging.get_logger("optuna_sac_nw").addHandler(logging.StreamHandler(sys.stdout))
study_name = "optuna_sac_nw"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=200)
print("Minimum mean squared error: " + str(study.best_value))
print("Best parameter: " + str(study.best_params))
fields = study.best_params
with open("parameters.csv", "a") as f:
    writer = csv.DictWriter(f, fieldnames=study.best_params.keys())
    writer.writeheader()
    writer.writerow(fields)
study.trials_dataframe().to_csv(path_or_buf="optuna_results_sac.csv")
# trpo_quadcopter(seed=1, fake_env=fake_env, env=env)
