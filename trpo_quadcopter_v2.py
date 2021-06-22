#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm (PyTorch).

Here it runs InvertedDoublePendulum-v2 environment with 100 iterations.
"""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer
import gym
import envs
from garage.torch.optimizers import ConjugateGradientOptimizer, OptimizerWrapper

@wrap_experiment(archive_launch_repo=False)
def trpo_quadcopter(ctxt=None, seed=1, fake_env=None, env=None):
    """Train TRPO with InvertedDoublePendulum-v2 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    trainer = Trainer(ctxt)

    policy = GaussianMLPPolicy(
        env.spec, hidden_sizes=(400, 400), hidden_nonlinearity=torch.relu, output_nonlinearity=torch.tanh
    )

    value_function = GaussianMLPValueFunction(
        env_spec=env.spec,
        hidden_sizes=(800, 800),
        hidden_nonlinearity=torch.relu,
        output_nonlinearity=torch.tanh,
    )

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=512)
    algo = TRPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        gae_lambda=1.0,
        sampler=sampler,
        discount=0.999,
        center_adv=True,
        entropy_method='no_entropy',
        policy_optimizer=OptimizerWrapper(
            (ConjugateGradientOptimizer, dict(max_constraint_value=0.1)), policy
        ),
    )

    trainer.setup(algo, env)
    #     trainer.restore('data/local/experiment/trpo_quadcopter_2')
    trainer.train(n_epochs=100_000, batch_size=128)


fake_env = gym.make("CustomEnv-v0")
env = GymEnv("CustomEnv-v0", max_episode_length=fake_env.numTimeStep)
trpo_quadcopter(seed=1, fake_env=fake_env, env=env)
