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
import garage
from garage.experiment.deterministic import set_seed


@wrap_experiment
def sac_helicopter_resume(ctxt=None, snapshot_dir="data/local/experiment/sac_helicopter_251", seed=1):
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    trainer.restore(snapshot_dir)
    trainer.resume(n_epochs=2, batch_size=128)


sac_helicopter_resume(seed=521)
