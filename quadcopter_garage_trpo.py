import gym
from garage.envs import GymEnv
import envs
from garage import wrap_experiment
from garage.envs import PointEnv
from garage.envs import normalize
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer
my_env = GymEnv("CustomEnv-v0")  # shorthand for GymEnv(gym.make('CartPole-v1'))
env = normalize(my_env)