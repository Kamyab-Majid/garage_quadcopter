import ray
from ray import tune

ray.init(ignore_reinit_error=True)

config = {
    "env": "CartPole-v0",
    "num_workers": 2,
    "vf_share_layers": tune.grid_search([True, False]),
    "lr": tune.grid_search([1e-4, 1e-5, 1e-6]),
}
results = tune.run("PPO", stop={"timesteps_total": 100000}, config=config)
