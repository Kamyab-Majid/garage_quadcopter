from gym.envs.registration import register

register(
    id='helicopter-v2',
    entry_point='gym_helicopter.envs:HelicopterEnv',
)
# abc = gym.make('gym_helicopter.envs:helicopter-v2')