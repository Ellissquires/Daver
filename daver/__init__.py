from gym.envs.registration import register

register(
    id='daver-v0',
    entry_point='daver.envs:DaverEnv',
)