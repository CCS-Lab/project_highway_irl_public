from gym.envs.registration import register

register(
    id='IRL-v2',
    entry_point='highway_irl_v2.envs.irl_env:IRLEnvV2',
)
