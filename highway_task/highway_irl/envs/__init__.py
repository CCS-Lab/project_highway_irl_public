from gym.envs.registration import register

register(
    id='IRL-v1',
    entry_point='highway_irl.envs.irl_env:IRLEnvV1',
)
