from gym.envs.registration import register

register(
    id='SpotDog-v0',
    entry_point='spotdog_env.spotdog_env:SpotDogEnv',
)
