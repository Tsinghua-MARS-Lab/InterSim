from gym.envs.registration import register

register(
    id='Drive-v0',
    entry_point='interactive_sim.envs:DriveEnv',
    max_episode_steps=1000,
    reward_threshold=500.0,
)