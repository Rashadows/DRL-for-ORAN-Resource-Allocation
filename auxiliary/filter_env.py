import argparser as args
import gym
import numpy as np

################ filter environment ################

def makeFilteredEnv(env):
    """Create a new environment class with actions and states normalized to [-1, 1]."""
    acsp = env.action_space
    obsp = env.observation_space

    if not isinstance(acsp, gym.spaces.Box):
        raise RuntimeError('Environment with continuous action space (i.e., Box) required.')
    if not isinstance(obsp, gym.spaces.Box):
        raise RuntimeError('Environment with continuous observation space (i.e., Box) required.')

    class FilteredEnv(gym.Wrapper):
        def __init__(self, env):
            super(FilteredEnv, self).__init__(env)

            # Observation space normalization
            if np.any(obsp.high < 1e10):
                h = obsp.high
                l = obsp.low
                self.o_c = (h + l) / 2.0
                sc = h - l
                self.o_sc = sc / 2.0
            else:
                self.o_c = np.zeros_like(obsp.high)
                self.o_sc = np.ones_like(obsp.high)

            # Action space normalization
            h = acsp.high
            l = acsp.low
            sc = h - l
            self.a_c = (h + l) / 2.0
            self.a_sc = sc / 2.0

            # Reward normalization
            self.r_sc = 0.1
            self.r_c = 0.0

            # Transformed observation and action spaces
            self.observation_space = gym.spaces.Box(
                low=self.filter_observation(obsp.low),
                high=self.filter_observation(obsp.high),
                dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-np.ones_like(acsp.high, dtype=np.float32),
                high=np.ones_like(acsp.high, dtype=np.float32),
                dtype=np.float32
            )

            # Validation
            self._validate_action_space()

        def filter_observation(self, obs):
            return (obs - self.o_c) / self.o_sc

        def filter_action(self, action):
            return self.a_sc * action + self.a_c

        def filter_reward(self, reward):
            """Has to be applied manually; otherwise, it makes the reward_threshold invalid."""
            return self.r_sc * reward + self.r_c

        def step(self, action):
            # Transform action and clip to valid range
            action_filtered = np.clip(
                self.filter_action(action),
                self.env.action_space.low,
                self.env.action_space.high
            )
            result = self.env.step(action_filtered)
            
            # Ensure result has exactly four values
            if len(result) == 4:
                obs, reward, done, info = result
            else:
                obs, reward, done, info = result[:4]

            # Transform observation
            obs_filtered = self.filter_observation(obs)
            reward = self.filter_reward(reward)

            return obs_filtered, reward, done, info

        def _validate_action_space(self):
            """Validate the transformed action space matches the original."""
            def assert_equal(a, b):
                assert np.allclose(a, b), f"{a} != {b}"
            
            assert_equal(self.filter_action(self.action_space.low), acsp.low)
            assert_equal(self.filter_action(self.action_space.high), acsp.high)

    # Instantiate and wrap the environment
    fenv = FilteredEnv(env)

    print(f'True action space: {acsp.low}, {acsp.high}')
    print(f'True state space: {obsp.low}, {obsp.high}')
    print(f'Filtered action space: {fenv.action_space.low}, {fenv.action_space.high}')
    print(f'Filtered state space: {fenv.observation_space.low}, {fenv.observation_space.high}')

    return fenv