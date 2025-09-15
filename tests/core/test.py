import gymnasium as gym
import numpy as np

_TRIAL_LEN = 5
SEED = 0
_REW_C = 0.1

class MockVecEnv(gym.Env):
    def __init__(self):
        self.num_envs = 3
        self.pos = np.array([[1.0], [1.0], [1.0]])
        self.vel = np.array([[0.0], [0.0], [0.0]])
        self.time_left = _TRIAL_LEN
        self.observation_space = gym.spaces.Box(
            -np.inf * np.ones(2), np.inf * np.ones(2), dtype=np.float32, shape=(2,)
        )
        self.action_space = gym.spaces.Box(
            -np.ones(1), np.ones(1), dtype=np.float32, shape=(1,)
        )
        self.action_space.seed(SEED)
        self.observation_space.seed(SEED)
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.pos = np.array([[1.0], [1.0], [1.0]])
        self.vel = np.array([[0.0], [0.0], [0.0]])
        self.time_left = _TRIAL_LEN
        return np.concatenate([self.pos, self.vel], axis=-1), {}
    
    def step(self, action: np.ndarray):
        self.vel += action
        self.pos += self.vel
        self.time_left -= 1
        done = np.array([self.time_left == 0] * self.num_envs)
        truncated = np.array([False] * self.num_envs)
        reward = -_REW_C * (self.pos ** 2)
        return np.concatenate([self.pos, self.vel], axis=-1), reward.squeeze(-1), done, truncated, {}


env = MockVecEnv()
obs, info = env.reset()
print("Initial Observation shape:", obs.shape)

action = np.array([[0.5], [0.5], [0.5]])
obs, reward, done, truncated, info = env.step(action)
print("Next Observation shape:", obs.shape)
print("Reward shape:", reward.shape)
print("Reward:", reward)
print("Done:", done)
print("Truncated:", truncated)