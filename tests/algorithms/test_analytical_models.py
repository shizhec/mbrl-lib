# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for MBPO and Pets with analytical models.
Tests the integration of analytical dynamics models with mbrl-lib algorithms.
"""

import os
import pathlib
import random
import tempfile
from typing import Dict

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml
import hydra
from omegaconf import OmegaConf

import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.env as mbrl_env


# Test configuration constants - smaller for faster testing
_TRIAL_LEN = 15
_NUM_TRIALS_PETS = 3
_NUM_TRIALS_MBPO = 4
_REW_C = 0.001
_INITIAL_EXPLORE = 50
_TARGET_REWARD = -50 * _REW_C

_REPO_DIR = pathlib.Path(os.getcwd())
_DIR = tempfile.TemporaryDirectory()

_SILENT = True
_DEBUG_MODE = False

SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class MockAnalyticalDynamics:
    """
    Mock analytical dynamics model that simulates the same linear system as MockLineEnv.
    This provides perfect dynamics for testing analytical model integration.
    """

    def __init__(self, state_dim=2, action_dim=1, dt=1.0, device="cpu"):
        self.state_dim = state_dim
        self.d_action = action_dim
        self.dt = dt
        self.device = device

    def rollout_open_loop(self, start_state, act_seq):
        """
        Simulate dynamics: vel += action, pos += vel
        This matches the MockLineEnv dynamics exactly.
        """
        batch_size = start_state.shape[0]
        horizon = act_seq.shape[1]

        # Initialize state sequence
        state_seq = torch.zeros((batch_size, horizon, self.state_dim), device=self.device)

        # Current state [pos, vel]
        current_state = start_state.clone()

        for t in range(horizon):
            # Apply action to velocity
            new_vel = current_state[:, 1:2] + act_seq[:, t, :]
            # Update position with new velocity
            new_pos = current_state[:, 0:1] + new_vel

            next_state = torch.cat([new_pos, new_vel], dim=1)
            state_seq[:, t] = next_state
            current_state = next_state

        return {
            'state_seq': state_seq
        }
    

    def obs2state(self, obs):
        """Convert observation to state (identity for this simple env)."""
        return obs


    def state2obs(self, state, initial_obs=None, state_dict=None, **kwargs):
        """Convert state back to observation (identity for this simple env)."""
        return state
    

    def get_next_obs(self, obs, action):
        """Get next observation given current observation and action."""
        state = self.obs2state(obs)
        next_state = self.rollout_open_loop(state, action.unsqueeze(1))['state_seq'][:, 0, :]
        next_obs = self.state2obs(next_state)
        return next_obs


# Use the same MockLineEnv and MockVecEnv from the original test file
class MockLineEnv(gym.Env):
    def __init__(self):
        self.pos = 1.0
        self.vel = 0.0
        self.time_left = _TRIAL_LEN
        self.observation_space = gym.spaces.Box(
            -np.inf * np.ones(2), np.inf * np.ones(2), shape=(2,)
        )
        self.action_space = gym.spaces.Box(-np.ones(1), np.ones(1), shape=(1,))
        self.action_space.seed(SEED)
        self.observation_space.seed(SEED)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.pos = 1.0
        self.vel = 0.0
        self.time_left = _TRIAL_LEN
        return np.array([self.pos, self.vel]), {}

    def step(self, action: np.ndarray):
        self.vel += action.item()
        self.pos += self.vel
        self.time_left -= 1
        reward = -_REW_C * (self.pos ** 2)
        return np.array([self.pos, self.vel]), reward, self.time_left == 0, False, {}


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


def mock_reward_fn(action, obs):
    """Reward function for Pets: minimize position squared."""
    return -_REW_C * (obs[:, 0] ** 2).unsqueeze(1)


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _create_analytical_dynamics_config():
    """Create configuration for analytical dynamics model."""
    return {
        "_target_": "mbrl.models.analytical_model.AnalyticalModel",
        "dynamics_model": {
            "_target_": "tests.algorithms.test_analytical_models.MockAnalyticalDynamics",
            "device": device,
        },
        "reward_fn": None,  # Will use learned rewards
        "device": device,
        "learned_rewards": True,
        "obs_dim": "???",
        "action_dim": "???",
        "in_size": "???",
        "out_size": "???",
    }

def test_analytical_model_basic():
    from mbrl.models import AnalyticalOneDTransitionRewardModel
    cfg = _create_analytical_dynamics_config()

    cfg["obs_dim"] = 2
    cfg["action_dim"] = 1
    cfg["in_size"] = 3
    cfg["out_size"] = 3

    dynamic_model = hydra.utils.instantiate(cfg)

    # Test that it's a proper OneDTransitionRewardModel
    from mbrl.models.analytical_model import AnalyticalModel
    assert isinstance(dynamic_model, AnalyticalModel)

    model = AnalyticalOneDTransitionRewardModel(
        model=dynamic_model,
        learned_rewards=True
    )

    # Test sample interface
    batch_size = 8
    obs = torch.randn(batch_size, 2)  # [pos, vel]
    action = torch.randn(batch_size, 1)
    model_state = {"obs": obs}

    next_state, reward, _, next_model_state = model.sample(action, model_state)

    assert next_state.shape == (batch_size, 2)
    assert reward.shape == (batch_size, 1)
    assert "obs" in next_model_state
    assert next_model_state["obs"].shape == (batch_size, 2)


def _check_pets_analytical_model(vectorized=False):
    """Test Pets with analytical model following mbrl-lib test patterns."""

    # Load base Pets configuration
    with open(_REPO_DIR / "mbrl" / "examples" / "conf" / "algorithm" / "pets.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    # Load action optimizer configuration
    with open(_REPO_DIR / "mbrl" / "examples" / "conf" / "action_optimizer" / "cem.yaml", "r") as f:
        action_optimizer_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": _create_analytical_dynamics_config(),
        "action_optimizer": action_optimizer_cfg,
        "overrides": {
            "use_analytical_model": True,  # Enable analytical model
            "learned_rewards": True,  # Learn rewards from data
            "num_steps": _NUM_TRIALS_PETS * _TRIAL_LEN,
            "model_lr": 1e-3,
            "model_wd": 1e-5,
            "model_batch_size": 64,  # Smaller for testing
            "validation_ratio": 0.1,
            "num_epochs_train_model": 10,  # Fewer epochs for testing
            "patience": 5,
            "cem_elite_ratio": 0.1,
            "cem_population_size": 100,  # Smaller for testing
            "cem_num_iters": 3,
            "cem_alpha": 0.1,
            "cem_clipped_normal": False,
            "planning_horizon": 10,  # Shorter for testing
            "num_elites": 3,
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": device,
    }

    cfg = OmegaConf.create(cfg_dict)
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_PETS + _INITIAL_EXPLORE
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.freq_train_model = _TRIAL_LEN

    if vectorized:
        env = MockVecEnv()
    else:
        env = MockLineEnv()

    term_fn = mbrl_env.termination_fns.no_termination
    reward_fn = mock_reward_fn

    max_reward = pets.train(
        env, term_fn, reward_fn, cfg, silent=_SILENT, work_dir=_DIR.name
    )

    # Should achieve good performance with perfect analytical dynamics
    assert max_reward > _TARGET_REWARD
    return max_reward


def _check_mbpo_analytical_model(vectorized=False):
    """Test MBPO with analytical model following mbrl-lib test patterns."""

    # Load base MBPO configuration
    with open(_REPO_DIR / "mbrl" / "examples" / "conf" / "algorithm" / "mbpo.yaml", "r") as f:
        algorithm_cfg = yaml.safe_load(f)

    cfg_dict = {
        "algorithm": algorithm_cfg,
        "dynamics_model": _create_analytical_dynamics_config(),
        "overrides": {
            "use_analytical_model": True,  # Enable analytical model
            "learned_rewards": True,  # Learn rewards from data
            "num_steps": _NUM_TRIALS_MBPO * _TRIAL_LEN,
            "epoch_length": _TRIAL_LEN,
            "freq_train_model": 5,
            "model_lr": 1e-3,
            "model_wd": 1e-5,
            "model_batch_size": 64,
            "validation_ratio": 0.1,
            "num_epochs_train_model": 10,
            "patience": 5,
            "effective_model_rollouts_per_step": 10,
            "rollout_schedule": [1, _NUM_TRIALS_MBPO, 5, 5],
            "num_sac_updates_per_step": 5,
            "num_epochs_to_retain_sac_buffer": 1,
            "num_elites": 3,
            "sac_updates_every_steps": 1,
            "sac_gamma": 0.99,
            "sac_tau": 0.005,
            "sac_alpha": 0.2,
            "sac_policy": "Gaussian",
            "sac_target_update_interval": 2,
            "sac_automatic_entropy_tuning": True,
            "sac_hidden_size": 64,  # Smaller for testing
            "sac_lr": 3e-4,
            "sac_batch_size": 64,
            "sac_target_entropy": -1.0,
        },
        "debug_mode": _DEBUG_MODE,
        "seed": SEED,
        "device": str(device),
        "log_frequency_agent": 50,
    }

    cfg = OmegaConf.create(cfg_dict)
    cfg.algorithm.initial_exploration_steps = _INITIAL_EXPLORE
    cfg.algorithm.dataset_size = _TRIAL_LEN * _NUM_TRIALS_MBPO + _INITIAL_EXPLORE

    if vectorized:
        env = MockVecEnv()
        test_env = MockVecEnv()
    else:
        env = MockLineEnv()
        test_env = MockLineEnv()

    term_fn = mbrl_env.termination_fns.no_termination

    max_reward = mbpo.train(
        env, test_env, term_fn, cfg, silent=_SILENT, work_dir=_DIR.name
    )

    # Should achieve good performance with perfect analytical dynamics
    assert max_reward > _TARGET_REWARD
    return max_reward


# Test functions following mbrl-lib naming conventions
def test_pets_analytical_model():
    """Test Pets with analytical model."""
    _check_pets_analytical_model(vectorized=False)


def test_pets_analytical_model_vectorized():
    """Test Pets with analytical model using vectorized environment."""
    _check_pets_analytical_model(vectorized=True)


def test_mbpo_analytical_model():
    """Test MBPO with analytical model."""
    _check_mbpo_analytical_model(vectorized=False)


def test_mbpo_analytical_model_vectorized():
    """Test MBPO with analytical model using vectorized environment."""
    _check_mbpo_analytical_model(vectorized=True)