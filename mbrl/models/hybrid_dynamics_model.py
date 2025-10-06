"""
Hybrid Dynamics Model for Kinova Manipulation Tasks

Combines analytical robot kinematics with learned object dynamics.
The core idea is to leverage what we know (robot kinematics) and only learn what we don't know (object dynamics).

For Kinova manipulation tasks:
- Robot state: position, velocity, gripper state (known dynamics via analytical model)
- Object state: position, orientation, velocity (learned dynamics via neural network)
- Contact/interaction effects: learned implicitly through object dynamics network

This provides better sample efficiency than full learned models while maintaining accuracy
for tasks involving object interaction like pick-and-place and pushing.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Callable, Any
from mbrl.models.model import Model
from mbrl.models.gaussian_mlp import GaussianMLP


class SceneDynamicNetwork(nn.Module):
    """
    Gaussian MLP ensemble that learns scene dynamics given robot state and action.

    Input: [robot_state, action, scene_dynamic]
    Output: [delta_scene_dynamic] or [next_scene_dynamic] with uncertainty
    """

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 obs_dim: int,
                 robot_state_dim: int,
                 ensemble_size: int = 5,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 predict_delta: bool = True,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 propagation_method: Optional[str] = None):
        super().__init__()

        self.predict_delta = predict_delta
        self.device = device
        self.dtype = dtype

        self.in_size = in_size
        self.out_size = out_size
        self.obs_dim = obs_dim
        self.robot_state_dim = robot_state_dim
        self.scene_dynamic_dim = obs_dim - robot_state_dim

        # Use GaussianMLP ensemble for uncertainty quantification
        self.ensemble = GaussianMLP(
            in_size=in_size,
            out_size=out_size,
            device=device,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hidden_size,
            deterministic=False,  # Enable uncertainty prediction
            propagation_method=propagation_method
        ).to(dtype=dtype)

    def forward(self, model_in: torch.Tensor, use_propagation: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict next scene dynamic state with uncertainty.

        Args:
            model_in: Input tensor [batch_size, obs_dim + action_dim] = [robot_state, scene_dynamic, action]
            use_propagation: Whether to use ensemble propagation

        Returns:
            next_scene_dynamic: Next scene dynamic state [batch_size, scene_dynamic_dim]
            logvar: Log variance of prediction [batch_size, scene_dynamic_dim] (if not using propagation)
        """
        # Extract components from model_in format [robot_state, scene_dynamic, action]
        # model_in = [obs, action] where obs = [robot_state, scene_dynamic]
        obs = model_in[:, :self.obs_dim]
        action = model_in[:, self.obs_dim:]

        robot_state = obs[:, :self.robot_state_dim]
        scene_dynamic = obs[:, self.robot_state_dim:]

        # Rearrange to [robot_state, action, scene_dynamic] for ensemble
        x = torch.cat([robot_state, action, scene_dynamic], dim=-1)
        pred_mean, pred_logvar = self.ensemble.forward(x, use_propagation=use_propagation)

        if self.predict_delta:
            if use_propagation or pred_mean.ndim == 2:
                next_scene_dynamic = scene_dynamic + pred_mean
            else:
                # Handle ensemble dimension when not using propagation
                next_scene_dynamic = scene_dynamic.unsqueeze(0) + pred_mean
            return next_scene_dynamic, pred_logvar
        else:
            return pred_mean, pred_logvar

    def sample_prediction(self, robot_state: torch.Tensor, action: torch.Tensor,
                         scene_dynamic: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Sample from the predictive distribution.

        Args:
            robot_state: Current robot state [batch_size, robot_state_dim]
            action: Robot action [batch_size, action_dim]
            scene_dynamic: Current scene dynamic state [batch_size, scene_dynamic_dim]
            rng: Random number generator

        Returns:
            sampled_next_scene_dynamic: Sampled next scene dynamic state [batch_size, scene_dynamic_dim]
        """
        pred_mean, pred_logvar = self.forward(robot_state, action, scene_dynamic, use_propagation=True)

        if pred_logvar is not None:
            # Sample from Gaussian distribution
            std = torch.exp(0.5 * pred_logvar)
            if rng is not None:
                eps = torch.randn_like(std, generator=rng)
            else:
                eps = torch.randn_like(std)
            return pred_mean + eps * std
        else:
            return pred_mean


class HybridDynamicsModel(Model):
    """
    Hybrid dynamics model that combines analytical robot kinematics with learned scene dynamics.

    Architecture:
    - Robot dynamics: Computed analytically using kinematic model (no learning needed)
    - Scene dynamics: Learned via Gaussian MLP ensemble based on robot-scene interaction
    - Rewards: Can be learned or provided analytically

    This approach leverages known physics for the robot while learning the complex
    scene interaction dynamics that are difficult to model analytically.
    """

    def __init__(self,
                 dynamics_model: Any,  # KinovaKinematicModel or similar
                 obs_dim: int,
                 action_dim: int,
                 in_size: int,
                 out_size: int,
                 reward_fn: Optional[Callable] = None,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 learned_rewards: bool = True,
                 scene_net_ensemble_size: int = 5,
                 scene_net_hidden_size: int = 256,
                 scene_net_num_layers: int = 4,
                 predict_scene_delta: bool = True,
                 scene_net_propagation_method: Optional[str] = None):
        """
        Args:
            dynamic_model: Analytical model for robot kinematics (e.g., KinovaKinematicModel)
            obs_dim: Full observation dimension
            action_dim: Action dimension
            robot_state_dim: Dimension of robot state portion in observation
            in_size: Input size for training (obs_dim + action_dim)
            out_size: Output size for training (obs_dim + reward_dim if learning rewards)
            reward_fn: Function for computing rewards if not learning them
            device: torch device
            dtype: torch dtype
            learned_rewards: Whether to learn rewards from data
            scene_net_ensemble_size: Number of models in scene dynamics ensemble
            scene_net_hidden_size: Hidden layer size for scene dynamics network
            scene_net_num_layers: Number of layers in scene dynamics network
            predict_scene_delta: Whether scene network predicts delta or absolute state
            scene_net_propagation_method: Uncertainty propagation method for ensemble
        """
        super().__init__(device)

        self.dynamics_model = dynamics_model
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.robot_state_dim = self.dynamics_model.state_dim  # Get from kinematic model
        self.scene_dynamic_dim = obs_dim - self.robot_state_dim  # Calculate automatically
        self.in_size = in_size
        self.out_size = out_size
        self.reward_fn = reward_fn
        self.device = device
        self.dtype = dtype
        self.learned_rewards = learned_rewards

        # Validate dimensions
        assert self.robot_state_dim <= obs_dim, \
            f"robot_state_dim ({self.robot_state_dim}) must be <= obs_dim ({obs_dim})"
        assert self.scene_dynamic_dim > 0, \
            f"scene_dynamic_dim ({self.scene_dynamic_dim}) must be > 0"

        # Scene dynamics network - only this part needs learning
        self.scene_dynamics = SceneDynamicNetwork(
            in_size=self.in_size,
            out_size=self.scene_dynamic_dim,
            obs_dim=self.obs_dim,
            robot_state_dim=self.robot_state_dim,
            ensemble_size=scene_net_ensemble_size,
            hidden_size=scene_net_hidden_size,
            num_layers=scene_net_num_layers,
            predict_delta=predict_scene_delta,
            device=device,
            dtype=dtype,
            propagation_method=scene_net_propagation_method
        )

        # Reward network if learning rewards
        if self.learned_rewards:
            self.reward_net = nn.Sequential(
                nn.Linear(self.in_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(device=device, dtype=dtype)

        # Set conversion functions from kinematic model if available
        if hasattr(dynamics_model, 'obs2state'):
            self.obs2state_fn = dynamics_model.obs2state
        else:
            self.obs2state_fn = lambda x: x  # Identity fallback

        if hasattr(dynamics_model, 'state2obs'):
            self.state2obs_fn = dynamics_model.state2obs
        else:
            self.state2obs_fn = lambda x: x  # Identity fallback

    def _split_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split observation into robot state and scene dynamic state.

        Args:
            obs: Full observation [batch_size, obs_dim]

        Returns:
            robot_state: [batch_size, robot_state_dim]
            scene_dynamic: [batch_size, scene_dynamic_dim]
        """
        robot_state = obs[:, :self.robot_state_dim]
        scene_dynamic = obs[:, self.robot_state_dim:]

        return robot_state, scene_dynamic

    def _combine_states(self, robot_state: torch.Tensor, scene_dynamic: torch.Tensor) -> torch.Tensor:
        """
        Combine robot state and scene dynamic state back into full observation.

        Args:
            robot_state: [batch_size, robot_state_dim]
            scene_dynamic: [batch_size, scene_dynamic_dim]

        Returns:
            obs: [batch_size, obs_dim]
        """
        return torch.cat([robot_state, scene_dynamic], dim=-1)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for compatibility with mbrl-lib training interface.
        Robot dynamics computed analytically, object dynamics learned, rewards optional.
        """
        # Split input into observation and action
        obs = x[:, :self.obs_dim]
        action = x[:, self.obs_dim:]

        # Get next observation using hybrid approach
        with torch.no_grad():
            next_obs = self.get_next_obs(obs, action)

        # Predict rewards if learning them
        if self.learned_rewards:
            rewards = self.reward_net(x)
            output = torch.cat([next_obs, rewards], dim=-1)
        else:
            output = next_obs

        return (output,)

    def sample_1d(self,
                  model_in: torch.Tensor,
                  model_state: Dict[str, torch.Tensor],
                  deterministic: bool = False,
                  rng: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Sample next states and rewards using the hybrid model.
        This is the key method used by ModelEnv during rollouts.
        """
        obs = model_in[:, :self.obs_dim]
        action = model_in[:, self.obs_dim:]

        # Use hybrid model to get next observation
        next_obs = self.get_next_obs(obs, action)

        # Compute rewards if needed
        if self.learned_rewards:
            rewards = self._compute_rewards(model_in)
            output = torch.cat([next_obs, rewards], dim=-1)
        else:
            output = next_obs

        # Create next model state
        next_model_state = {"obs": next_obs}

        return output, next_model_state

    def get_next_obs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next observation using hybrid approach:
        1. Use analytical model for robot kinematics
        2. Use learned model for scene dynamics
        3. Combine results

        Args:
            obs: Current observation [batch_size, obs_dim]
            action: Action [batch_size, action_dim]

        Returns:
            next_obs: Next observation [batch_size, obs_dim]
        """
        # Split observation into components
        robot_state, scene_dynamic = self._split_observation(obs)

        next_robot_state = self.dynamics_model.get_next_state(robot_state, action)

        # 2. Get next scene dynamic state using learned dynamics with uncertainty
        model_in_scene = torch.cat([obs, action], dim=-1)
        next_scene_dynamic, _ = self.scene_dynamics(model_in_scene, use_propagation=True)

        # 3. Combine into full next observation
        next_obs = self._combine_states(next_robot_state, next_scene_dynamic)

        return next_obs

    def _compute_rewards(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards using either learned reward network or provided reward function.
        """
        if self.learned_rewards and hasattr(self, 'reward_net'):
            rewards = self.reward_net(x)
        elif self.reward_fn is not None:
            rewards = self.reward_fn(x[:, :self.obs_dim], x[:, self.obs_dim:])
        else:
            rewards = torch.zeros((x.shape[0], 1), device=self.device)

        return rewards

    def loss(self, model_in: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for hybrid model:
        - Robot dynamics loss is always 0 (analytical)
        - Scene dynamics loss computed from learned ensemble
        - Reward loss computed if learning rewards
        """
        if target is None:
            return torch.tensor(0.0, device=self.device), {}

        # Target next observation (and possibly rewards)
        if self.learned_rewards:
            target_next_obs = target[:, :self.obs_dim]
            target_rewards = target[:, -1:]
        else:
            target_next_obs = target
            target_rewards = None

        # Extract target scene dynamics (only part we're learning)
        _, target_scene_dynamic = self._split_observation(target_next_obs)

        # Scene dynamics loss - only part we're learning
        # Use the ensemble's loss function which includes Gaussian NLL
        scene_loss, scene_loss_dict = self.scene_dynamics.ensemble.loss(model_in, target_scene_dynamic)

        loss_dict = {"scene_dynamics_loss": scene_loss.item()}
        loss_dict.update(scene_loss_dict)
        total_loss = scene_loss

        # Reward loss if learning rewards
        if self.learned_rewards and target_rewards is not None:
            pred_rewards = self.reward_net(model_in)
            reward_loss = nn.functional.mse_loss(pred_rewards, target_rewards)
            loss_dict["reward_loss"] = reward_loss.item()
            total_loss = total_loss + reward_loss

        return total_loss, loss_dict

    def reset_1d(self, obs: torch.Tensor, rng: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """Reset the model state."""
        return {}

    def save(self, save_dir):
        """Save learned components (scene dynamics and reward networks)."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Save scene dynamics ensemble
        self.scene_dynamics.ensemble.save(save_dir)

        # Save reward network if learning rewards
        if self.learned_rewards and hasattr(self, 'reward_net'):
            torch.save(self.reward_net.state_dict(),
                      os.path.join(save_dir, "reward_net.pth"))

    def load(self, load_dir):
        """Load learned components (scene dynamics and reward networks)."""
        import os

        # Load scene dynamics ensemble
        self.scene_dynamics.ensemble.load(load_dir)

        # Load reward network if learning rewards
        if self.learned_rewards and hasattr(self, 'reward_net'):
            reward_net_path = os.path.join(load_dir, "reward_net.pth")
            if os.path.exists(reward_net_path):
                self.reward_net.load_state_dict(torch.load(reward_net_path, map_location=self.device))

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor = None) -> Dict[str, float]:
        """Evaluation metrics for the hybrid model."""
        if target is None:
            return {}

        with torch.no_grad():
            loss_val, loss_dict = self.loss(model_in, target)

        # Add scene dynamics prediction accuracy
        obs = model_in[:, :self.obs_dim]
        action = model_in[:, self.obs_dim:]

        if self.learned_rewards:
            target_next_obs = target[:, :self.obs_dim]
        else:
            target_next_obs = target

        robot_state, scene_dynamic = self._split_observation(obs)
        _, target_scene_dynamic = self._split_observation(target_next_obs)

        with torch.no_grad():
            pred_next_scene_dynamic, _ = self.scene_dynamics(robot_state, action, scene_dynamic, use_propagation=True)
            scene_mse = nn.functional.mse_loss(pred_next_scene_dynamic, target_scene_dynamic)

        eval_dict = loss_dict.copy()
        eval_dict["scene_dynamics_mse"] = scene_mse.item()

        return eval_dict