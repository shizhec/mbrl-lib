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


class HybridResidualNetwork(nn.Module):
    """
    Gaussian MLP ensemble that learns both robot state residuals and scene dynamics.

    Input: [obs, action] where obs = [robot_state, scene_dynamic]
    Output: [robot_state_residual, scene_dynamic_delta] with uncertainty

    The robot_state_residual corrects analytical model predictions for unmodeled effects
    like friction, collisions, and environmental disturbances.
    """

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 obs_dim: int,
                 robot_state_dim: int,
                 ensemble_size: int = 5,
                 hidden_size: int = 256,
                 num_layers: int = 4,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 propagation_method: Optional[str] = None):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.in_size = in_size
        self.out_size = out_size
        self.obs_dim = obs_dim
        self.robot_state_dim = robot_state_dim
        self.scene_dynamic_dim = obs_dim - robot_state_dim

        # Output includes both robot residuals and scene deltas
        assert out_size == obs_dim, f"Output size must equal obs_dim ({obs_dim}), got {out_size}"

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
        Predict observation delta (robot residual + scene delta) with uncertainty.

        Args:
            model_in: Input tensor [batch_size, obs_dim + action_dim] = [obs, action]
            use_propagation: Whether to use ensemble propagation

        Returns:
            obs_delta: Full observation delta [batch_size, obs_dim]
                       [:robot_state_dim] = robot residual
                       [robot_state_dim:] = scene delta
            logvar: Log variance of prediction [batch_size, obs_dim] (if not using propagation)
        """
        # Forward pass through ensemble - input is [obs, action]
        obs_delta, pred_logvar = self.ensemble.forward(model_in, use_propagation=use_propagation)

        return obs_delta, pred_logvar



class HybridDynamicsModel(Model):
    """
    Hybrid dynamics model that combines analytical robot kinematics with learned residuals.

    Architecture:
    - Robot dynamics: Analytical kinematic model + learned residual corrections
    - Scene dynamics: Learned deltas via Gaussian MLP ensemble
    - Rewards: Can be learned or provided analytically

    The learned network predicts full observation deltas [robot_residual, scene_delta]:
    - robot_residual: Corrects analytical model for friction, collisions, disturbances
    - scene_delta: Predicts scene/object dynamics from robot-scene interactions

    This approach leverages physics priors while learning corrections for unmodeled effects.
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
                 scene_net_propagation_method: Optional[str] = None):
        """
        Args:
            dynamic_model: Analytical model for robot kinematics (e.g., KinovaKinematicModel)
            obs_dim: Full observation dimension
            action_dim: Action dimension
            in_size: Input size for training (obs_dim + action_dim)
            out_size: Output size for training (obs_dim + reward_dim if learning rewards)
            reward_fn: Function for computing rewards if not learning them
            device: torch device
            dtype: torch dtype
            learned_rewards: Whether to learn rewards from data
            scene_net_ensemble_size: Number of models in residual network ensemble
            scene_net_hidden_size: Hidden layer size for residual network
            scene_net_num_layers: Number of layers in residual network
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

        # Hybrid residual network - learns both robot residuals and scene deltas
        self.residual_network = HybridResidualNetwork(
            in_size=self.in_size,
            out_size=self.obs_dim,  # Full observation delta
            obs_dim=self.obs_dim,
            robot_state_dim=self.robot_state_dim,
            ensemble_size=scene_net_ensemble_size,
            hidden_size=scene_net_hidden_size,
            num_layers=scene_net_num_layers,
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


    def _split_observation(self, obs: torch.Tensor, with_eef_pos: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split observation into robot state and scene dynamic state.

        Args:
            obs: Full observation [batch_size, obs_dim]

        Returns:
            robot_state: [batch_size, robot_state_dim]
            scene_dynamic: [batch_size, scene_dynamic_dim]
        """
        if with_eef_pos:
            robot_state = obs[:, :self.robot_state_dim + 3]
            scene_dynamic = obs[:, self.robot_state_dim + 3:]
        else:
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
        Predict next observation using hybrid residual approach:
        1. Use rollout_open_loop to get analytical robot state [pos, vel, eef_pos]
        2. Combine with current scene to form analytical next_obs
        3. Use learned network to predict full obs_delta
        4. Apply: next_obs = next_obs_analytical + obs_delta

        Args:
            obs: Current observation [batch_size, obs_dim]
            action: Action [batch_size, action_dim]

        Returns:
            next_obs: Next observation [batch_size, obs_dim]
        """
        # Split observation into components
        robot_state, scene_dynamic = self._split_observation(obs)

        # 1. Use rollout_open_loop to get next robot state [pos, vel, eef_pos]
        action_seq = action.unsqueeze(1)  # Add time dimension: [batch_size, 1, action_dim]
        state_dict = self.dynamics_model.rollout_open_loop(robot_state, action_seq)

        # Extract next state: [pos, vel] from state_seq[:, 0, :]
        next_joint_state = state_dict['state_seq'][:, 0, :]  # [batch_size, 2*n_dofs]
        # Extract eef_pos from ee_pos_seq[:, 0, :]
        next_eef_pos = state_dict['ee_pos_seq'][:, 0, :]  # [batch_size, 3]

        # Combine to form full robot state [pos, vel, eef_pos]
        next_robot_state_analytical = torch.cat([next_joint_state, next_eef_pos], dim=-1)

        # 2. Form analytical next observation [robot_state, scene_dynamic]
        next_obs_analytical = self._combine_states(next_robot_state_analytical, scene_dynamic)

        # 3. Get learned observation delta
        model_in = torch.cat([obs, action], dim=-1)
        obs_delta, _ = self.residual_network(model_in, use_propagation=True)

        # 4. Apply delta to analytical prediction
        next_obs = next_obs_analytical + obs_delta

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
        Compute loss for hybrid residual model:
        - Observation delta loss: trains network to predict obs_delta
        - obs_delta = target_next_obs - analytical_next_obs
        - Analytical prediction uses rollout_open_loop for full robot state
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

        # Extract current observation and action
        obs = model_in[:, :self.obs_dim]
        action = model_in[:, self.obs_dim:]
        robot_state, scene_dynamic = self._split_observation(obs)

        # Compute analytical next observation using rollout_open_loop
        action_seq = action.unsqueeze(1)  # [batch_size, 1, action_dim]
        state_dict = self.dynamics_model.rollout_open_loop(robot_state, action_seq)

        # Extract and combine to form analytical robot state [pos, vel, eef_pos]
        next_joint_state = state_dict['state_seq'][:, 0, :]  # [batch_size, 2*n_dofs]
        next_eef_pos = state_dict['ee_pos_seq'][:, 0, :]  # [batch_size, 3]
        analytical_next_robot = torch.cat([next_joint_state, next_eef_pos], dim=-1)

        # Form analytical next observation
        analytical_next_obs = self._combine_states(analytical_next_robot, scene_dynamic)

        # Compute target observation delta
        # obs_delta = target_next_obs - analytical_next_obs
        target_obs_delta = target_next_obs - analytical_next_obs

        # Compute loss using ensemble's Gaussian NLL
        obs_delta_loss, loss_dict = self.residual_network.ensemble.loss(model_in, target_obs_delta)

        loss_dict["obs_delta_loss"] = obs_delta_loss.item()
        total_loss = obs_delta_loss

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

    def set_elite(self, elite_indices: list):
        """Set elite ensemble members. Delegates to the residual network's ensemble."""
        self.residual_network.ensemble.set_elite(elite_indices)

    def save(self, save_dir):
        """Save learned components (residual network and reward networks)."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Save residual network ensemble
        self.residual_network.ensemble.save(save_dir)

        # Save reward network if learning rewards
        if self.learned_rewards and hasattr(self, 'reward_net'):
            torch.save(self.reward_net.state_dict(),
                      os.path.join(save_dir, "reward_net.pth"))

    def load(self, load_dir):
        """Load learned components (residual network and reward networks)."""
        import os

        # Load residual network ensemble
        self.residual_network.ensemble.load(load_dir)

        # Load reward network if learning rewards
        if self.learned_rewards and hasattr(self, 'reward_net'):
            reward_net_path = os.path.join(load_dir, "reward_net.pth")
            if os.path.exists(reward_net_path):
                self.reward_net.load_state_dict(torch.load(reward_net_path, map_location=self.device))

    def eval_score(self, model_in: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Evaluation metrics for the hybrid residual model.

        Returns:
            score: Per-sample, per-ensemble-member squared errors [num_members, batch_size, obs_dim]
            meta: Dictionary of additional metrics
        """
        if target is None:
            num_members = self.residual_network.ensemble.num_members
            batch_size = model_in.shape[0]
            return torch.zeros(num_members, batch_size, self.obs_dim, device=self.device), {}

        # Get target next observation
        if self.learned_rewards:
            target_next_obs = target[:, :self.obs_dim]
        else:
            target_next_obs = target

        # Compute analytical next observation (same as in loss function)
        obs = model_in[:, :self.obs_dim]
        action = model_in[:, self.obs_dim:]
        robot_state, scene_dynamic = self._split_observation(obs)

        # Get analytical prediction using rollout_open_loop
        action_seq = action.unsqueeze(1)
        state_dict = self.dynamics_model.rollout_open_loop(robot_state, action_seq)
        next_joint_state = state_dict['state_seq'][:, 0, :]
        next_eef_pos = state_dict['ee_pos_seq'][:, 0, :]
        analytical_next_robot = torch.cat([next_joint_state, next_eef_pos], dim=-1)
        analytical_next_obs = self._combine_states(analytical_next_robot, scene_dynamic)

        # Target delta for the residual network
        target_obs_delta = target_next_obs - analytical_next_obs

        # Use the ensemble's eval_score to get per-member squared errors
        with torch.no_grad():
            # This returns [num_members, batch_size, obs_dim] with reduction="none"
            squared_errors, _ = self.residual_network.ensemble.eval_score(model_in, target_obs_delta)

            # Compute additional metadata using single prediction (with propagation)
            pred_next_obs = self.get_next_obs(obs, action)
            obs_mse = nn.functional.mse_loss(pred_next_obs, target_next_obs)

            pred_robot, pred_scene = self._split_observation(pred_next_obs)
            target_robot, target_scene = self._split_observation(target_next_obs)

            robot_mse = nn.functional.mse_loss(pred_robot, target_robot)
            scene_mse = nn.functional.mse_loss(pred_scene, target_scene)

        eval_dict = {
            "obs_mse": obs_mse.item(),
            "robot_mse": robot_mse.item(),
            "scene_mse": scene_mse.item()
        }

        return squared_errors, eval_dict