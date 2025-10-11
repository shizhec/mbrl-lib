"""
Analytical Model Wrapper for mbrl-lib integration
Allows MBPO and Pets to use analytical dynamics models instead of learned models
"""
import torch
from typing import Dict, Optional, Tuple, Callable, Any

from mbrl.models.model import Model

class AnalyticalModel(Model):
    """
    Wrapper that makes analytical dynamics models compatible with mbrl-lib interface.
    This allows MBPO and Pets to use analytical models instead of learned models.
    For analytical models:
    - Dynamics are computed analytically (no training needed)
    - Rewards can be learned from data if learned_rewards=True
    """

    def __init__(self,
                 dynamics_model: Any,
                 obs_dim: int,
                 action_dim: int,
                 in_size: int,
                 out_size: int,
                 reward_fn: Optional[Callable] = None,
                 device: str = "cpu",
                 learned_rewards: bool = True):
        """
        Args:
            dynamics_model: Your analytical dynamics model (e.g., KinovaKinematicModel)
            reward_fn: Function that computes rewards given (state, action, next_state)
                      Only used if learned_rewards=False
            device: torch device
            learned_rewards: Whether to learn rewards from data (True) or use reward_fn (False)
        """
        super().__init__(device)
        self.dynamics_model = dynamics_model
        self.reward_fn = reward_fn
        self.learned_rewards = learned_rewards
        self.device = device

        # RL Env dimensions (will be set later via set_obs_action_dims)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.in_size = in_size
        self.out_size = out_size

        # Reward predictor network (will be created when dimensions are set)
         # Rebuild reward network with correct input size if learning rewards
        if self.learned_rewards:
            # Get dtype from dynamics model if available
            dtype = getattr(dynamics_model, 'dtype', torch.float32)
            self.reward_net = torch.nn.Sequential(
                torch.nn.Linear(self.in_size, 128),  # state + action
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            ).to(device=self.device, dtype=dtype)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for compatibility with mbrl-lib training interface.
        For analytical models: dynamics are computed analytically, rewards may be predicted.
        """
        # Split input into state and action
        obs = x[:, :self.obs_dim]
        action = x[:, self.obs_dim:]

        # Get next state analytically (no gradients needed for dynamics)
        with torch.no_grad():
            next_obs = self._get_next_obs_analytical(obs, action)

        # Predict rewards if learning rewards
        if self.learned_rewards:
            # Reward network takes state + action + next_state as input
            rewards = self.reward_net(x)
            output = torch.cat([next_obs, rewards], dim=-1)
        else:
            output = next_obs

        return (output,)  # Return as tuple for compatibility

    def sample_1d(self,
                  model_in: torch.Tensor,
                  model_state: Dict[str, torch.Tensor],
                  deterministic: bool = False,
                  rng: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Sample next states and rewards using the analytical model.
        This is the key method used by ModelEnv during rollouts.
        """
        obs = model_in[:, :self.obs_dim]
        action = model_in[:, self.obs_dim:]

        # Use your analytical model to get next state
        next_obs = self._get_next_obs_analytical(obs, action)

        # Compute rewards if needed
        if self.learned_rewards:
            rewards = self._compute_rewards(model_in)
            # Concatenate next observation and rewards as expected by OneDTransitionRewardModel
            output = torch.cat([next_obs, rewards], dim=-1)
        else:
            output = next_obs

        # Create next model state
        next_model_state = {"obs": next_obs}

        return output, next_model_state

    def reset_1d(self, obs: torch.Tensor, rng: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        """
        Reset the model state. For analytical models, this just returns the observation.
        """
        return {}  # No internal state needed for analytical models

    def _get_next_obs_analytical(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Use your analytical dynamics model to predict next state.
        Adapts your model's interface to mbrl-lib's expected format.
        """
        with torch.no_grad():
            next_obs = self.dynamics_model.get_next_obs(obs, action)

        return next_obs

    def _compute_rewards(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards using either learned reward network or provided reward function.
        """
        if self.learned_rewards and hasattr(self, 'reward_net'):
            # Use learned reward network
            rewards = self.reward_net(x)
        elif self.reward_fn is not None:
            # Use provided reward function
            rewards = self.reward_fn(x[:, :self.obs_dim], x[:, self.obs_dim:])
        else:
            # No reward function provided, return zeros
            rewards = torch.zeros((x.shape[0], 1), device=self.device)

        return rewards

    # Required methods for mbrl-lib compatibility
    def loss(self, model_in, target=None):
        """
        Compute loss for analytical models:
        - Dynamics loss is always 0 (analytical)
        - Reward loss is computed if learning rewards
        """
        if not self.learned_rewards:
            return torch.tensor(0.0, device=self.device), {}

        # Predict rewards
        pred_rewards = self.reward_net(model_in)

        # Extract target rewards (last column of target)
        target_rewards = target[:, -1:] if target is not None else None

        if target_rewards is not None:
            reward_loss = torch.nn.functional.mse_loss(pred_rewards, target_rewards)
            return reward_loss, {"reward_loss": reward_loss.item()}
        else:
            return torch.tensor(0.0, device=self.device), {}

    def save(self, save_dir):
        """
        Save analytical model.
        - For pure analytical models (learned_rewards=False): nothing to save
        - For hybrid models (learned_rewards=True): save the learned reward network
        """
        if self.learned_rewards and hasattr(self, 'reward_net'):
            import pathlib
            save_path = pathlib.Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                self.reward_net.state_dict(),
                save_path / "analytical_model_reward_net.pth"
            )

    def load(self, load_dir):
        """
        Load analytical model.
        - For pure analytical models (learned_rewards=False): nothing to load
        - For hybrid models (learned_rewards=True): load the learned reward network
        """
        if self.learned_rewards and hasattr(self, 'reward_net'):
            import pathlib
            load_path = pathlib.Path(load_dir) / "analytical_model_reward_net.pth"
            if load_path.exists():
                self.reward_net.load_state_dict(torch.load(load_path))
            else:
                print(f"Warning: No saved reward network found at {load_path}")

    def eval_score(self, model_in, target = None):
        """
        Compute evaluation score for analytical models.
        - Dynamics: always perfect (analytical), so score is 0
        - Rewards: if learned, compute MSE between predicted and target rewards
        """
        if not self.learned_rewards or target is None:
            return torch.tensor(0.0, device=self.device), {}

        with torch.no_grad():
            # Predict rewards
            pred_rewards = self.reward_net(model_in)

            # Extract target rewards (last column of target)
            target_rewards = target[:, -1:]

            # Compute MSE for evaluation
            reward_mse = torch.nn.functional.mse_loss(pred_rewards, target_rewards, reduction='none')

            return reward_mse, {"reward_mse": reward_mse.mean().item()}
