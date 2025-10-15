# marl/nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticPPO(nn.Module):
    """
    Actor–Critic network for PPO-based multi-agent reinforcement learning.

    Architecture:
        Shared base feature extractor (first hidden layer)
        ├── Actor head -> action logits (Softmax)
        └── Critic head -> scalar state value

    Args:
        input_dim (int): Dimension of flattened observation.
        output_dim (int): Number of discrete actions.
        hidden_dims (tuple[int, int]): Sizes of hidden layers (default: (64, 64))
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(64, 64)):
        super().__init__()

        # Shared base (first hidden layer)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

        # Optional initialization: small weights for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """Forward pass returning (action_probs, value)."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        features = self.shared(x)
        logits = self.actor(features)
        action_probs = F.softmax(logits, dim=-1)
        value = self.critic(features)
        return action_probs, value
