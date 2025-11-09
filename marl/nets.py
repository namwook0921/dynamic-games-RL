# marl/nets.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCriticPPO(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU()
        )
        self.actor_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim)
        )
        self.critic_layer = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )

    def forward(self, x):
        x = self.net(x)
        action_probs = F.softmax(self.actor_layer(x), dim=-1)
        value = self.critic_layer(x)
        return action_probs, value
