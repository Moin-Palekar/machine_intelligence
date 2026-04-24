import numpy as np
import torch
import torch.nn as nn


class Packet(nn.Module):
    """
    A single prediction primitive.
    Takes current state, predicts next state.
    Learns by minimizing surprise (prediction error).
    Frozen after K episodes.
    """

    def __init__(self, input_dim, hidden_dim=96,
                 output_dim=None, max_episodes=100):
        super().__init__()
        output_dim = output_dim or input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.episodes_lived = 0
        self.max_episodes = max_episodes
        self.frozen = False
        self.activation = None

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3
        )

    def forward(self, x):
        self.activation = self.net(x)
        return self.activation

    def freeze(self):
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False
        print(f"Packet frozen after {self.episodes_lived} episodes")

    def end_episode(self):
        if not self.frozen:
            self.episodes_lived += 1
            if self.episodes_lived >= self.max_episodes:
                self.freeze()

    def update(self, obs, actual):
        """
        Recompute forward fresh each update step.
        Avoids inplace gradient modification errors.
        """
        if self.frozen:
            return

        self.optimizer.zero_grad()

        # fresh forward pass — clean computation graph
        predicted = self.net(obs)
        diff = predicted - actual
        loss = torch.mean(diff * diff)

        loss.backward()
        self.optimizer.step()

        return loss.item()