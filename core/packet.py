import numpy as np
import torch
import torch.nn as nn


# per-packet thresholds
PRESSURE_GROWTH_THRESHOLD = 0.6    # pressure above this → signal to grow
SURPRISE_FREEZE_THRESHOLD = 0.001  # loss below this consistently → freeze
FREEZE_WINDOW             = 20     # steps loss must stay below threshold


class Packet(nn.Module):
    """
    A single prediction primitive.
    Takes current state, predicts next state.
    Learns by minimizing surprise (prediction error).

    Freezes when its OWN per-step loss stays below
    SURPRISE_FREEZE_THRESHOLD for FREEZE_WINDOW consecutive steps.
    Not age-based — saturation-based.

    Signals need_growth=True when pressure exceeds
    PRESSURE_GROWTH_THRESHOLD — graph should insert a new packet here.
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

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.episodes_lived  = 0
        self.frozen          = False
        self.activation      = None
        self.need_growth     = False   # set True when pressure too high

        # pressure — mean activation magnitude over recent steps
        self.pressure_history = []

        # per-step loss history for saturation detection
        self.loss_history     = []
        self._low_loss_streak = 0      # consecutive steps below threshold

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=1e-3
        )

    def forward(self, x):
        self.activation = self.net(x)

        # track pressure
        p = self.activation.detach().abs().mean().item()
        self.pressure_history.append(p)
        if len(self.pressure_history) > 500:
            self.pressure_history.pop(0)

        # signal growth if pressure too high
        self.need_growth = (self.get_pressure() > PRESSURE_GROWTH_THRESHOLD)

        return self.activation

    def get_pressure(self):
        if not self.pressure_history:
            return 0.0
        return float(np.mean(self.pressure_history[-100:]))

    def freeze(self):
        self.frozen = True
        for param in self.parameters():
            param.requires_grad = False
        print(f"  ❄ packet frozen (saturation) after "
              f"{self.episodes_lived} episodes")

    def end_episode(self):
        if not self.frozen:
            self.episodes_lived += 1

    def update(self, obs, actual, injected_surprise=None):
        """
        Update packet weights.

        injected_surprise:
          None       — normal MSE loss
          float > 1  — HIGH: loss = MSE + injected (large gradient)
          float == 0 — LOW:  loss = 0 (suppress gradient entirely)
        """
        if self.frozen:
            return None

        self.optimizer.zero_grad()
        predicted = self.net(obs)
        diff      = predicted - actual
        mse       = torch.mean(diff * diff)

        if injected_surprise is None:
            loss = mse
        elif injected_surprise > 1.0:
            loss = mse + injected_surprise
        else:
            loss = mse * max(injected_surprise, 0.0)

        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        if len(self.loss_history) > 500:
            self.loss_history.pop(0)

        # saturation check — only count natural MSE steps, not injected
        # injected low surprise forces loss=0 which is not real saturation
        if injected_surprise is None:
            if loss_val < SURPRISE_FREEZE_THRESHOLD:
                self._low_loss_streak += 1
            else:
                self._low_loss_streak = 0
            if self._low_loss_streak >= FREEZE_WINDOW:
                self.freeze()

        return loss_val

    def get_mean_loss(self):
        if not self.loss_history:
            return 0.0
        return float(np.mean(self.loss_history[-50:]))