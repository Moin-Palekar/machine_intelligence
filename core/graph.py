import numpy as np
import torch
import torch.nn as nn
from core.packet import Packet


class Connection:
    """
    Hebbian connection between two packets.
    Always live. Never frozen.
    """
    def __init__(self, source_idx, target_idx):
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.weight = 0.1

    def hebbian_update(self, pre_activation,
                       post_activation, lr=0.001):
        pre  = pre_activation.detach().mean().item()
        post = post_activation.detach().mean().item()
        self.weight += lr * pre * post
        self.weight  = np.clip(self.weight, -1.0, 1.0)

    def decay(self, rate=0.0001):
        self.weight *= (1.0 - rate)


class PacketGraph:
    """
    Growing graph of prediction packets.
    Starts with one packet.
    Grows when surprise plateaus.
    """

    def __init__(self, input_dim, action_dim,
                 hidden_dim=96, max_episodes=100):
        self.input_dim     = input_dim
        self.action_dim    = action_dim
        self.hidden_dim    = hidden_dim
        self.max_episodes  = max_episodes
        self.full_dim      = input_dim + action_dim

        self.packets = nn.ModuleList([
            Packet(
                input_dim    = input_dim,
                hidden_dim   = hidden_dim,
                output_dim   = self.full_dim,
                max_episodes = max_episodes
            )
        ])

        self.connections       = []
        self.surprise_history  = []
        self.plateau_window    = 20
        self.plateau_threshold = 0.01

    def forward(self, x):
        out = x
        activations = []

        for packet in self.packets:
            out = packet(out)
            activations.append(out)
            if out.shape[-1] != self.input_dim:
                out = out[..., :self.input_dim]

        for conn in self.connections:
            conn.hebbian_update(
                activations[conn.source_idx],
                activations[conn.target_idx]
            )
            conn.decay()

        return activations[-1]

    def update(self, obs, actual):
        """
        Update all plastic packets with fresh forward pass.
        obs   — previous proprioception tensor
        actual — actual next state tensor (prop + action)
        """
        for packet in self.packets:
            if not packet.frozen:
                packet.update(obs, actual)

    def compute_surprise(self, obs, actual):
        """
        Compute surprise without gradient for logging.
        """
        with torch.no_grad():
            predicted = self.packets[0].net(obs)
            diff = predicted - actual
            return torch.mean(diff * diff).item()

    def end_episode(self):
        for packet in self.packets:
            packet.end_episode()

    def record_surprise(self, surprise_val):
        self.surprise_history.append(surprise_val)

    def check_plateau(self):
        if len(self.surprise_history) < self.plateau_window:
            return False
        recent = self.surprise_history[-self.plateau_window:]
        early  = np.mean(recent[:self.plateau_window // 2])
        late   = np.mean(recent[self.plateau_window // 2:])
        return (early - late) < self.plateau_threshold

    def grow(self):
        new_idx = len(self.packets)
        new_packet = Packet(
            input_dim    = self.input_dim,
            hidden_dim   = self.hidden_dim,
            output_dim   = self.full_dim,
            max_episodes = self.max_episodes
        )
        self.packets.append(new_packet)
        conn = Connection(
            source_idx = new_idx - 1,
            target_idx = new_idx
        )
        self.connections.append(conn)
        print(f"Graph grew — now {len(self.packets)} packets")