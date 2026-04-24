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
        self.weight = 0.1  # start weak

    def hebbian_update(self, pre_activation, post_activation, lr=0.001):
        """always updating — free operation"""
        pre  = pre_activation.detach().mean().item()
        post = post_activation.detach().mean().item()
        self.weight += lr * pre * post
        self.weight  = np.clip(self.weight, -1.0, 1.0)

    def decay(self, rate=0.0001):
        """unused connections decay toward zero"""
        self.weight *= (1.0 - rate)


class PacketGraph:
    """
    Growing graph of prediction packets.
    Starts with one packet.
    Grows when surprise plateaus.
    """

    def __init__(self, input_dim, action_dim, hidden_dim=96, max_episodes=100):
        self.input_dim  = input_dim           # proprioception dim
        self.action_dim = action_dim          # joint torques dim
        self.hidden_dim = hidden_dim
        self.max_episodes = max_episodes

        # full prediction target = proprioception + action
        self.full_dim = input_dim + action_dim

        # start with one packet
        # predicts full next state (proprioception + action)
        self.packets = nn.ModuleList([
            Packet(
                input_dim  = input_dim,
                hidden_dim = hidden_dim,
                output_dim = self.full_dim,
                max_episodes = max_episodes
            )
        ])

        # connections between packets (Hebbian)
        self.connections = []

        # surprise history for plateau detection
        self.surprise_history = []
        self.plateau_window   = 20   # episodes to check
        self.plateau_threshold = 0.01  # min improvement to not be plateau

    def forward(self, x):
        """
        forward pass through graph
        returns predicted next state (proprioception + action)
        """
        # for now — serial chain through packets
        out = x
        activations = []

        for packet in self.packets:
            out = packet(out)
            activations.append(out)

            # resize for next packet if needed
            if out.shape[-1] != self.packets[0].input_dim:
                # take only proprioception part as input to next packet
                out = out[..., :self.input_dim]

        # hebbian update between adjacent packets
        for conn in self.connections:
            conn.hebbian_update(
                activations[conn.source_idx],
                activations[conn.target_idx]
            )
            conn.decay()

        # final output = last packet's prediction
        return activations[-1]

    def compute_surprise(self, predicted, actual):
        """
        surprise = mean squared difference
        between predicted next state and actual next state
        """
        return torch.mean((predicted - actual) ** 2)

    def update(self, surprise_val):
        """update all plastic packets"""
        for packet in self.packets:
            if not packet.frozen:
                packet.update(surprise_val)

    def end_episode(self):
        """end of episode housekeeping"""
        for packet in self.packets:
            packet.end_episode()

    def record_surprise(self, surprise_val):
        """track surprise history for plateau detection"""
        self.surprise_history.append(surprise_val)

    def check_plateau(self):
        """
        returns True if surprise has stopped reducing
        time to grow a new packet
        """
        if len(self.surprise_history) < self.plateau_window:
            return False

        recent   = self.surprise_history[-self.plateau_window:]
        early    = np.mean(recent[:self.plateau_window//2])
        late     = np.mean(recent[self.plateau_window//2:])
        improvement = early - late

        return improvement < self.plateau_threshold

    def grow(self):
        """
        add a new packet to the graph
        connects to last packet
        starts plastic
        """
        last_packet = self.packets[-1]
        new_idx = len(self.packets)

        new_packet = Packet(
            input_dim    = self.input_dim,
            hidden_dim   = self.hidden_dim,
            output_dim   = self.full_dim,
            max_episodes = self.max_episodes
        )

        self.packets.append(new_packet)

        # add Hebbian connection from previous to new
        conn = Connection(
            source_idx = new_idx - 1,
            target_idx = new_idx
        )
        self.connections.append(conn)

        print(f"Graph grew — now {len(self.packets)} packets")