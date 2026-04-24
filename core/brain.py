import torch
import numpy as np
from core.graph import PacketGraph


class Brain:
    """
    Full system.
    Always live. Always learning.
    No training/inference distinction.
    Single objective: minimize surprise.

    When external_action is provided:
      - brain observes and learns only
      - does NOT interfere with action

    When external_action is None:
      - brain controls itself
    """

    def __init__(self, proprioception_dim=113, action_dim=12):
        self.prop_dim   = proprioception_dim
        self.action_dim = action_dim

        self.graph = PacketGraph(
            input_dim    = proprioception_dim,
            action_dim   = action_dim,
            hidden_dim   = 96,
            max_episodes = 100
        )

        self.prev_obs    = None
        self.prev_action = None
        self.episode_surprise = []

    def step(self, obs, external_action=None):
        human_driving = external_action is not None

        if self.prev_obs is not None:
            obs_tensor = torch.tensor(
                self.prev_obs, dtype=torch.float32
            )

            # determine action
            if human_driving:
                action = external_action
            else:
                # get action from forward pass
                with torch.no_grad():
                    predicted = self.graph.forward(obs_tensor)
                action = predicted[
                    ..., self.prop_dim:
                ].numpy()
                action = np.clip(action, -1.0, 1.0)

            # actual next state = current obs + action taken
            actual = torch.tensor(
                np.concatenate([obs, action]),
                dtype=torch.float32
            )

            # update — fresh forward pass inside packet
            self.graph.update(obs_tensor, actual)

            # log surprise for monitoring
            surprise = self.graph.compute_surprise(
                obs_tensor, actual
            )
            self.episode_surprise.append(surprise)

        else:
            # first step of episode
            if human_driving:
                action = external_action
            else:
                x = torch.tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    predicted = self.graph.forward(x)
                action = predicted[
                    ..., self.prop_dim:
                ].numpy()
                action = np.clip(action, -1.0, 1.0)

        self.prev_obs    = obs.copy()
        self.prev_action = action

        return action

    def end_episode(self):
        mean_surprise = np.mean(self.episode_surprise) \
            if self.episode_surprise else 0.0

        print(f"episode surprise: {mean_surprise:.4f} | "
              f"packets: {len(self.graph.packets)}")

        self.graph.record_surprise(mean_surprise)
        self.graph.end_episode()

        if self.graph.check_plateau():
            self.graph.grow()
            self.graph.surprise_history = []

        self.episode_surprise = []
        self.prev_obs         = None
        self.prev_action      = None