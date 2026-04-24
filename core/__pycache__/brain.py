import torch
import numpy as np
from core.graph import PacketGraph

class Brain:
    """
    Full system.
    Always live. Always learning.
    No training/inference distinction.
    Single objective: minimize surprise.
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
        """
        one timestep.
        obs = raw proprioception (113,)
        external_action = if you are controlling the robot
                          None if robot controls itself
        returns action (12,)
        """
        x = torch.tensor(obs, dtype=torch.float32)

        # forward — predict next state
        predicted = self.graph.forward(x)

        # action falls out of prediction naturally
        # last action_dim values of predicted state = action
        if external_action is not None:
            # you are controlling — use your action
            action = external_action
        else:
            # robot controls itself
            # extract action from predicted next state
            action = predicted[..., self.prop_dim:].detach().numpy()
            action = np.clip(action, -1.0, 1.0)

        # compute surprise if we have previous prediction
        if self.prev_obs is not None:
            # actual next state = current obs + action taken
            actual_next = torch.tensor(
                np.concatenate([obs, action]),
                dtype=torch.float32
            )

            # predicted next state from previous step
            surprise = self.graph.compute_surprise(
                self.prev_predicted, actual_next
            )

            # update all plastic packets
            self.graph.update(surprise)

            self.episode_surprise.append(surprise.item())

        # store for next step
        self.prev_obs       = obs
        self.prev_action    = action
        self.prev_predicted = predicted

        return action

    def end_episode(self):
        """call at end of each episode"""
        mean_surprise = np.mean(self.episode_surprise) \
                        if self.episode_surprise else 0.0

        print(f"episode surprise: {mean_surprise:.4f} | "
              f"packets: {len(self.graph.packets)}")

        self.graph.record_surprise(mean_surprise)
        self.graph.end_episode()

        # check if graph should grow
        if self.graph.check_plateau():
            self.graph.grow()
            self.graph.surprise_history = []  # reset after growing

        self.episode_surprise = []
        self.prev_obs         = None
        self.prev_action      = None
        self.prev_predicted   = None