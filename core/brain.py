import torch
import numpy as np
from core.graph import PacketGraph, PROP_DIM, VISION_DIM, ACTION_DIM


class Brain:
    """
    Full system. Always live. Always learning.
    No training/inference distinction.
    Single objective: minimize surprise.

    Accepts two input streams per step:
      - proprioception (113-dim)
      - vision (768-dim, egocentric camera)

    When external_action is provided:
      brain observes and learns — does NOT interfere with action

    When external_action is None:
      brain drives

    When injected_surprise is provided (from whistle):
      overrides computed surprise for Hebbian scaling and logging
      brain still runs normal forward/update pass
    """

    def __init__(self):
        self.graph = PacketGraph(
            hidden_dim   = 96,
            max_episodes = 100
        )

        self.prev_prop    = None
        self.prev_vision  = None
        self.prev_action  = None
        self.episode_surprise = []

    def step(self, prop, vision,
             external_action=None,
             injected_surprise=None):
        """
        prop              — np.ndarray (113,)
        vision            — np.ndarray (768,)
        external_action   — np.ndarray (12,) or None
        injected_surprise — float or None
        """
        human_driving = external_action is not None

        prop_t   = torch.tensor(prop,   dtype=torch.float32)
        vision_t = torch.tensor(vision, dtype=torch.float32)

        if self.prev_prop is not None:
            # determine action
            if human_driving:
                action = external_action
            else:
                with torch.no_grad():
                    action_t = self.graph.forward(prop_t, vision_t)
                action = np.clip(action_t.numpy(), -1.0, 1.0)

            # actual next state = prop + action taken
            actual = torch.tensor(
                np.concatenate([prop, action]),
                dtype=torch.float32
            )

            # always compute real surprise for logging
            computed_surprise = self.graph.compute_surprise(
                prop_t, vision_t, actual
            )

            # what gets logged — real signal, not injected
            surprise_to_log = computed_surprise

            # update graph — injection modulates loss directly in packets
            self.graph.update(
                torch.tensor(self.prev_prop,   dtype=torch.float32),
                torch.tensor(self.prev_vision, dtype=torch.float32),
                actual,
                surprise         = computed_surprise,
                injected_surprise = injected_surprise
            )

            self.episode_surprise.append(surprise_to_log)

        else:
            # first step — no previous state to update from
            if human_driving:
                action = external_action
            else:
                with torch.no_grad():
                    action_t = self.graph.forward(prop_t, vision_t)
                action = np.clip(action_t.numpy(), -1.0, 1.0)

        self.prev_prop   = prop.copy()
        self.prev_vision = vision.copy()
        self.prev_action = action

        return action

    def end_episode(self):
        mean_surprise = float(np.mean(self.episode_surprise)) \
            if self.episode_surprise else 0.0

        print(f"  surprise: {mean_surprise:.4f} | "
              f"packets: {len(self.graph.packets)}")

        self.graph.record_surprise(mean_surprise)
        self.graph.end_episode()

        # per-packet pressure check — grow where needed
        self.graph.check_and_grow()

        ep_surprise = self.episode_surprise[:]   # copy for logging
        self.episode_surprise = []
        self.prev_prop        = None
        self.prev_vision      = None
        self.prev_action      = None

        return mean_surprise, ep_surprise