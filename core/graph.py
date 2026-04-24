import numpy as np
import torch
import torch.nn as nn
from core.packet import Packet


# ── dimensions ────────────────────────────────────────────
PROP_DIM   = 113
VISION_W   = 16
VISION_H   = 16
VISION_DIM = VISION_W * VISION_H * 3   # 768 — full flat frame
ACTION_DIM = 12
N_JOINTS   = 12
N_PIXELS   = VISION_W * VISION_H       # 256

# each vision packet sees one pixel — 3 values (RGB)
PIXEL_DIM  = 3

# each joint packet sees its own slice of prop obs
# joint angles: obs[5:17], joint vels: obs[17:29]
# so each joint i sees: angle[i], vel[i], + 6 base vels = 8 dims
JOINT_INPUT_DIM = 8

# action output packet: produces one joint torque (scalar → 1)
# but we use dim=1 output and concat all 12
ACTION_OUTPUT_DIM = 1


class Connection:
    """
    Hebbian connection between two packets.
    Strengthens on co-activation scaled by surprise.
    Decays when unused.
    """
    def __init__(self, source_idx, target_idx, weight=None):
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.weight     = weight if weight is not None else (
            0.05 + np.random.rand() * 0.1
        )

    def hebbian_update(self, pre, post, surprise, lr=0.001):
        pre_val  = pre.detach().abs().mean().item()
        post_val = post.detach().abs().mean().item()
        self.weight += lr * pre_val * post_val * (1.0 + abs(surprise))
        self.weight  = float(np.clip(self.weight, 0.0, 1.0))

    def decay(self, rate=0.0001):
        self.weight *= (1.0 - rate)


class PacketGraph:
    """
    Growing graph of uniform packets.

    Initial topology:
      - 12 joint entry packets   (one per joint)
      - 256 vision entry packets (one per pixel, 3-dim RGB input)
      - 8 internal core packets  (seed — grows via pressure)
      - 12 action output packets (one per joint actuator)

    All packets are the same type.
    Boundary vs internal is only a label — growth and connection
    rules are identical everywhere.
    The graph grows wherever pressure is highest, boundary or not.
    """

    def __init__(self, hidden_dim=32, max_episodes=100):
        self.hidden_dim   = hidden_dim
        self.max_episodes = max_episodes

        self.packets     = nn.ModuleList()
        self.connections = []

        # role registry — label only, no behavioral difference
        self.joint_entry_idxs  = []   # 12
        self.vision_entry_idxs = []   # 256
        self.internal_idxs     = []   # starts at 8, grows
        self.action_output_idxs = []  # 12

        self._build_initial_topology()

        self.surprise_history  = []
        self.plateau_window    = 20
        self.plateau_threshold = 0.01
        self.last_surprise     = 0.0
        self._last_activations = {}

    # ── construction ──────────────────────────────────────

    def _make_packet(self, input_dim, output_dim):
        idx = len(self.packets)
        self.packets.append(Packet(
            input_dim    = input_dim,
            hidden_dim   = self.hidden_dim,
            output_dim   = output_dim,
            max_episodes = self.max_episodes
        ))
        return idx

    def _connect(self, src, tgt, weight=None):
        self.connections.append(Connection(src, tgt, weight))

    def _build_initial_topology(self):
        # ── joint entry packets ───────────────────────────
        for _ in range(N_JOINTS):
            idx = self._make_packet(JOINT_INPUT_DIM, self.hidden_dim)
            self.joint_entry_idxs.append(idx)

        # ── vision entry packets ──────────────────────────
        for _ in range(N_PIXELS):
            idx = self._make_packet(PIXEL_DIM, self.hidden_dim)
            self.vision_entry_idxs.append(idx)

        # ── internal core packets ─────────────────────────
        for _ in range(8):
            idx = self._make_packet(self.hidden_dim, self.hidden_dim)
            self.internal_idxs.append(idx)

        # ── action output packets ─────────────────────────
        for _ in range(N_JOINTS):
            idx = self._make_packet(self.hidden_dim, ACTION_OUTPUT_DIM)
            self.action_output_idxs.append(idx)

        core = self.internal_idxs
        out  = self.action_output_idxs

        # each joint entry → 2 random core packets
        for ji in self.joint_entry_idxs:
            targets = np.random.choice(len(core), size=2, replace=False)
            for t in targets:
                self._connect(ji, core[t])

        # vision entry packets — sparse random connections to core
        # each pixel connects to 1 random core packet
        # (sparse — most vision→core connections will grow via pressure)
        for vi in self.vision_entry_idxs:
            t = np.random.randint(len(core))
            self._connect(vi, core[t])

        # core → core sparse random
        for i, ci in enumerate(core):
            for j, cj in enumerate(core):
                if i != j and np.random.rand() < 0.4:
                    self._connect(ci, cj)

        # core → each action output
        for oi in out:
            # each output gets 2-3 random core inputs
            srcs = np.random.choice(len(core),
                                    size=np.random.randint(2, 4),
                                    replace=False)
            for s in srcs:
                self._connect(core[s], oi)

    # ── forward ───────────────────────────────────────────

    def forward(self, prop, vision):
        """
        prop   — torch tensor (113,)
        vision — torch tensor (768,) flat RGB frame
        Returns action tensor (12,)
        """
        acts = {}

        # joint angles: prop[5:17], vels: prop[17:29], base: prop[29:35]
        base_vels = prop[29:35]   # 6 dims

        for i, ji in enumerate(self.joint_entry_idxs):
            angle = prop[5 + i].unsqueeze(0)    # (1,)
            vel   = prop[17 + i].unsqueeze(0)   # (1,)
            inp   = torch.cat([angle, vel, base_vels])  # (8,)
            acts[ji] = self.packets[ji](inp)

        # vision — each pixel packet sees 3 RGB values
        for i, vi in enumerate(self.vision_entry_idxs):
            pixel = vision[i*3 : i*3 + 3]   # (3,)
            acts[vi] = self.packets[vi](pixel)

        # internal packets — sum weighted inputs
        for ci in self.internal_idxs:
            inp = self._collect_inputs(ci, acts)
            if inp is not None:
                acts[ci] = self.packets[ci](inp)
            else:
                acts[ci] = torch.zeros(self.hidden_dim)

        # action output packets — one scalar per joint
        action_parts = []
        for oi in self.action_output_idxs:
            inp = self._collect_inputs(oi, acts)
            if inp is not None:
                out = self.packets[oi](inp)
            else:
                out = torch.zeros(ACTION_OUTPUT_DIM)
            acts[oi] = out
            action_parts.append(out)

        # stack 12 scalars → (12,)
        action = torch.cat(action_parts)

        # hebbian updates
        for conn in self.connections:
            if conn.source_idx in acts and conn.target_idx in acts:
                conn.hebbian_update(
                    acts[conn.source_idx],
                    acts[conn.target_idx],
                    self.last_surprise
                )
                conn.decay()

        self._last_activations = acts
        return action

    def _collect_inputs(self, target_idx, acts):
        parts = []
        for conn in self.connections:
            if conn.target_idx == target_idx and \
               conn.source_idx in acts:
                parts.append(acts[conn.source_idx] * conn.weight)
        if not parts:
            return None
        total = torch.stack(parts).sum(dim=0)
        exp   = self.packets[target_idx].input_dim
        if total.shape[0] < exp:
            total = torch.nn.functional.pad(total, (0, exp - total.shape[0]))
        elif total.shape[0] > exp:
            total = total[:exp]
        return total

    # ── update ────────────────────────────────────────────

    def update(self, prop, vision, actual, surprise,
               injected_surprise=None):
        """
        Injection enters ONLY at boundary packets (joint sensors + vision).
        Internal and output packets feel it only through propagation.
        """
        self.last_surprise = surprise
        base_vels = prop[29:35]

        for i, ji in enumerate(self.joint_entry_idxs):
            if self.packets[ji].frozen:
                continue
            angle = prop[5 + i].unsqueeze(0)
            vel   = prop[17 + i].unsqueeze(0)
            inp   = torch.cat([angle, vel, base_vels])
            tgt   = self._pad_tgt(actual, self.hidden_dim)
            self.packets[ji].update(inp, tgt, injected_surprise=None)  # joints dont see color

        for i, vi in enumerate(self.vision_entry_idxs):
            if self.packets[vi].frozen:
                continue
            pixel = vision[i*3 : i*3 + 3]
            tgt   = self._pad_tgt(actual, self.hidden_dim)
            self.packets[vi].update(pixel, tgt, injected_surprise)

        for idx in self.internal_idxs + self.action_output_idxs:
            if self.packets[idx].frozen:
                continue
            if idx in self._last_activations:
                act = self._last_activations[idx].detach()
                exp = self.packets[idx].input_dim
                inp = act[:exp] if act.shape[0] >= exp else \
                    torch.nn.functional.pad(act, (0, exp - act.shape[0]))
                out_exp = self.packets[idx].output_dim
                tgt = self._pad_tgt(actual, out_exp)
                self.packets[idx].update(inp, tgt, injected_surprise=None)  # no injection — propagates naturally

    @staticmethod
    def _pad_tgt(actual, size):
        if actual.shape[0] >= size:
            return actual[:size]
        return torch.nn.functional.pad(
            actual, (0, size - actual.shape[0])
        )

    def compute_surprise(self, prop, vision, actual):
        with torch.no_grad():
            ji  = self.joint_entry_idxs[0]
            base_vels = prop[29:35]
            inp = torch.cat([
                prop[5].unsqueeze(0),
                prop[17].unsqueeze(0),
                base_vels
            ])
            pred = self.packets[ji].net(inp)
            tgt  = self._pad_tgt(actual, self.hidden_dim)
            return torch.mean((pred - tgt) ** 2).item()

    # ── growth — same rule everywhere ─────────────────────

    def end_episode(self):
        for p in self.packets:
            p.end_episode()

    def record_surprise(self, val):
        self.surprise_history.append(val)

    def check_and_grow(self):
        """
        Per-packet pressure check — grow wherever pressure exceeds
        threshold. Called once per episode.
        Returns list of packet indices where growth was triggered.
        """
        from core.packet import PRESSURE_GROWTH_THRESHOLD
        grew_at = []
        # snapshot current packets — dont iterate while growing
        for idx in list(range(len(self.packets))):
            if self.packets[idx].need_growth:
                self._grow_at(idx)
                grew_at.append(idx)
                self.packets[idx].need_growth = False
        return grew_at

    def _grow_at(self, hottest):
        """
        Insert new packet next to hottest pressure node.
        """
        hot_pkt = self.packets[hottest]
        new_idx = self._make_packet(hot_pkt.input_dim, hot_pkt.output_dim)

        if hottest in self.joint_entry_idxs:
            self.joint_entry_idxs.append(new_idx)
        elif hottest in self.vision_entry_idxs:
            self.vision_entry_idxs.append(new_idx)
        elif hottest in self.action_output_idxs:
            # action output boundary cannot grow outward —
            # grow inward as internal packet instead
            self.internal_idxs.append(new_idx)
        else:
            self.internal_idxs.append(new_idx)

        self._connect(hottest, new_idx)
        downstream = [c.target_idx for c in self.connections
                      if c.source_idx == hottest]
        for d in downstream[:2]:
            self._connect(new_idx, d)

        print(f"  ↑ grew p{new_idx} at p{hottest} "
              f"(pressure={hot_pkt.get_pressure():.3f}) "
              f"→ {len(self.packets)} packets")

    # ── consciousness field ────────────────────────────────

    def get_phi(self):
        if not self._last_activations:
            return {i: 0.0 for i in range(len(self.packets))}
        acts = self._last_activations
        phi  = {}
        idxs = list(acts.keys())
        for i in idxs:
            ai    = acts[i].detach()
            total = 0.0
            for j in idxs:
                if i == j:
                    continue
                aj     = acts[j].detach()
                min_l  = min(ai.shape[0], aj.shape[0])
                total += torch.dot(ai[:min_l], aj[:min_l]).abs().item()
            phi[i] = total
        return phi

    def get_connection_weights(self):
        return [(c.source_idx, c.target_idx, c.weight)
                for c in self.connections]

    # ── save / load full topology ──────────────────────────

    def get_topology(self):
        """
        Serialize full graph topology — packet roles, dims,
        freeze state, connections — everything needed to
        reconstruct the graph exactly on reload.
        """
        packets_meta = []
        for i, p in enumerate(self.packets):
            packets_meta.append({
                "idx":           i,
                "input_dim":     p.input_dim,
                "output_dim":    p.output_dim,
                "episodes_lived":p.episodes_lived,
                "frozen":        p.frozen,
            })

        connections = [
            {"src": c.source_idx, "tgt": c.target_idx, "w": c.weight}
            for c in self.connections
        ]

        return {
            "packets_meta":      packets_meta,
            "connections":       connections,
            "joint_entry_idxs":  self.joint_entry_idxs,
            "vision_entry_idxs": self.vision_entry_idxs,
            "internal_idxs":     self.internal_idxs,
            "action_output_idxs":self.action_output_idxs,
            "surprise_history":  self.surprise_history,
            "hidden_dim":        self.hidden_dim,
            "max_episodes":      self.max_episodes,
        }

    @classmethod
    def from_topology(cls, topology, state_dict):
        """
        Reconstruct graph from saved topology + packet weights.
        """
        g = cls.__new__(cls)
        g.hidden_dim        = topology["hidden_dim"]
        g.max_episodes      = topology["max_episodes"]
        g.surprise_history  = topology["surprise_history"]
        g.plateau_window    = 20
        g.plateau_threshold = 0.01
        g.last_surprise     = 0.0
        g._last_activations = {}

        g.joint_entry_idxs   = topology["joint_entry_idxs"]
        g.vision_entry_idxs  = topology["vision_entry_idxs"]
        g.internal_idxs      = topology["internal_idxs"]
        g.action_output_idxs = topology["action_output_idxs"]

        # rebuild packets
        g.packets = nn.ModuleList()
        for meta in topology["packets_meta"]:
            p = Packet(
                input_dim    = meta["input_dim"],
                hidden_dim   = g.hidden_dim,
                output_dim   = meta["output_dim"],
                max_episodes = g.max_episodes,
            )
            p.episodes_lived = meta["episodes_lived"]
            if meta["frozen"]:
                p.frozen = True
                for param in p.parameters():
                    param.requires_grad = False
            g.packets.append(p)

        # load weights
        g.packets.load_state_dict(state_dict)

        # rebuild connections
        g.connections = []
        for c in topology["connections"]:
            g.connections.append(
                Connection(c["src"], c["tgt"], c["w"])
            )

        return g