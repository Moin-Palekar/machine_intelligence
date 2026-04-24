import gymnasium as gym
import numpy as np
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.brain import Brain
from core.vision import get_vision
from viz.graph_viz import GraphViz

# ── paths ──────────────────────────────────────────────────
SAVE_PATH  = "models/brain.pt"
SCENE_PATH = os.path.abspath("mujoco_menagerie/unitree_go1/scene.xml")
os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)
os.makedirs("data/graphs", exist_ok=True)

# ── circle positions ───────────────────────────────────────
RED_POS   = np.array([-2.0, 0.0])
GREEN_POS = np.array([ 2.0, 0.0])
CIRCLE_R  = 0.8

# ── surprise injection values ──────────────────────────────
HIGH_SURPRISE = 50.0
LOW_SURPRISE  = 0.0

# ── mode flags ─────────────────────────────────────────────
auto_inject  = True    # A key toggles
test_mode    = False   # T key toggles

# ── keyboard via glfw ──────────────────────────────────────
keys_held    = set()
_manual_inj  = None   # from [ ] keys

def _key_callback(window, key, scancode, action, mods):
    global _manual_inj, auto_inject, test_mode
    import glfw as _glfw

    if action == _glfw.PRESS:
        keys_held.add(key)
        if key == _glfw.KEY_LEFT_BRACKET:
            _manual_inj = ('high', HIGH_SURPRISE)
        elif key == _glfw.KEY_RIGHT_BRACKET:
            _manual_inj = ('low', LOW_SURPRISE)
        elif key == _glfw.KEY_T:
            test_mode = not test_mode
            print(f"\n  ── TEST MODE: {'ON' if test_mode else 'OFF'} ──")
        elif key == _glfw.KEY_A:
            auto_inject = not auto_inject
            print(f"\n  ── AUTO-INJECT: {'ON' if auto_inject else 'OFF'} ──")

    elif action == _glfw.RELEASE:
        keys_held.discard(key)
        if key in (_glfw.KEY_LEFT_BRACKET, _glfw.KEY_RIGHT_BRACKET):
            _manual_inj = None

_glfw_registered = False

def _ensure_glfw_callback(env):
    global _glfw_registered
    if _glfw_registered:
        return
    try:
        import glfw as _glfw
        viewer = env.unwrapped.mujoco_renderer.viewer
        if viewer is not None and hasattr(viewer, 'window'):
            _glfw.set_key_callback(viewer.window, _key_callback)
            _glfw_registered = True
    except Exception:
        pass

# ── action builder ─────────────────────────────────────────
def keys_to_action():
    try:
        import glfw as _glfw
        action = np.zeros(12, dtype=np.float32)
        if _glfw.KEY_W in keys_held:
            action[[0,3]]  =  0.6; action[[6,9]]  = -0.6
            action[[1,4]]  = -0.4; action[[7,10]] = -0.4
            action[[2,5]]  =  0.6; action[[8,11]] =  0.6
        if _glfw.KEY_S in keys_held:
            action[[0,3]]  = -0.6; action[[6,9]]  =  0.6
            action[[1,4]]  = -0.4; action[[7,10]] = -0.4
            action[[2,5]]  =  0.6; action[[8,11]] =  0.6
        if _glfw.KEY_A in keys_held:
            action[[0,6]] =  0.4;  action[[3,9]] = -0.2
        if _glfw.KEY_D in keys_held:
            action[[3,9]] =  0.4;  action[[0,6]] = -0.2
        return action
    except Exception:
        return np.zeros(12, dtype=np.float32)

# ── helpers ────────────────────────────────────────────────
def teleport_to(env, circle_pos):
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    qpos[0]   = circle_pos[0]
    qpos[1]   = circle_pos[1]
    qpos[2]   = 0.35
    qpos[3:7] = [1, 0, 0, 0]
    qpos[7:]  = 0.0
    qvel[:]   = 0.0
    env.unwrapped.set_state(qpos, qvel)
    return env.unwrapped._get_obs()

def on_circle(env, circle_pos):
    xy = env.unwrapped.data.qpos[:2]
    return np.linalg.norm(xy - circle_pos) < CIRCLE_R

# ── surprise plot ──────────────────────────────────────────
def save_surprise_plot(log, test_start_ep):
    red_eps   = [e["episode"] for e in log if e["spawn"] == "RED"]
    green_eps = [e["episode"] for e in log if e["spawn"] == "GREEN"]
    red_surp  = [e["mean_surprise"] for e in log if e["spawn"] == "RED"]
    green_surp= [e["mean_surprise"] for e in log if e["spawn"] == "GREEN"]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.plot(red_eps,   red_surp,   color="#ff4444", linewidth=1.5,
            label="Red circle",   alpha=0.8)
    ax.plot(green_eps, green_surp, color="#44ff88", linewidth=1.5,
            label="Green circle", alpha=0.8)

    if test_start_ep is not None:
        ax.axvline(x=test_start_ep, color="white", linewidth=1,
                   linestyle="--", alpha=0.6, label="Test mode start")

    ax.set_xlabel("Episode", color="white")
    ax.set_ylabel("Mean Surprise", color="white")
    ax.set_title("Brain Surprise by Circle — Red should rise, Green should fall",
                 color="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")
    ax.legend(facecolor="#1a1a1a", edgecolor="#333333",
              labelcolor="white", fontsize=9)

    fig.savefig("data/surprise_plot.png", dpi=100,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)

# ── env ────────────────────────────────────────────────────
env = gym.make(
    "Ant-v5",
    xml_file=SCENE_PATH,
    render_mode="human",
    terminate_when_unhealthy=False,
)

# ── brain ──────────────────────────────────────────────────
brain = Brain()

if os.path.exists(SAVE_PATH):
    checkpoint = torch.load(SAVE_PATH, weights_only=False)
    if "topology" in checkpoint:
        from core.graph import PacketGraph
        brain.graph = PacketGraph.from_topology(
            checkpoint["topology"],
            checkpoint["packets"]
        )
        print(f"Loaded brain from {SAVE_PATH} ({len(brain.graph.packets)} packets)")
    else:
        brain.graph.packets.load_state_dict(checkpoint["packets"])
        brain.graph.surprise_history = checkpoint.get("surprise_history", [])
        print(f"Loaded brain from {SAVE_PATH} (legacy)")
else:
    print("Starting fresh brain")

viz         = GraphViz()
episode_log = []
test_start_ep = None

print("\nControls (focus MuJoCo window first):")
print("  W/A/S/D  — move dog (human control)")
print("  B        — toggle brain/human control")
print("  [        — manual HIGH surprise")
print("  ]        — manual LOW surprise")
print("  A        — toggle auto-inject ON/OFF")
print("  T        — toggle test mode ON/OFF")
print("  Q        — quit and save\n")
print(f"  Auto-inject: ON  |  Test mode: OFF\n")

NUM_EPISODES  = 10000
human_control = False

for ep in range(NUM_EPISODES):

    obs, _ = env.reset()
    _ensure_glfw_callback(env)

    spawn_red   = np.random.rand() < 0.5
    spawn_pos   = RED_POS if spawn_red else GREEN_POS
    circle_name = "RED" if spawn_red else "GREEN"

    obs = teleport_to(env, spawn_pos)
    obs = obs[:113]

    # determine episode injection mode
    if test_mode:
        ep_auto_surprise = None
        mode_label = "TEST"
    elif auto_inject:
        ep_auto_surprise = HIGH_SURPRISE if spawn_red else LOW_SURPRISE
        mode_label = "AUTO"
    else:
        ep_auto_surprise = None
        mode_label = "MANUAL"

    ctrl = "BRAIN" if not human_control else "HUMAN"
    print(f"Ep {ep+1:4d} | {circle_name:5s} | inj:{mode_label:6s} | ctrl:{ctrl} | "
          f"packets: {len(brain.graph.packets)}", end="", flush=True)

    ep_surprises      = []
    ep_on_red_steps   = 0
    ep_on_green_steps = 0

    for step in range(150):

        try:
            import glfw as _glfw
            if _glfw.KEY_B in keys_held:
                human_control = not human_control
                keys_held.discard(_glfw.KEY_B)
            if _glfw.KEY_Q in keys_held:
                torch.save({
                    "packets":  brain.graph.packets.state_dict(),
                    "topology": brain.graph.get_topology(),
                }, SAVE_PATH)
                np.save("data/episode_log.npy", episode_log)
                save_surprise_plot(episode_log, test_start_ep)
                viz.save_final()
                print("\nSaved and quitting.")
                env.close()
                exit()
        except Exception:
            pass

        vision = get_vision(env)

        # resolve injection for this step
        # priority: manual > auto > none
        if _manual_inj is not None:
            inj_surprise = _manual_inj[1]
        elif ep_auto_surprise is not None:
            inj_surprise = ep_auto_surprise
        else:
            inj_surprise = None

        if on_circle(env, RED_POS):
            ep_on_red_steps += 1
        if on_circle(env, GREEN_POS):
            ep_on_green_steps += 1

        if human_control:
            action = keys_to_action()
            brain.step(obs, vision,
                       external_action=action,
                       injected_surprise=inj_surprise)
        else:
            action = brain.step(obs, vision,
                                external_action=None,
                                injected_surprise=inj_surprise)

        obs, _, terminated, truncated, _ = env.step(action)
        obs = obs[:113]

        if terminated or truncated:
            break

    mean_surprise, _ = brain.end_episode()
    ep_surprises.append(mean_surprise)

    # track test mode start
    if test_mode and test_start_ep is None:
        test_start_ep = ep + 1

    print(f" | surprise: {mean_surprise:.4f} | "
          f"red: {ep_on_red_steps}s green: {ep_on_green_steps}s")

    episode_log.append({
        "episode":       ep + 1,
        "spawn":         circle_name,
        "mean_surprise": mean_surprise,
        "mode":          mode_label,
        "n_packets":     len(brain.graph.packets),
    })

    viz.update(brain.graph)
    if ep == 0:
        import subprocess
        subprocess.Popen(["open", "data/graph_latest.html"])
    save_surprise_plot(episode_log, test_start_ep)

    torch.save({
        "packets":  brain.graph.packets.state_dict(),
        "topology": brain.graph.get_topology(),
    }, SAVE_PATH)
    np.save("data/episode_log.npy", episode_log)

env.close()