import gymnasium as gym
import numpy as np
import os
import torch
from pynput import keyboard
from core.brain import Brain

# ── save path ──────────────────────────────────────────────
SAVE_PATH = "models/brain.pt"
os.makedirs("models", exist_ok=True)

# ── keyboard state ─────────────────────────────────────────
keys_held = set()

def on_press(key):
    try:
        keys_held.add(key.char)
    except AttributeError:
        keys_held.add(key)

def on_release(key):
    try:
        keys_held.discard(key.char)
    except AttributeError:
        keys_held.discard(key)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)
listener.start()

# ── action builder from keys ───────────────────────────────
# Go1 joints:
# 0  FR hip    1  FR thigh   2  FR calf
# 3  FL hip    4  FL thigh   5  FL calf
# 6  RR hip    7  RR thigh   8  RR calf
# 9  RL hip    10 RL thigh   11 RL calf

def keys_to_action():
    action = np.zeros(12, dtype=np.float32)

    if 'w' in keys_held:   # forward
        action[[0, 3]] =  0.6   # front hips forward
        action[[6, 9]] = -0.6   # rear hips forward
        action[[1, 4]] = -0.4   # front thighs down
        action[[7,10]] = -0.4   # rear thighs down
        action[[2, 5]] =  0.6   # front calves extend
        action[[8,11]] =  0.6   # rear calves extend

    if 's' in keys_held:   # backward
        action[[0, 3]] = -0.6
        action[[6, 9]] =  0.6
        action[[1, 4]] = -0.4
        action[[7,10]] = -0.4
        action[[2, 5]] =  0.6
        action[[8,11]] =  0.6

    if 'a' in keys_held:   # turn left
        action[[0, 6]] =  0.4   # right side hips
        action[[3, 9]] = -0.2   # left side hips

    if 'd' in keys_held:   # turn right
        action[[3, 9]] =  0.4   # left side hips
        action[[0, 6]] = -0.2   # right side hips

    return action

# ── env ────────────────────────────────────────────────────
env = gym.make(
    "Ant-v5",
    xml_file=os.path.abspath(
        "mujoco_menagerie/unitree_go1/scene.xml"
    ),
    render_mode="human",
    terminate_when_unhealthy=False,
)

# ── brain ──────────────────────────────────────────────────
brain = Brain(
    proprioception_dim=113,
    action_dim=12
)

if os.path.exists(SAVE_PATH):
    checkpoint = torch.load(SAVE_PATH, weights_only=False)
    brain.graph.packets.load_state_dict(
        checkpoint["packets"]
    )
    brain.graph.surprise_history = checkpoint.get(
        "surprise_history", []
    )
    print(f"Loaded brain from {SAVE_PATH}")
else:
    print("Starting fresh brain")

# ── main loop ──────────────────────────────────────────────
NUM_EPISODES = 10000
human_control = True   # start in human control mode

print("\nControls:")
print("  W/A/S/D  — move the dog")
print("  B        — toggle brain control on/off")
print("  Q        — quit and save\n")

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    obs = obs[:113]

    print(f"\n=== Episode {ep+1} | "
          f"packets: {len(brain.graph.packets)} | "
          f"mode: {'YOU' if human_control else 'BRAIN'} ===")

    for step in range(1000):

        # toggle brain/human control
        if 'b' in keys_held:
            human_control = not human_control
            print(f"Switched to: "
                  f"{'YOU' if human_control else 'BRAIN'}")
            keys_held.discard('b')

        # quit and save
        if 'q' in keys_held:
            torch.save({
                "packets": brain.graph.packets.state_dict(),
                "surprise_history": brain.graph.surprise_history,
            }, SAVE_PATH)
            print("Saved and quitting.")
            env.close()
            listener.stop()
            exit()

        if human_control:
            # you drive — brain watches
            action = keys_to_action()
            brain.step(obs, external_action=action)
        else:
            # brain drives
            action = brain.step(obs, external_action=None)

        obs, _, terminated, truncated, _ = env.step(action)
        obs = obs[:113]

        if terminated or truncated:
            break

    brain.end_episode()

    # save every episode
    torch.save({
        "packets": brain.graph.packets.state_dict(),
        "surprise_history": brain.graph.surprise_history,
    }, SAVE_PATH)

env.close()
listener.stop()