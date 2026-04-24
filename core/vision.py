import numpy as np


VISION_W = 16
VISION_H = 16
VISION_DIM = VISION_W * VISION_H * 3   # 768


def get_vision(env) -> np.ndarray:
    """
    Render egocentric camera frame from MuJoCo.
    Downsample to 16x16, normalize to [-1, 1].
    Returns flat vector of shape (768,).
    """
    try:
        frame = env.unwrapped.mujoco_renderer.render(
            "rgb_array",
            camera_name="egocentric"
        )
    except Exception:
        # fallback — return zeros if render fails
        return np.zeros(VISION_DIM, dtype=np.float32)

    # downsample to 16x16 using simple stride sampling
    h, w, c = frame.shape
    row_idx = np.linspace(0, h - 1, VISION_H, dtype=int)
    col_idx = np.linspace(0, w - 1, VISION_W, dtype=int)
    small = frame[np.ix_(row_idx, col_idx)]   # (16, 16, 3)

    # normalize uint8 [0, 255] → float32 [-1, 1]
    normalized = small.astype(np.float32) / 127.5 - 1.0

    return normalized.flatten()   # (768,)