import os

def _get_env(name, default_val):
    val = os.environ.get(name)
    if val is None:
        return default_val
    if isinstance(default_val, int):
        return int(val)
    if isinstance(default_val, float):
        return float(val)
    if isinstance(default_val, bool):
        return val.lower() in ("true", "1", "yes")
    return val

# ── Keypoint schema (11 points) ──────────────────────────────────────────────
# Order matters: must match the SLEAP skeleton definition exactly.
KEYPOINTS = [
    "nose",           # nose tip
    "left_ear",
    "right_ear",
    "head_center",    # center of head — scratch TARGET reference
    "neck",
    "left_fore_paw",
    "right_fore_paw",
    "spine_mid",
    "left_hind_paw",  # ★ scratch detection core (effector)
    "right_hind_paw", # ★ scratch detection core (effector)
    "tail_base",      # color-ID sampling zone
]

# SLEAP skeleton edges (pairs to connect in the skeleton graph)
SKELETON_EDGES = [
    ("nose",          "head_center"),
    ("head_center",   "left_ear"),
    ("head_center",   "right_ear"),
    ("head_center",   "neck"),
    ("neck",          "left_fore_paw"),
    ("neck",          "right_fore_paw"),
    ("neck",          "spine_mid"),
    ("spine_mid",     "left_hind_paw"),
    ("spine_mid",     "right_hind_paw"),
    ("spine_mid",     "tail_base"),
]

# ── Individual ID via tail-color marking ─────────────────────────────────────
# Each mouse has its tail painted a unique color.
# HSV ranges tuned for lab lighting (adjust after camera calibration).
MOUSE_COLORS = {
    1: {"name": "red",    "hsv_lower": (0,   120,  80), "hsv_upper": (10,  255, 255)},
    2: {"name": "blue",   "hsv_lower": (100, 100,  80), "hsv_upper": (130, 255, 255)},
    3: {"name": "yellow", "hsv_lower": (20,  120, 120), "hsv_upper": (35,  255, 255)},
    4: {"name": "black",  "hsv_lower": (0,   0,    0),  "hsv_upper": (180, 40,   40)}, # M-1: 黒のVを縮小
}
# Red also wraps around hue=180 in OpenCV HSV — handled in color_id.py
RED_WRAP_LOWER = (170, 120, 80)
RED_WRAP_UPPER = (180, 255, 255)

# ── SLEAP confidence thresholds ──────────────────────────────────────────────
MIN_CONFIDENCE          = _get_env("MIN_CONFIDENCE", 0.40)  # general keypoints
HIND_PAW_MIN_CONF       = _get_env("HIND_PAW_MIN_CONF", 0.35)  # hind paws can be partially occluded — relax slightly
INTERPOLATION_MAX_GAP   = _get_env("INTERPOLATION_MAX_GAP", 5)     # frames: interpolate gaps shorter than this

# ── Scratch-detection thresholds ─────────────────────────────────────────────
SCRATCH_WINDOW_FRAMES    = _get_env("SCRATCH_WINDOW_FRAMES", 15)   # sliding window for oscillation analysis
SCRATCH_MIN_VELOCITY     = _get_env("SCRATCH_MIN_VELOCITY", 30)   # px/frame — hind-paw minimum speed
SCRATCH_MAX_HEAD_DIST    = _get_env("SCRATCH_MAX_HEAD_DIST", 150)  # px  — hind paw must be this close to head
SCRATCH_MIN_OSCILLATIONS = _get_env("SCRATCH_MIN_OSCILLATIONS", 3)    # velocity-sign reversals per window

# ── Bout post-processing ──────────────────────────────────────────────────────
BOUT_MIN_FRAMES   = _get_env("BOUT_MIN_FRAMES", 5)   # discard bouts shorter than this
BOUT_MERGE_GAP    = _get_env("BOUT_MERGE_GAP", 10)  # merge adjacent bouts separated by ≤ this many frames

# ── Camera / video defaults ───────────────────────────────────────────────────
PIXEL_TO_MM_RATIO  = _get_env("PIXEL_TO_MM_RATIO", 0.25)  # mm/pixel: 1px = 0.25mm
DEFAULT_FPS        = _get_env("DEFAULT_FPS", 30)
CAGE_MICE_COUNT    = _get_env("CAGE_MICE_COUNT", 4)

# ── Paths (override via CLI or environment variables) ─────────────────────────
VIDEO_DIR   = _get_env("VIDEO_DIR", "data/videos/")
SLEAP_DIR   = _get_env("SLEAP_DIR", "data/sleap/")
OUTPUT_DIR  = _get_env("OUTPUT_DIR", "data/outputs/")
SIMBA_DIR   = _get_env("SIMBA_DIR", "data/simba_project/")

