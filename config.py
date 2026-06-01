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
# Each mouse has its tail painted a unique color, or has a natural coat color.
# HSV ranges tuned for lab lighting (adjust after camera calibration).
# ID 1-4: マーカー（尾部塗装）による個体識別
# ID 5:   白色マウス（無マーカー）— 高明度・低彩度のHSV範囲
# ID 6-8: 拡張スロット（ケージ内個体数が増加した際に追加）
MOUSE_COLORS = {
    1: {"name": "red",    "hsv_lower": (0,   120,  80), "hsv_upper": (10,  255, 255)},
    2: {"name": "blue",   "hsv_lower": (100, 100,  80), "hsv_upper": (130, 255, 255)},
    3: {"name": "yellow", "hsv_lower": (20,  120, 120), "hsv_upper": (35,  255, 255)},
    4: {"name": "black",  "hsv_lower": (0,   0,    0),  "hsv_upper": (180, 40,   40)},  # M-1: 黒のVを縮小
    # 白色マウス: 彩度が低く (S < 50) 明度が高い (V > 170) 領域 — マーカーなし個体の自然毛色
    # 注意: 背景・床面が白い場合は hsv_lower の V を上げて誤検出を抑制すること
    5: {"name": "white",  "hsv_lower": (0,   0,  170), "hsv_upper": (180, 50, 255)},
    # オレンジ・緑・紫: 将来の拡張スロット（有効にするには実験条件に合わせてHSV値を調整）
    6: {"name": "orange", "hsv_lower": (10,  120,  80), "hsv_upper": (22,  255, 255)},
    7: {"name": "green",  "hsv_lower": (40,  80,   60), "hsv_upper": (80,  255, 255)},
    8: {"name": "purple", "hsv_lower": (130, 60,   50), "hsv_upper": (160, 255, 255)},
}
# Red also wraps around hue=180 in OpenCV HSV — handled in color_id.py
RED_WRAP_LOWER = (170, 120, 80)
RED_WRAP_UPPER = (180, 255, 255)

# 実験で実際に使用するマウスIDの集合（0 = すべてのIDを自動使用）
# 例: ACTIVE_MOUSE_IDS = "1,2,5" → マーカーなし白マウス (ID=5) を含む3匹構成
ACTIVE_MOUSE_IDS_STR = _get_env("ACTIVE_MOUSE_IDS", "")
ACTIVE_MOUSE_IDS: list = (
    [int(x) for x in ACTIVE_MOUSE_IDS_STR.split(",") if x.strip().isdigit()]
    if ACTIVE_MOUSE_IDS_STR else []
)

# ── 個体識別: 色 + 帯本数 (color + band count) ────────────────────────────────
# 尾に塗った「色」だけでなく「帯の本数」も併用して個体を区別する方式。
# 例: 赤1本/赤2本/青1本/青2本 の 4 通りで 4 個体を一意に識別する。
# 純粋な色相のみ (MOUSE_COLORS) では「赤1本」と「赤2本」を区別できないため拡張する。
#
# 対応表: (色名, 帯本数) → mouse_id。色名は MOUSE_COLORS の "name" と一致させること。
# 帯本数は {1, 2} にクランプして評価する（3 本以上は 2 として扱う）。
MOUSE_ID_BY_COLOR_COUNT = {
    ("red",  1): 1,   # 赤1本
    ("red",  2): 2,   # 赤2本
    ("blue", 1): 3,   # 青1本
    ("blue", 2): 4,   # 青2本
}
# 帯本数判定に使用する色（MOUSE_COLORS のキー）。ここに無い色は本数判定の対象外。
BAND_COUNT_COLOR_IDS = [1, 2]  # 1=red, 2=blue

# ── 尾チェーン (tail chain) — 帯カウント用の尾部折れ線 ────────────────────────
# スクラッチ判定用の 11 点スケルトンとは別系統で、尾に沿った順序付きキーポイント列を保持する。
# DLC SuperAnimal-TopViewMouse は tail_base〜tail_end の複数尾ノードを出力するため、
# それらを base→end の順に並べた折れ線として帯サンプリングに用いる。
# pose_engine 側では「名前が 'tail' で始まる bodypart」を末尾番号順に保持する（命名差に頑健）。
# feature_extraction は KEYPOINTS(11点) のみ参照するため、本チェーンは特徴量に非干渉。
TAIL_CHAIN_KEYPOINTS = ["tail_base", "tail1", "tail2", "tail3", "tail_end"]

# ── 帯カウントの幾何パラメータ (px) ──────────────────────────────────────────
# 実測校正: 提示動画 300.mov (1920x1080) の赤/青マーカー連結成分を 40 フレームで計測。
#   赤帯: 尾方向長さ 中央値 27px / 尾を横切る幅 中央値 18px
#   青帯: 尾方向長さ 中央値 15px (p25=12px) / 幅 中央値 11px
# 解像度依存のため、撮影解像度が変わった場合は再計測すること（mm 換算は本カウントには不要）。
TAIL_SAMPLE_STEP_PX   = _get_env("TAIL_SAMPLE_STEP_PX",   4)   # 尾折れ線に沿った標本化間隔
TAIL_SAMPLE_RADIUS_PX = _get_env("TAIL_SAMPLE_RADIUS_PX", 7)   # 各標本点の円形パッチ半径（細い青帯にも収まる）
BAND_MIN_LEN_PX       = _get_env("BAND_MIN_LEN_PX",       6)   # 1 帯とみなす最小の尾方向連続長（最小実帯~12px の約半分→ノイズ除去）
BAND_MIN_GAP_PX       = _get_env("BAND_MIN_GAP_PX",       10)  # 2 帯を分離する皮膚ギャップの最小長（市松/途切れは結合し過分割を防ぐ）

# ── SLEAP confidence thresholds ──────────────────────────────────────────────
MIN_CONFIDENCE          = _get_env("MIN_CONFIDENCE", 0.40)  # general keypoints
HIND_PAW_MIN_CONF       = _get_env("HIND_PAW_MIN_CONF", 0.35)  # hind paws can be partially occluded — relax slightly
INTERPOLATION_MAX_GAP   = _get_env("INTERPOLATION_MAX_GAP", 5)     # frames: interpolate gaps shorter than this

# ── Scratch-detection thresholds (PHYSICAL UNITS — cross-study reproducible) ─
# 文献根拠:
#  - Kuraishi et al. (1995) Eur J Pharmacol 275:229: マウスのscratchは ~10-15 Hz の振動
#  - Akiyama et al. (2010) J Invest Dermatol 130:2474: scratch bout の典型継続時間 0.3-3 秒
#  - Inagaki et al. (2002) Eur J Pharmacol 437:175: 後肢ピーク速度は数百 mm/s 級
#
# SCRATCH_WINDOW_FRAMES: 振動カウント窓。30 frames = 1秒。Default 30 で  10 Hz scratch を確実に捕捉。
# （以前の 15 frames = 0.5 秒では振動が確実に捕捉されない場合があった）
SCRATCH_WINDOW_FRAMES    = _get_env("SCRATCH_WINDOW_FRAMES", 30)     # frames — oscillation analysis window

# 最小後肢速度（スムージング後）。Documented scratch peak velocities 200-500 mm/s だが、
# 5 フレーム移動平均後の値なので 150 mm/s が現実的下限。
SCRATCH_MIN_VELOCITY_MM_S    = _get_env("SCRATCH_MIN_VELOCITY_MM_S",  150.0)

# 後肢→頭部最近接距離 (mm)。体長 80-110 mm の ~半分。
# 注: feature_extraction では min(hind_to_ear, hind_to_head_center) を用いる。
SCRATCH_MAX_HEAD_DIST_MM     = _get_env("SCRATCH_MAX_HEAD_DIST_MM",    50.0)

# 最低振動回数 / SCRATCH_WINDOW_FRAMES 窓。
# SCRATCH_WINDOW_FRAMES=30 (1秒) で 10 振動 = 10 Hz 相当。
# 文献の 10-15 Hz 範囲の下限に対応。
SCRATCH_MIN_OSCILLATIONS     = _get_env("SCRATCH_MIN_OSCILLATIONS",       10)

# 振動カウントの最小振幅 (mm)。0.75 mm 未満の変位はジッターとみなす。
# 旧 SCRATCH_MIN_OSC_AMPLITUDE_PX (= 3.0 px @ 0.25 mm/px = 0.75 mm) を mm 単位に変換。
# mm 単位にすることで、px-to-mm キャリブレーションが変わっても閾値の生物学的意味が保持される。
SCRATCH_MIN_OSC_AMPLITUDE_MM = _get_env("SCRATCH_MIN_OSC_AMPLITUDE_MM", 0.75)

# 後方互換: 旧 PX 名でも環境変数指定可能（mm に変換して使用）。
# 明示的に PX 値が指定された場合は PIXEL_TO_MM_RATIO で mm に変換して上書きする。
_OSC_AMP_PX_OVERRIDE = os.environ.get("SCRATCH_MIN_OSC_AMPLITUDE_PX")
SCRATCH_MIN_OSC_AMPLITUDE_PX = _get_env("SCRATCH_MIN_OSC_AMPLITUDE_PX",   3.0)  # px (legacy)

# 速度スムージング窓 (フレーム)。
SCRATCH_VELOCITY_SMOOTH_FRAMES = _get_env("SCRATCH_VELOCITY_SMOOTH_FRAMES",   5)

# DEPRECATED (pixel-based) — retained only for /api/config backward-compatibility
SCRATCH_MIN_VELOCITY  = _get_env("SCRATCH_MIN_VELOCITY",  30)   # px/frame
SCRATCH_MAX_HEAD_DIST = _get_env("SCRATCH_MAX_HEAD_DIST", 150)  # px

# ── Bout post-processing ──────────────────────────────────────────────────────
# BOUT_MIN_FRAMES: 最短 bout 継続フレーム数。
# 10 frames @ 30fps = 0.33 秒 — Akiyama et al. (2010) の典型 bout 下限に合致。
# 以前の 5 frames (0.17 秒) は単発の偽陽性を多く含む可能性があった。
BOUT_MIN_FRAMES   = _get_env("BOUT_MIN_FRAMES", 10)
# BOUT_MERGE_GAP: 隣接 bout を連結する最大ギャップ。
# 10 frames @ 30fps = 0.33 秒。これより短いギャップは「同一掻き行動の継続」とみなす。
BOUT_MERGE_GAP    = _get_env("BOUT_MERGE_GAP", 10)

# ── Camera / video defaults ───────────────────────────────────────────────────
PIXEL_TO_MM_RATIO  = _get_env("PIXEL_TO_MM_RATIO", 0.25)  # mm/pixel: 1px = 0.25mm

# 旧 SCRATCH_MIN_OSC_AMPLITUDE_PX を明示的に指定した場合、現在の PIXEL_TO_MM_RATIO で
# mm に変換して上書きする（後方互換）。
if _OSC_AMP_PX_OVERRIDE is not None:
    try:
        SCRATCH_MIN_OSC_AMPLITUDE_MM = float(_OSC_AMP_PX_OVERRIDE) * PIXEL_TO_MM_RATIO
        print(f"[config] [Info] SCRATCH_MIN_OSC_AMPLITUDE_PX={_OSC_AMP_PX_OVERRIDE} → "
              f"mm 換算 {SCRATCH_MIN_OSC_AMPLITUDE_MM:.3f} mm (PIXEL_TO_MM_RATIO={PIXEL_TO_MM_RATIO})")
    except (TypeError, ValueError):
        pass
DEFAULT_FPS        = _get_env("DEFAULT_FPS", 30)
# CAGE_MICE_COUNT: 0 = 動画から自動検出、1以上 = 固定頭数。デフォルト 0 (自動)
CAGE_MICE_COUNT    = _get_env("CAGE_MICE_COUNT", 0)
# 最大サポート個体数（GUI描画・タイムライン・データ構造の上限）
MAX_MICE_LIMIT     = _get_env("MAX_MICE_LIMIT", 8)

# ── 解剖学的整合性フィルタ ─────────────────────────────────────────────────────
# 体長チェック: neck-to-tail_base の距離（mm）が範囲外なら誤検出とみなす
# 実験室マウス (C57BL/6 成体) の典型体長は 80〜110 mm
BODY_LENGTH_MIN_MM    = _get_env("BODY_LENGTH_MIN_MM",   20.0)   # mm: これ未満は誤検出
BODY_LENGTH_MAX_MM    = _get_env("BODY_LENGTH_MAX_MM",  130.0)   # mm: これ超は誤検出
# テレポート検出: 連続フレーム間の重心移動量が「体長 × 倍率」を超えたら無効
MAX_TELEPORT_BODY_LENGTHS = _get_env("MAX_TELEPORT_BODY_LENGTHS", 2.0)
# 解剖学的チェックの有効/無効フラグ（デフォルト有効）
ENABLE_ANATOMICAL_CHECKS  = _get_env("ENABLE_ANATOMICAL_CHECKS", True)

# ── Paths (override via CLI or environment variables) ─────────────────────────
VIDEO_DIR   = _get_env("VIDEO_DIR", "data/videos/")
SLEAP_DIR   = _get_env("SLEAP_DIR", "data/sleap/")
OUTPUT_DIR  = _get_env("OUTPUT_DIR", "data/outputs/")
SIMBA_DIR   = _get_env("SIMBA_DIR", "data/simba_project/")

