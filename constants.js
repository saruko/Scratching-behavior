/**
 * Mouse Scratch Detection System - Shared Constants
 */
const KEYPOINTS_LIST = [
    "nose",
    "left_ear",
    "right_ear",
    "head_center",
    "neck",
    "left_fore_paw",
    "right_fore_paw",
    "spine_mid",
    "left_hind_paw",
    "right_hind_paw",
    "tail_base"
];

// Thresholds for scratch behavior classification.
// 値はバックエンド (config.py / feature_extraction.py) と一致させる。
// 実行時に /api/config から上書きされる（app.js）。
const SCRATCH_THRESHOLDS = {
    // ── 物理単位 (mm) — バックエンドの判定と同一基準 ──
    minVelocityMmS: 150.0,      // SCRATCH_MIN_VELOCITY_MM_S
    maxHeadDistMm: 50.0,        // SCRATCH_MAX_HEAD_DIST_MM
    minOscillations: 3,         // SCRATCH_MIN_OSCILLATIONS
    minOscAmplitudePx: 3.0,     // SCRATCH_MIN_OSC_AMPLITUDE_PX
    velocitySmoothFrames: 5,    // SCRATCH_VELOCITY_SMOOTH_FRAMES
    windowFrames: 15,           // SCRATCH_WINDOW_FRAMES
    pixelToMmRatio: 0.25,       // PIXEL_TO_MM_RATIO
    fps: 30,                    // DEFAULT_FPS
    interpolationMaxGap: 5,     // INTERPOLATION_MAX_GAP — 線形補間の最大ギャップ
    // ── Bout 後処理（フロント側手動補正後の bout 再構築用、バックエンドと同値）──
    boutMinFrames: 5,           // BOUT_MIN_FRAMES — Bout として認める最小フレーム数
    boutMergeGap:  10,          // BOUT_MERGE_GAP  — 隣接 Bout を結合する最大ギャップ
    // ── 後方互換 (px) — 旧表示用フォールバックのみ ──
    minVelocity: 30,
    maxHeadDist: 150
};

