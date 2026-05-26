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
    // 文献値に基づく: Kuraishi 1995 (~10-15 Hz), Akiyama 2010 (bout 0.3-3秒)
    minVelocityMmS: 150.0,      // SCRATCH_MIN_VELOCITY_MM_S
    maxHeadDistMm: 50.0,        // SCRATCH_MAX_HEAD_DIST_MM
    minOscillations: 10,        // SCRATCH_MIN_OSCILLATIONS — 30フレーム窓で10振動 = 10Hz相当
    minOscAmplitudeMm: 0.75,    // SCRATCH_MIN_OSC_AMPLITUDE_MM — キャリブレーション独立
    velocitySmoothFrames: 5,    // SCRATCH_VELOCITY_SMOOTH_FRAMES
    windowFrames: 30,           // SCRATCH_WINDOW_FRAMES — 30フレーム = 1秒（10Hz検出に十分）
    pixelToMmRatio: 0.25,       // PIXEL_TO_MM_RATIO
    fps: 30,                    // DEFAULT_FPS
    interpolationMaxGap: 5,     // INTERPOLATION_MAX_GAP — 線形補間の最大ギャップ
    // ── Bout 後処理（フロント側手動補正後の bout 再構築用、バックエンドと同値）──
    boutMinFrames: 10,          // BOUT_MIN_FRAMES — 0.33秒 (Akiyama 2010)
    boutMergeGap:  10,          // BOUT_MERGE_GAP  — 隣接 Bout を結合する最大ギャップ
    // ── 後方互換 (px) — 旧表示用フォールバックのみ ──
    minOscAmplitudePx: 3.0,     // 旧 PX 名（参考値）
    minVelocity: 30,
    maxHeadDist: 150
};

/**
 * マウス個体カラーパレット（最大8匹対応）
 * バックエンドの config.py MOUSE_COLORS と対応させること。
 * ID 1-4: マーカー（尾部塗装）色
 * ID 5:   白色マウス（無マーカー）
 * ID 6-8: 拡張スロット
 */
const MOUSE_COLOR_PALETTE = {
    1: { hex: "#EF4444", name: "赤",   cssVar: "--color-mouse-1", glowRgb: "239, 68, 68"   },
    2: { hex: "#3B82F6", name: "青",   cssVar: "--color-mouse-2", glowRgb: "59, 130, 246"  },
    3: { hex: "#EAB308", name: "黄",   cssVar: "--color-mouse-3", glowRgb: "234, 179, 8"   },
    4: { hex: "#9CA3AF", name: "黒",   cssVar: "--color-mouse-4", glowRgb: "156, 163, 175" },
    5: { hex: "#F0F0F0", name: "白",   cssVar: "--color-mouse-5", glowRgb: "240, 240, 240" },
    6: { hex: "#F97316", name: "橙",   cssVar: "--color-mouse-6", glowRgb: "249, 115, 22"  },
    7: { hex: "#22C55E", name: "緑",   cssVar: "--color-mouse-7", glowRgb: "34, 197, 94"   },
    8: { hex: "#A855F7", name: "紫",   cssVar: "--color-mouse-8", glowRgb: "168, 85, 247"  },
};

/** 最大サポート個体数 */
const MAX_MICE_LIMIT = 8;

