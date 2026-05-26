"""
Feature Extraction and Rule-Based Scratch Classifier  (v2 — Academic Quality)
==============================================================================
全8項目の学術品質修正を適用済み:
  [1] スクラッチ判定を mm/s・mm 物理単位の閾値に切り替え
  [2] Bout 時刻に動画の実測 FPS を使用
  [3] scratch_prob を 3 条件のマージン比に基づく連続スコア [0,1] に変更
  [4] 速度に SCRATCH_VELOCITY_SMOOTH_FRAMES フレームの移動平均を適用
  [5] 振動カウントに SCRATCH_MIN_OSC_AMPLITUDE_PX の最小振幅閾値を導入
  [6] 信頼スコア閾値を下回るキーポイント座標を補間前に NaN でマスク
  [8] compute_summary_stats() で観察時間正規化済み統計量を出力
"""

import numpy as np
import pandas as pd
from collections import deque

from config import (
    KEYPOINTS,
    SCRATCH_WINDOW_FRAMES,
    SCRATCH_MIN_VELOCITY_MM_S,       # [1] mm/s
    SCRATCH_MAX_HEAD_DIST_MM,        # [1] mm
    SCRATCH_MIN_OSCILLATIONS,
    SCRATCH_MIN_OSC_AMPLITUDE_PX,    # [5] px
    SCRATCH_VELOCITY_SMOOTH_FRAMES,  # [4] frames
    BOUT_MIN_FRAMES,
    BOUT_MERGE_GAP,
    DEFAULT_FPS,
    INTERPOLATION_MAX_GAP,
    MIN_CONFIDENCE,
    HIND_PAW_MIN_CONF,
)


# ── ヘルパー関数 ──────────────────────────────────────────────────────────────

def compute_distance(p1_x, p1_y, p2_x, p2_y):
    """2点間のユークリッド距離。いずれかが NaN なら NaN を返す。"""
    if np.isnan(p1_x) or np.isnan(p1_y) or np.isnan(p2_x) or np.isnan(p2_y):
        return np.nan
    return float(np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2))


def _count_oscillations(history: list, min_amplitude_px: float) -> int:
    """
    [5] 座標履歴から方向転換回数を計算する。
    min_amplitude_px 未満の微小変位（姿勢推定ノイズ）は無視する。
    """
    count = 0
    last_d = 0.0
    for i in range(1, len(history)):
        prev_val, curr_val = history[i - 1], history[i]
        if np.isnan(prev_val) or np.isnan(curr_val):
            continue
        d = curr_val - prev_val
        if abs(d) < min_amplitude_px:
            continue  # ジッター無視 — last_d も更新しない
        if last_d != 0 and ((d > 0 and last_d < 0) or (d < 0 and last_d > 0)):
            count += 1
        last_d = d
    return count


def _compute_scratch_prob(speed_mm_s: float, dist_mm: float, osc: int,
                           speed_thresh: float, dist_thresh: float,
                           osc_thresh: int) -> float:
    """
    [3] 3 条件それぞれの閾値超過マージンに基づく連続スコア [0, 1]。

    - 条件が 1 つでも未達なら 0.0 を返す（flag=1 と完全対応）
    - 各条件が閾値のちょうど 2 倍に達したとき対応成分 = 1.0
    - 3 成分の算術平均を最終スコアとする

    速度成分: (v - v_thresh) / v_thresh  → 2× 閾値で 1.0
    距離成分: (d_thresh - d) / d_thresh  → 距離 0 で 1.0, 閾値で 0
    振動成分: (osc - osc_thresh) / osc_thresh → 2× 閾値で 1.0
    """
    if np.isnan(speed_mm_s) or speed_mm_s <= speed_thresh:
        return 0.0
    if np.isnan(dist_mm) or dist_mm >= dist_thresh:
        return 0.0
    if osc < osc_thresh:
        return 0.0

    s_speed = float(np.clip((speed_mm_s - speed_thresh) / speed_thresh, 0.0, 1.0))
    s_dist  = float(np.clip((dist_thresh  - dist_mm)    / dist_thresh,  0.0, 1.0))
    s_osc   = float(np.clip((osc - osc_thresh) / max(osc_thresh, 1),    0.0, 1.0))
    return float(np.clip((s_speed + s_dist + s_osc) / 3.0, 0.0, 1.0))


# ── サマリー統計量 ────────────────────────────────────────────────────────────

def compute_summary_stats(bouts_df: pd.DataFrame,
                           pose_df: pd.DataFrame,
                           fps: float) -> pd.DataFrame:
    """
    [8] 個体ごとの観察時間正規化済み統計量を算出する。

    出力カラム:
      mouse_id, observation_duration_s, observation_duration_min,
      n_bouts, total_scratch_duration_s,
      scratch_rate_bouts_per_min, scratch_rate_s_per_min,
      mean_bout_duration_s, sem_bout_duration_s, median_bout_duration_s,
      longest_bout_s, fps_used
    """
    if fps <= 0:
        fps = float(DEFAULT_FPS)

    # 観察時間はポーズデータのフレーム範囲から推定
    if pose_df.empty or "frame_idx" not in pose_df.columns:
        obs_s   = 0.0
    else:
        obs_frames = int(pose_df["frame_idx"].max() - pose_df["frame_idx"].min() + 1)
        obs_s      = obs_frames / fps
    obs_min = obs_s / 60.0

    mice = []
    if "mouse_id" in pose_df.columns:
        mice = sorted(pose_df["mouse_id"].dropna().unique())

    records = []
    for m_id in mice:
        mb = bouts_df[bouts_df["mouse_id"] == m_id] if not bouts_df.empty else pd.DataFrame()
        n      = len(mb)
        durs   = mb["bout_duration_s"].to_numpy(dtype=float) if n > 0 else np.array([])
        total  = float(durs.sum())   if n > 0 else 0.0
        mean   = float(durs.mean())  if n > 0 else np.nan
        sem    = float(durs.std() / np.sqrt(n)) if n > 1 else np.nan
        median = float(np.median(durs))    if n > 0 else np.nan
        longest = float(durs.max())        if n > 0 else np.nan

        records.append({
            "mouse_id":                  int(m_id),
            "observation_duration_s":    round(obs_s,   3),
            "observation_duration_min":  round(obs_min, 3),
            "n_bouts":                   n,
            "total_scratch_duration_s":  round(total,   3),
            "scratch_rate_bouts_per_min": round(n / obs_min,   3) if obs_min > 0 else 0.0,
            "scratch_rate_s_per_min":    round(total / obs_min, 3) if obs_min > 0 else 0.0,
            "mean_bout_duration_s":      round(mean,    3) if not np.isnan(mean)    else np.nan,
            "sem_bout_duration_s":       round(sem,     3) if not np.isnan(sem)     else np.nan,
            "median_bout_duration_s":    round(median,  3) if not np.isnan(median)  else np.nan,
            "longest_bout_s":            round(longest, 3) if not np.isnan(longest) else np.nan,
            "fps_used":                  fps,
        })

    return pd.DataFrame(records)


# ── メイン特徴量抽出関数 ──────────────────────────────────────────────────────

def extract_features_and_bouts(pose_df: pd.DataFrame,
                                pixel_to_mm_ratio: float = 0.25,
                                fps: float = None
                                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    トラッキング済みポーズデータから特徴量を算出し、スクラッチ Bout を検出する。

    Parameters
    ----------
    pose_df : DataFrame (wide フォーマット)
    pixel_to_mm_ratio : mm/px キャリブレーション比率
    fps : 動画の実測フレームレート [2]。None の場合 DEFAULT_FPS を使用。

    Returns
    -------
    (processed_df, bouts_df, summary_df)
      summary_df は compute_summary_stats() の出力。
    """
    pose_df = pose_df.copy()

    # [2] FPS の解決（環境変数 → DEFAULT_FPS の順にフォールバック）
    if fps is None:
        import os as _os
        try:
            fps = float(_os.environ.get("DEFAULT_FPS", DEFAULT_FPS))
        except ValueError:
            fps = float(DEFAULT_FPS)

    # 特徴量カラムの初期化（NaN — 盲目的ゼロ埋めを排除）
    feature_cols = [
        # px 単位（後方互換・GUI 表示用）
        "body_length_px", "head_to_nose_dist",
        "hind_to_head_dist_L", "hind_to_head_dist_R",
        "hind_to_ear_dist_L",  "hind_to_ear_dist_R",
        "hind_paw_speed_L",    "hind_paw_speed_R",
        # mm 単位（学術論文用）
        "body_length_mm", "head_to_nose_dist_mm",
        "hind_to_head_dist_L_mm", "hind_to_head_dist_R_mm",
        "hind_to_ear_dist_L_mm",  "hind_to_ear_dist_R_mm",
        "hind_paw_speed_L_mm_s",  "hind_paw_speed_R_mm_s",
        # [4] スムージング済み速度（スクラッチ判定に使用）
        "hind_paw_speed_L_mm_s_smooth", "hind_paw_speed_R_mm_s_smooth",
        # 振動
        "hind_osc_x_L", "hind_osc_y_L", "hind_osc_x_R", "hind_osc_y_R",
        # スクラッチ判定
        "scratch_flag",
        "scratch_prob_L",  # [3] 左肢の連続スコア
        "scratch_prob_R",  # [3] 右肢の連続スコア
        "scratch_prob",    # [3] max(L, R)
    ]
    for col in feature_cols:
        pose_df[col] = np.nan

    # 個体（track）ごとに処理
    for track_id, track_grp in pose_df.groupby("track"):
        sorted_grp = track_grp.sort_values("frame_idx").copy()

        # ── [6] 信頼スコアによるマスキング（補間前に適用） ──────────────────
        for kp in KEYPOINTS:
            conf_col = f"{kp}_conf"
            if conf_col not in sorted_grp.columns:
                continue
            thresh = HIND_PAW_MIN_CONF if kp in ("left_hind_paw", "right_hind_paw") else MIN_CONFIDENCE
            low_conf = sorted_grp[conf_col] < thresh
            sorted_grp.loc[low_conf, f"{kp}_x"] = np.nan
            sorted_grp.loc[low_conf, f"{kp}_y"] = np.nan

        # ── 線形補間（マスク後の NaN に適用） ───────────────────────────────
        coord_cols = [f"{kp}_{ax}" for kp in KEYPOINTS for ax in ("x", "y")]
        sorted_grp[coord_cols] = sorted_grp[coord_cols].interpolate(
            method="linear", limit=INTERPOLATION_MAX_GAP, limit_direction="forward"
        )

        indices = sorted_grp.index.tolist()

        # バッファの初期化
        feats = {col: [] for col in feature_cols}
        prev_x_L, prev_y_L, prev_f_L = None, None, None
        prev_x_R, prev_y_R, prev_f_R = None, None, None
        x_hist_L, y_hist_L = [], []
        x_hist_R, y_hist_R = [], []
        vel_L_buf = deque(maxlen=SCRATCH_VELOCITY_SMOOTH_FRAMES)  # [4]
        vel_R_buf = deque(maxlen=SCRATCH_VELOCITY_SMOOTH_FRAMES)  # [4]

        for idx in indices:
            row = sorted_grp.loc[idx]
            curr_f = row["frame_idx"]

            def get_xy(kp):
                return row.get(f"{kp}_x", np.nan), row.get(f"{kp}_y", np.nan)

            nose_x,   nose_y   = get_xy("nose")
            head_x,   head_y   = get_xy("head_center")
            ear_l_x,  ear_l_y  = get_xy("left_ear")
            ear_r_x,  ear_r_y  = get_xy("right_ear")
            neck_x,   neck_y   = get_xy("neck")
            tail_x,   tail_y   = get_xy("tail_base")
            hind_l_x, hind_l_y = get_xy("left_hind_paw")
            hind_r_x, hind_r_y = get_xy("right_hind_paw")

            # ── A. 距離特徴量 (px) ───────────────────────────────────────────
            body_length_px    = compute_distance(neck_x,   neck_y,   tail_x,   tail_y)
            head_to_nose_px   = compute_distance(nose_x,   nose_y,   head_x,   head_y)
            hind_to_head_L_px = compute_distance(hind_l_x, hind_l_y, head_x,   head_y)
            hind_to_head_R_px = compute_distance(hind_r_x, hind_r_y, head_x,   head_y)
            hind_to_ear_L_px  = compute_distance(hind_l_x, hind_l_y, ear_l_x,  ear_l_y)
            hind_to_ear_R_px  = compute_distance(hind_r_x, hind_r_y, ear_r_x,  ear_r_y)

            # ── B. フレーム間速度（px/frame, 欠損フレーム差を考慮） ──────────
            # 直近の「有効な」後肢座標との変位をフレーム差で割り、1 フレームあたりに
            # 正規化する。トラックが途中で欠損して数フレーム飛んでいる区間で、隣接行を
            # 1 フレーム差として扱い速度が過大評価されるのを防ぐ。欠損が補間限界
            # (INTERPOLATION_MAX_GAP) を超える区間は信頼できないため速度を NaN とする。
            if prev_x_L is not None and not np.isnan(hind_l_x):
                gap_L = curr_f - prev_f_L
                if 1 <= gap_L <= INTERPOLATION_MAX_GAP + 1:
                    speed_L_px = compute_distance(hind_l_x, hind_l_y, prev_x_L, prev_y_L) / gap_L
                else:
                    speed_L_px = np.nan
            else:
                speed_L_px = np.nan
            if not np.isnan(hind_l_x):
                prev_x_L, prev_y_L, prev_f_L = hind_l_x, hind_l_y, curr_f

            if prev_x_R is not None and not np.isnan(hind_r_x):
                gap_R = curr_f - prev_f_R
                if 1 <= gap_R <= INTERPOLATION_MAX_GAP + 1:
                    speed_R_px = compute_distance(hind_r_x, hind_r_y, prev_x_R, prev_y_R) / gap_R
                else:
                    speed_R_px = np.nan
            else:
                speed_R_px = np.nan
            if not np.isnan(hind_r_x):
                prev_x_R, prev_y_R, prev_f_R = hind_r_x, hind_r_y, curr_f

            # ── C. 物理単位換算 ──────────────────────────────────────────────
            def _mm(v_px): return v_px * pixel_to_mm_ratio if not np.isnan(v_px) else np.nan
            def _mm_s(v_px): return v_px * pixel_to_mm_ratio * fps if not np.isnan(v_px) else np.nan

            body_length_mm    = _mm(body_length_px)
            head_to_nose_mm   = _mm(head_to_nose_px)
            hind_to_head_L_mm = _mm(hind_to_head_L_px)
            hind_to_head_R_mm = _mm(hind_to_head_R_px)
            hind_to_ear_L_mm  = _mm(hind_to_ear_L_px)
            hind_to_ear_R_mm  = _mm(hind_to_ear_R_px)
            speed_L_mm_s      = _mm_s(speed_L_px)
            speed_R_mm_s      = _mm_s(speed_R_px)

            # ── D. [4] 速度スムージング（移動平均）─────────────────────────
            if not np.isnan(speed_L_mm_s):
                vel_L_buf.append(speed_L_mm_s)
            if not np.isnan(speed_R_mm_s):
                vel_R_buf.append(speed_R_mm_s)

            speed_L_smooth = float(np.mean(list(vel_L_buf))) if vel_L_buf else np.nan
            speed_R_smooth = float(np.mean(list(vel_R_buf))) if vel_R_buf else np.nan

            # ── E. [5] 振動数（最小振幅閾値付き）───────────────────────────
            x_hist_L.append(hind_l_x);  y_hist_L.append(hind_l_y)
            x_hist_R.append(hind_r_x);  y_hist_R.append(hind_r_y)
            if len(x_hist_L) > SCRATCH_WINDOW_FRAMES: x_hist_L.pop(0); y_hist_L.pop(0)
            if len(x_hist_R) > SCRATCH_WINDOW_FRAMES: x_hist_R.pop(0); y_hist_R.pop(0)

            osc_x_L = _count_oscillations(x_hist_L, SCRATCH_MIN_OSC_AMPLITUDE_PX)
            osc_y_L = _count_oscillations(y_hist_L, SCRATCH_MIN_OSC_AMPLITUDE_PX)
            osc_x_R = _count_oscillations(x_hist_R, SCRATCH_MIN_OSC_AMPLITUDE_PX)
            osc_y_R = _count_oscillations(y_hist_R, SCRATCH_MIN_OSC_AMPLITUDE_PX)
            osc_L   = max(osc_x_L, osc_y_L)
            osc_R   = max(osc_x_R, osc_y_R)

            # ── F. [1][3] スクラッチ判定（mm 単位・連続スコア） ─────────────
            prob_L = _compute_scratch_prob(
                speed_L_smooth, hind_to_ear_L_mm, osc_L,
                SCRATCH_MIN_VELOCITY_MM_S, SCRATCH_MAX_HEAD_DIST_MM, SCRATCH_MIN_OSCILLATIONS
            )
            prob_R = _compute_scratch_prob(
                speed_R_smooth, hind_to_ear_R_mm, osc_R,
                SCRATCH_MIN_VELOCITY_MM_S, SCRATCH_MAX_HEAD_DIST_MM, SCRATCH_MIN_OSCILLATIONS
            )
            scratch_prob = float(max(prob_L, prob_R))
            scratch_flag = 1.0 if scratch_prob > 0 else 0.0

            # ── 特徴量の蓄積 ────────────────────────────────────────────────
            feats["body_length_px"].append(body_length_px)
            feats["head_to_nose_dist"].append(head_to_nose_px)
            feats["hind_to_head_dist_L"].append(hind_to_head_L_px)
            feats["hind_to_head_dist_R"].append(hind_to_head_R_px)
            feats["hind_to_ear_dist_L"].append(hind_to_ear_L_px)
            feats["hind_to_ear_dist_R"].append(hind_to_ear_R_px)
            feats["hind_paw_speed_L"].append(speed_L_px)
            feats["hind_paw_speed_R"].append(speed_R_px)

            feats["body_length_mm"].append(body_length_mm)
            feats["head_to_nose_dist_mm"].append(head_to_nose_mm)
            feats["hind_to_head_dist_L_mm"].append(hind_to_head_L_mm)
            feats["hind_to_head_dist_R_mm"].append(hind_to_head_R_mm)
            feats["hind_to_ear_dist_L_mm"].append(hind_to_ear_L_mm)
            feats["hind_to_ear_dist_R_mm"].append(hind_to_ear_R_mm)
            feats["hind_paw_speed_L_mm_s"].append(speed_L_mm_s)
            feats["hind_paw_speed_R_mm_s"].append(speed_R_mm_s)
            feats["hind_paw_speed_L_mm_s_smooth"].append(speed_L_smooth)
            feats["hind_paw_speed_R_mm_s_smooth"].append(speed_R_smooth)

            feats["hind_osc_x_L"].append(osc_x_L)
            feats["hind_osc_y_L"].append(osc_y_L)
            feats["hind_osc_x_R"].append(osc_x_R)
            feats["hind_osc_y_R"].append(osc_y_R)

            feats["scratch_flag"].append(scratch_flag)
            feats["scratch_prob_L"].append(prob_L)
            feats["scratch_prob_R"].append(prob_R)
            feats["scratch_prob"].append(scratch_prob)

        # DataFrame への一括書き戻し
        for col, val_list in feats.items():
            sorted_grp[col] = val_list
        pose_df.loc[sorted_grp.index, sorted_grp.columns] = sorted_grp

    # ── Bout 検出 ────────────────────────────────────────────────────────────
    bouts_list = []

    if "mouse_id" not in pose_df.columns:
        print("[feature_extraction] [Warning] 'mouse_id' カラムが見つかりません。'track' から暫定IDを生成します。")
        pose_df["mouse_id"] = pose_df["track"].map(
            lambda t: int(t.split("_")[-1]) + 1 if isinstance(t, str) and t.split("_")[-1].isdigit() else 1
        )

    valid_mice_df = pose_df[pose_df["mouse_id"].notna() & (pose_df["mouse_id"] > 0)]

    if not valid_mice_df.empty:
        for m_id in sorted(valid_mice_df["mouse_id"].unique()):
            mouse_grp     = valid_mice_df[valid_mice_df["mouse_id"] == m_id].sort_values("frame_idx")
            scratch_frames = mouse_grp[mouse_grp["scratch_flag"] == 1]["frame_idx"].tolist()
            if not scratch_frames:
                continue

            bouts = []
            start_f = scratch_frames[0]
            last_f  = scratch_frames[0]
            for f in scratch_frames[1:]:
                if f - last_f <= BOUT_MERGE_GAP:
                    last_f = f
                else:
                    if last_f - start_f + 1 >= BOUT_MIN_FRAMES:
                        bouts.append((start_f, last_f))
                    start_f = f
                    last_f  = f
            if last_f - start_f + 1 >= BOUT_MIN_FRAMES:
                bouts.append((start_f, last_f))

            for b_idx, (sf, ef) in enumerate(bouts, 1):
                n_f = ef - sf + 1
                # [2] 実測 fps で時刻を計算
                bouts_list.append({
                    "mouse_id":          int(m_id),
                    "bout_id":           b_idx,
                    "bout_start_frame":  int(sf),
                    "bout_end_frame":    int(ef),
                    "bout_start_time_s": round(sf / fps, 4),
                    "bout_end_time_s":   round(ef / fps, 4),
                    "bout_duration_s":   round(n_f / fps, 4),
                    "bout_n_frames":     int(n_f),
                    "fps_used":          fps,
                })

    bouts_df = pd.DataFrame(bouts_list) if bouts_list else pd.DataFrame(columns=[
        "mouse_id", "bout_id", "bout_start_frame", "bout_end_frame",
        "bout_start_time_s", "bout_end_time_s", "bout_duration_s", "bout_n_frames", "fps_used"
    ])

    # [8] サマリー統計量
    summary_df = compute_summary_stats(bouts_df, pose_df, fps)

    return pose_df, bouts_df, summary_df
