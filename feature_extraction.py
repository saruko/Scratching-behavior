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

import os
import numpy as np
import pandas as pd
from collections import deque

# ── ROI（関心領域）読み込み — pipeline.py が環境変数経由で渡す ─────────
# 全 4 値が揃った場合のみ ROI フィルタを有効化（部分指定は無効）。
def _load_roi_from_env():
    try:
        x1 = os.environ.get("ROI_X1"); y1 = os.environ.get("ROI_Y1")
        x2 = os.environ.get("ROI_X2"); y2 = os.environ.get("ROI_Y2")
        if all(v is not None and v != "" for v in (x1, y1, x2, y2)):
            return (int(x1), int(y1), int(x2), int(y2))
    except Exception:
        return None
    return None
ROI_BOX = _load_roi_from_env()


def _is_inside_roi(center_px) -> bool:
    """ROI 矩形内に重心が含まれるかを判定。ROI 未設定時は常に True。"""
    if ROI_BOX is None or center_px is None:
        return True
    x1, y1, x2, y2 = ROI_BOX
    cx, cy = center_px
    if cx is None or cy is None or np.isnan(cx) or np.isnan(cy):
        return True  # 座標が無い場合はフィルタしない（NaN 補間に委ねる）
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

from config import (
    KEYPOINTS,
    SCRATCH_WINDOW_FRAMES,
    SCRATCH_MIN_VELOCITY_MM_S,       # [1] mm/s
    SCRATCH_MAX_HEAD_DIST_MM,        # [1] mm
    SCRATCH_MIN_OSCILLATIONS,
    SCRATCH_MIN_OSC_AMPLITUDE_MM,    # [5] mm (キャリブレーション独立)
    SCRATCH_VELOCITY_SMOOTH_FRAMES,  # [4] frames
    BOUT_MIN_FRAMES,
    BOUT_MERGE_GAP,
    DEFAULT_FPS,
    INTERPOLATION_MAX_GAP,
    MIN_CONFIDENCE,
    HIND_PAW_MIN_CONF,
    BODY_LENGTH_MIN_MM,
    BODY_LENGTH_MAX_MM,
    MAX_TELEPORT_BODY_LENGTHS,
    ENABLE_ANATOMICAL_CHECKS,
    PIXEL_TO_MM_RATIO,
)


# ── ヘルパー関数 ──────────────────────────────────────────────────────────────

def compute_distance(p1_x, p1_y, p2_x, p2_y):
    """2点間のユークリッド距離。いずれかが NaN なら NaN を返す。"""
    if np.isnan(p1_x) or np.isnan(p1_y) or np.isnan(p2_x) or np.isnan(p2_y):
        return np.nan
    return float(np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2))


def _count_oscillations(history, min_amplitude_px: float) -> int:
    """
    [5] 座標履歴から方向転換回数を計算する。
    min_amplitude_px 未満の微小変位（姿勢推定ノイズ）は無視する。

    NOTE: 位置インデックス (history[i]) を使わず逐次走査するため、list / deque
    いずれを渡しても O(N)。deque への添字アクセスは O(N) であり、添字ループだと
    全体が O(N^2) になるのを避ける意図。
    """
    count = 0
    last_d = 0.0
    prev_val = None
    for curr_val in history:
        if prev_val is None:
            prev_val = curr_val
            continue
        if np.isnan(prev_val) or np.isnan(curr_val):
            prev_val = curr_val
            continue
        d = curr_val - prev_val
        prev_val = curr_val
        if abs(d) < min_amplitude_px:
            continue  # ジッター無視 — last_d も更新しない
        if last_d != 0 and ((d > 0 and last_d < 0) or (d < 0 and last_d > 0)):
            count += 1
        last_d = d
    return count


def _validate_detection(
    body_length_px: float,
    pixel_to_mm_ratio: float,
    prev_center: tuple | None = None,
    curr_center: tuple | None = None,
    body_length_px_for_teleport: float | None = None,
) -> tuple[bool, str]:
    """
    解剖学的整合性チェック — 誤検出除外フィルタ。

    以下の 2 条件のいずれかを満たさない場合は無効（False）とする:

    [A] 体長チェック:
        neck-to-tail_base 距離（mm）が [BODY_LENGTH_MIN_MM, BODY_LENGTH_MAX_MM] 範囲内か。
        範囲外は骨格推定の破綻（複数個体の骨格混合など）とみなす。

    [B] テレポート検出:
        直前フレームからの重心（spine_mid 等）移動量が、体長の MAX_TELEPORT_BODY_LENGTHS 倍を
        超える場合、物理的に不可能な動きとして除外する。
        （注: 補間後のデータには適用しないこと — 補間座標は滑らかなため誤除外の原因となる）

    Parameters
    ----------
    body_length_px              : neck-to-tail_base のピクセル距離
    pixel_to_mm_ratio           : mm/px キャリブレーション値
    prev_center                 : 前フレームの重心座標 (x, y) px, または None
    curr_center                 : 現フレームの重心座標 (x, y) px, または None
    body_length_px_for_teleport : テレポート判定に使う体長 (px)。None なら body_length_px 使用

    Returns
    -------
    (is_valid: bool, reason: str)
    """
    if not ENABLE_ANATOMICAL_CHECKS:
        return True, "checks_disabled"

    # [A] 体長チェック（mm 換算）
    if not np.isnan(body_length_px):
        body_mm = body_length_px * pixel_to_mm_ratio
        if body_mm < BODY_LENGTH_MIN_MM:
            return False, f"body_too_short ({body_mm:.1f} mm < {BODY_LENGTH_MIN_MM} mm)"
        if body_mm > BODY_LENGTH_MAX_MM:
            return False, f"body_too_long ({body_mm:.1f} mm > {BODY_LENGTH_MAX_MM} mm)"

    # [B] テレポートチェック
    if prev_center is not None and curr_center is not None:
        ref_len = body_length_px_for_teleport if body_length_px_for_teleport is not None else body_length_px
        if not np.isnan(ref_len) and ref_len > 0:
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            move_px = float(np.sqrt(dx**2 + dy**2))
            max_move_px = ref_len * MAX_TELEPORT_BODY_LENGTHS
            if move_px > max_move_px:
                return False, (f"teleport ({move_px:.1f} px > {max_move_px:.1f} px "
                               f"= body_len × {MAX_TELEPORT_BODY_LENGTHS})")

    return True, "ok"


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
        # mouse_id=0 は「誤検出（除外）」の特別値。サマリ集計から完全に除外する
        # （解析対象の判定 valid_mice_df の mouse_id > 0 フィルタと一貫させる）。
        mice = sorted(m for m in pose_df["mouse_id"].dropna().unique() if m > 0)

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
        x_hist_L = deque(maxlen=SCRATCH_WINDOW_FRAMES)
        y_hist_L = deque(maxlen=SCRATCH_WINDOW_FRAMES)
        x_hist_R = deque(maxlen=SCRATCH_WINDOW_FRAMES)
        y_hist_R = deque(maxlen=SCRATCH_WINDOW_FRAMES)
        vel_L_buf = deque(maxlen=SCRATCH_VELOCITY_SMOOTH_FRAMES)  # [4]
        vel_R_buf = deque(maxlen=SCRATCH_VELOCITY_SMOOTH_FRAMES)  # [4]
        # テレポート検出用: 前フレームの重心座標 (neck と spine_mid の中点)
        prev_center_px: tuple | None = None
        prev_body_len_px: float = np.nan

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

            # ── [解剖学チェック] 体長・テレポート検出 ──────────────────────────
            # 現フレームの重心 = neck と spine_mid の中点（両方が有効な場合のみ）
            curr_center_px: tuple | None = None
            if not (np.isnan(neck_x) or np.isnan(neck_y) or
                    np.isnan(row.get("spine_mid_x", np.nan)) or np.isnan(row.get("spine_mid_y", np.nan))):
                curr_center_px = (
                    (neck_x + row.get("spine_mid_x", neck_x)) / 2.0,
                    (neck_y + row.get("spine_mid_y", neck_y)) / 2.0,
                )

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
            # NOTE: SCRATCH_MIN_OSC_AMPLITUDE_MM (mm) を、現在のキャリブレーションで
            # px 換算してから _count_oscillations に渡す。これにより、カメラ設置高さや
            # 拡大率が異なる実験間でも、生物学的に等価な振幅閾値が適用される。
            _min_amp_px = SCRATCH_MIN_OSC_AMPLITUDE_MM / pixel_to_mm_ratio if pixel_to_mm_ratio > 0 else 3.0

            # deque(maxlen=SCRATCH_WINDOW_FRAMES) なので古い要素は自動破棄される
            x_hist_L.append(hind_l_x);  y_hist_L.append(hind_l_y)
            x_hist_R.append(hind_r_x);  y_hist_R.append(hind_r_y)

            osc_x_L = _count_oscillations(x_hist_L, _min_amp_px)
            osc_y_L = _count_oscillations(y_hist_L, _min_amp_px)
            osc_x_R = _count_oscillations(x_hist_R, _min_amp_px)
            osc_y_R = _count_oscillations(y_hist_R, _min_amp_px)
            osc_L   = max(osc_x_L, osc_y_L)
            osc_R   = max(osc_x_R, osc_y_R)

            # ── [ROI チェック] 関心領域外の検出は判定対象外 ───────────────────
            # ROI 未設定時は常に通過。設定時は重心 (curr_center_px) が矩形内かを評価。
            roi_ok = _is_inside_roi(curr_center_px)

            # ── [解剖学チェック] 有効性検証 ────────────────────────────────────
            is_valid, invalid_reason = _validate_detection(
                body_length_px=body_length_px,
                pixel_to_mm_ratio=pixel_to_mm_ratio,
                prev_center=prev_center_px,
                curr_center=curr_center_px,
                body_length_px_for_teleport=prev_body_len_px,
            )
            if not roi_ok:
                is_valid = False
                invalid_reason = "outside_roi"
            if not is_valid:
                # 無効フレームは scratch_flag=0, prob=0 として扱い、速度バッファも更新しない
                # 座標自体は NaN にしない（視覚化には使用するが、判定からは除外）
                feats["scratch_flag"].append(0.0)
                feats["scratch_prob_L"].append(0.0)
                feats["scratch_prob_R"].append(0.0)
                feats["scratch_prob"].append(0.0)
                for col in feature_cols:
                    if col not in ("scratch_flag", "scratch_prob_L", "scratch_prob_R", "scratch_prob"):
                        feats[col].append(np.nan)
                # テレポート等では前フレーム座標を引き継がない（追跡断絶とみなす）
                if "teleport" in invalid_reason:
                    prev_x_L, prev_y_L, prev_f_L = None, None, None
                    prev_x_R, prev_y_R, prev_f_R = None, None, None
                    prev_center_px = None
                    prev_body_len_px = np.nan
                continue  # 次のフレームへ

            # 次フレームのテレポート検出のために現フレーム情報を保存
            prev_center_px = curr_center_px
            prev_body_len_px = body_length_px if not np.isnan(body_length_px) else prev_body_len_px

            # ── F. [1][3] スクラッチ判定（mm 単位・連続スコア） ─────────────
            # マウスの掻き動作は耳だけでなく頭部全体（顔・首）に向かうため、
            # ear と head_center の両方からの最近接距離を判定に使用する。
            # min(ear, head_center) を採ることで、顔面掻き行動も漏れなく検出する。
            def _safe_min(a, b):
                if np.isnan(a) and np.isnan(b): return np.nan
                if np.isnan(a): return b
                if np.isnan(b): return a
                return min(a, b)

            head_dist_L_mm = _safe_min(hind_to_ear_L_mm, hind_to_head_L_mm)
            head_dist_R_mm = _safe_min(hind_to_ear_R_mm, hind_to_head_R_mm)

            prob_L = _compute_scratch_prob(
                speed_L_smooth, head_dist_L_mm, osc_L,
                SCRATCH_MIN_VELOCITY_MM_S, SCRATCH_MAX_HEAD_DIST_MM, SCRATCH_MIN_OSCILLATIONS
            )
            prob_R = _compute_scratch_prob(
                speed_R_smooth, head_dist_R_mm, osc_R,
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
