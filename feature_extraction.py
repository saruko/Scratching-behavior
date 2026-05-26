"""
Feature Extraction and Rule-Based Scratch Classifier (Revised)
=============================================================
マウスの骨格関節座標からキネマティクス特徴量を算出し、ひっかき行動の自動判定
および一連の行動（Bout）の抽出を行います。
本改定版では、線形補間、NaNの適正伝搬、およびミリメートル（mm）物理単位への変換を実装しています。
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from config import (
    KEYPOINTS,
    SCRATCH_WINDOW_FRAMES,
    SCRATCH_MIN_VELOCITY,
    SCRATCH_MAX_HEAD_DIST,
    SCRATCH_MIN_OSCILLATIONS,
    BOUT_MIN_FRAMES,
    BOUT_MERGE_GAP,
    DEFAULT_FPS,
    INTERPOLATION_MAX_GAP
)

def compute_distance(p1_x, p1_y, p2_x, p2_y):
    """2点間のユークリッド距離を計算し、浮動小数点数（またはNaN）を返します。"""
    if np.isnan(p1_x) or np.isnan(p1_y) or np.isnan(p2_x) or np.isnan(p2_y):
        return np.nan
    return np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

def extract_features_and_bouts(pose_df: pd.DataFrame, pixel_to_mm_ratio: float = 0.25, fps: float = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    トラッキング済みのポーズデータから各種キネマティクス特徴量を計算し、
    ルールベース分類器によってスクラッチ判定（scratch_flag, scratch_prob）を行い、
    さらに Bout 情報を集計したデータフレームを出力します。
    """
    pose_df = pose_df.copy()
    
    # 動的なFPSの伝搬 (Bug 11対策)
    if fps is None:
        import os
        from config import DEFAULT_FPS as CFG_FPS
        try:
            fps = float(os.environ.get("DEFAULT_FPS", CFG_FPS))
        except ValueError:
            fps = float(CFG_FPS)

    
    # 欠落している特徴量カラムを初期化 (px と mm の両方を保持)
    feature_cols = [
        # ピクセル単位特徴量 (後方互換・GUI表示用)
        "body_length_px", "head_to_nose_dist",
        "hind_to_head_dist_L", "hind_to_head_dist_R",
        "hind_to_ear_dist_L", "hind_to_ear_dist_R",
        "hind_paw_speed_L", "hind_paw_speed_R",
        
        # 学術論文用物理単位特徴量 (mm / mm_s)
        "body_length_mm", "head_to_nose_dist_mm",
        "hind_to_head_dist_L_mm", "hind_to_head_dist_R_mm",
        "hind_to_ear_dist_L_mm", "hind_to_ear_dist_R_mm",
        "hind_paw_speed_L_mm_s", "hind_paw_speed_R_mm_s",
        
        "hind_osc_x_L", "hind_osc_y_L", "hind_osc_x_R", "hind_osc_y_R",
        "scratch_flag", "scratch_prob"
    ]
    for col in feature_cols:
        pose_df[col] = np.nan  # NaNで初期化し、伝搬を適正化

    # 個体（track）ごとに時系列ソートして計算
    grouped = pose_df.groupby("track")
    
    for track_id, track_grp in grouped:
        # 時系列順にソート
        sorted_grp = track_grp.sort_values("frame_idx").copy()
        
        # 1. 座標の短い欠損（オクルージョン）に対する線形補間を適用 (学術精度向上)
        coord_cols = []
        for kp in KEYPOINTS:
            coord_cols.extend([f"{kp}_x", f"{kp}_y"])
            
        # 線形補間の実行 (最大5フレームのギャップまで補間)
        sorted_grp[coord_cols] = sorted_grp[coord_cols].interpolate(
            method="linear", limit=INTERPOLATION_MAX_GAP, limit_direction="forward"
        )
        
        indices = sorted_grp.index.tolist()
        
        # Accumulators for batch updates (B-5)
        feats = {
            "body_length_px": [], "head_to_nose_dist": [], "hind_to_head_dist_L": [], "hind_to_head_dist_R": [],
            "hind_to_ear_dist_L": [], "hind_to_ear_dist_R": [], "hind_paw_speed_L": [], "hind_paw_speed_R": [],
            "body_length_mm": [], "head_to_nose_dist_mm": [], "hind_to_head_dist_L_mm": [], "hind_to_head_dist_R_mm": [],
            "hind_to_ear_dist_L_mm": [], "hind_to_ear_dist_R_mm": [], "hind_paw_speed_L_mm_s": [], "hind_paw_speed_R_mm_s": [],
            "hind_osc_x_L": [], "hind_osc_y_L": [], "hind_osc_x_R": [], "hind_osc_y_R": [],
            "scratch_flag": [], "scratch_prob": []
        }
        
        # 前フレームの座標保持用
        prev_x_L, prev_y_L = None, None
        prev_x_R, prev_y_R = None, None
        
        # 振動数計算用のx/y座標履歴（最大15フレーム）
        x_history_L = []
        x_history_R = []
        y_history_L = []
        y_history_R = []
        
        for idx in indices:
            row = sorted_grp.loc[idx]
            
            def get_xy(kp):
                return row.get(f"{kp}_x", np.nan), row.get(f"{kp}_y", np.nan)
            
            nose_x, nose_y = get_xy("nose")
            head_x, head_y = get_xy("head_center")
            ear_l_x, ear_l_y = get_xy("left_ear")
            ear_r_x, ear_r_y = get_xy("right_ear")
            neck_x, neck_y = get_xy("neck")
            tail_x, tail_y = get_xy("tail_base")
            hind_l_x, hind_l_y = get_xy("left_hind_paw")
            hind_r_x, hind_r_y = get_xy("right_hind_paw")
            
            # --- A. 距離特徴量の計算 (ピクセル単位) ---
            body_length_px = compute_distance(neck_x, neck_y, tail_x, tail_y)
            head_to_nose_px = compute_distance(nose_x, nose_y, head_x, head_y)
            
            hind_to_head_L_px = compute_distance(hind_l_x, hind_l_y, head_x, head_y)
            hind_to_head_R_px = compute_distance(hind_r_x, hind_r_y, head_x, head_y)
            
            hind_to_ear_L_px = compute_distance(hind_l_x, hind_l_y, ear_l_x, ear_l_y)
            hind_to_ear_R_px = compute_distance(hind_r_x, hind_r_y, ear_r_x, ear_r_y)
            
            # --- B. Speed calculation (displacement, px/frame) ---
            if prev_x_L is not None and not np.isnan(hind_l_x) and not np.isnan(prev_x_L):
                speed_L_px = compute_distance(hind_l_x, hind_l_y, prev_x_L, prev_y_L)
            else:
                speed_L_px = np.nan
            prev_x_L, prev_y_L = hind_l_x, hind_l_y

            if prev_x_R is not None and not np.isnan(hind_r_x) and not np.isnan(prev_x_R):
                speed_R_px = compute_distance(hind_r_x, hind_r_y, prev_x_R, prev_y_R)
            else:
                speed_R_px = np.nan
            prev_x_R, prev_y_R = hind_r_x, hind_r_y

            # --- C. 振動数の計算 (15フレームのローリング窓) ---
            # 履歴の追加 (x軸・y軸を同じタイミングで同期して追加: Bug 2 対策)
            x_history_L.append(hind_l_x)
            y_history_L.append(hind_l_y)
            if len(x_history_L) > SCRATCH_WINDOW_FRAMES:
                x_history_L.pop(0)
            if len(y_history_L) > SCRATCH_WINDOW_FRAMES:
                y_history_L.pop(0)
                
            x_history_R.append(hind_r_x)
            y_history_R.append(hind_r_y)
            if len(x_history_R) > SCRATCH_WINDOW_FRAMES:
                x_history_R.pop(0)
            if len(y_history_R) > SCRATCH_WINDOW_FRAMES:
                y_history_R.pop(0)
                
            # 左肢のx軸振動数計算 (反転数)
            osc_L = 0
            last_dx_L = 0
            for i in range(1, len(x_history_L)):
                p_x = x_history_L[i-1]
                c_x = x_history_L[i]
                if np.isnan(p_x) or np.isnan(c_x):
                    continue
                dx = c_x - p_x
                if dx != 0:
                    if last_dx_L != 0 and ((dx > 0 and last_dx_L < 0) or (dx < 0 and last_dx_L > 0)):
                        osc_L += 1
                    last_dx_L = dx
                    
            # 右肢のx軸振動数計算 (反転数)
            osc_R = 0
            last_dx_R = 0
            for i in range(1, len(x_history_R)):
                p_x = x_history_R[i-1]
                c_x = x_history_R[i]
                if np.isnan(p_x) or np.isnan(c_x):
                    continue
                dx = c_x - p_x
                if dx != 0:
                    if last_dx_R != 0 and ((dx > 0 and last_dx_R < 0) or (dx < 0 and last_dx_R > 0)):
                        osc_R += 1
                    last_dx_R = dx

            # 左肢のy軸振動数計算 (反転数)
            osc_y_L = 0
            last_dy_L = 0
            for i in range(1, len(y_history_L)):
                p_y = y_history_L[i-1]
                c_y = y_history_L[i]
                if np.isnan(p_y) or np.isnan(c_y):
                    continue
                dy = c_y - p_y
                if dy != 0:
                    if last_dy_L != 0 and ((dy > 0 and last_dy_L < 0) or (dy < 0 and last_dy_L > 0)):
                        osc_y_L += 1
                    last_dy_L = dy
            
            # 右肢のy軸振動数計算 (反転数)
            osc_y_R = 0
            last_dy_R = 0
            for i in range(1, len(y_history_R)):
                p_y = y_history_R[i-1]
                c_y = y_history_R[i]
                if np.isnan(p_y) or np.isnan(c_y):
                    continue
                dy = c_y - p_y
                if dy != 0:
                    if last_dy_R != 0 and ((dy > 0 and last_dy_R < 0) or (dy < 0 and last_dy_R > 0)):
                        osc_y_R += 1
                    last_dy_R = dy
            
            feats["hind_osc_x_L"].append(osc_L)
            feats["hind_osc_y_L"].append(osc_y_L)
            feats["hind_osc_x_R"].append(osc_R)
            feats["hind_osc_y_R"].append(osc_y_R)
            
            # --- D. 物理単位への換算 ---
            # 速度: mm/sec = (px/frame) * (mm/px) * (frames/sec)  (Bug 11 対策: 動的 fps を使用)
            speed_L_mm_s = speed_L_px * pixel_to_mm_ratio * fps if not np.isnan(speed_L_px) else np.nan
            speed_R_mm_s = speed_R_px * pixel_to_mm_ratio * fps if not np.isnan(speed_R_px) else np.nan
            
            # 距離: mm = px * (mm/px)
            body_length_mm = body_length_px * pixel_to_mm_ratio if not np.isnan(body_length_px) else np.nan
            head_to_nose_mm = head_to_nose_px * pixel_to_mm_ratio if not np.isnan(head_to_nose_px) else np.nan
            hind_to_head_L_mm = hind_to_head_L_px * pixel_to_mm_ratio if not np.isnan(hind_to_head_L_px) else np.nan
            hind_to_head_R_mm = hind_to_head_R_px * pixel_to_mm_ratio if not np.isnan(hind_to_head_R_px) else np.nan
            hind_to_ear_L_mm = hind_to_ear_L_px * pixel_to_mm_ratio if not np.isnan(hind_to_ear_L_px) else np.nan
            hind_to_ear_R_mm = hind_to_ear_R_px * pixel_to_mm_ratio if not np.isnan(hind_to_ear_R_px) else np.nan
            
            # 元の座標および補間済み座標の書き戻しは、ループ後に DataFrame 一括更新 (B-5) で行うため、ここではリスト蓄積のみ行う
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
            
            # --- E. ルールベースによるひっかき判定 (pxの既存閾値をそのまま適用) ---
            # 学術精度向上 (Bug 10対策): X軸方向の往復運動だけでなく、Y軸（上下）方向の往復運動もひっかき特有の高速往復（oscillation）に含む
            scratch_L = (not np.isnan(speed_L_px) and speed_L_px > SCRATCH_MIN_VELOCITY and
                         not np.isnan(hind_to_ear_L_px) and hind_to_ear_L_px < SCRATCH_MAX_HEAD_DIST and
                         max(osc_L, osc_y_L) >= SCRATCH_MIN_OSCILLATIONS)
                          
            scratch_R = (not np.isnan(speed_R_px) and speed_R_px > SCRATCH_MIN_VELOCITY and
                         not np.isnan(hind_to_ear_R_px) and hind_to_ear_R_px < SCRATCH_MAX_HEAD_DIST and
                         max(osc_R, osc_y_R) >= SCRATCH_MIN_OSCILLATIONS)
                          
            if scratch_L or scratch_R:
                feats["scratch_flag"].append(1.0)
                feats["scratch_prob"].append(0.95)
            else:
                feats["scratch_flag"].append(0.0)
                feats["scratch_prob"].append(0.0)
        
        # Assign all accumulated lists back to sorted_grp in batch (B-5)
        for col, val_list in feats.items():
            sorted_grp[col] = val_list
            
        # Write back the entire sorted_grp (including interpolated coordinates and features) in batch
        pose_df.loc[sorted_grp.index, sorted_grp.columns] = sorted_grp

    # --- F. Bout (ひっかき一連動作) の統合処理 ---
    bouts_list = []
    
    # mouse_id カラムが存在しない場合は track から暫定IDを生成
    if "mouse_id" not in pose_df.columns:
        print("[feature_extraction] [Warning] 'mouse_id' カラムが見つかりません。'track' カラムから暫定IDを生成します。")
        pose_df["mouse_id"] = pose_df["track"].map(
            lambda t: int(t.split("_")[-1]) + 1 if isinstance(t, str) and t.split("_")[-1].isdigit() else 1
        )
    
    # 有効な個体IDが存在する行のみ対象 (mouse_id > 0)
    valid_mice_df = pose_df[pose_df["mouse_id"].notna() & (pose_df["mouse_id"] > 0)]
    
    if not valid_mice_df.empty:
        unique_mice = sorted(valid_mice_df["mouse_id"].unique())
        
        for m_id in unique_mice:
            mouse_grp = valid_mice_df[valid_mice_df["mouse_id"] == m_id].sort_values("frame_idx")
            
            # フラグが1のフレームインデックスを抽出
            scratch_frames = mouse_grp[mouse_grp["scratch_flag"] == 1]["frame_idx"].tolist()
            if not scratch_frames:
                continue
                
            # 連続区間の検出とマージ
            bouts = []
            if len(scratch_frames) > 0:
                start_f = scratch_frames[0]
                last_f = scratch_frames[0]
                
                for f in scratch_frames[1:]:
                    if f - last_f <= BOUT_MERGE_GAP:
                        last_f = f
                    else:
                        duration_f = last_f - start_f + 1
                        if duration_f >= BOUT_MIN_FRAMES:
                            bouts.append((start_f, last_f))
                        start_f = f
                        last_f = f
                
                # 最後の区間
                duration_f = last_f - start_f + 1
                if duration_f >= BOUT_MIN_FRAMES:
                    bouts.append((start_f, last_f))

            # Bout 情報の作成
            for b_idx, (start_f, end_f) in enumerate(bouts, 1):
                n_frames = end_f - start_f + 1
                bouts_list.append({
                    "mouse_id": int(m_id),
                    "bout_id": b_idx,
                    "bout_start_frame": int(start_f),
                    "bout_end_frame": int(end_f),
                    "bout_start_time_s": float(start_f / DEFAULT_FPS),
                    "bout_end_time_s": float(end_f / DEFAULT_FPS),
                    "bout_duration_s": float(n_frames / DEFAULT_FPS),
                    "bout_n_frames": int(n_frames)
                })

    bouts_df = pd.DataFrame(bouts_list)
    if bouts_df.empty:
        bouts_df = pd.DataFrame(columns=[
            "mouse_id", "bout_id", "bout_start_frame", "bout_end_frame",
            "bout_start_time_s", "bout_end_time_s", "bout_duration_s", "bout_n_frames"
        ])
        
    return pose_df, bouts_df
