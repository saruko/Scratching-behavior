"""
Feature Extraction and Rule-Based Scratch Classifier
===================================================
マウスの骨格関節座標からキネマティクス特徴量を算出し、ひっかき行動の自動判定
および一連の行動（Bout）の抽出を行います。
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
    DEFAULT_FPS
)

def compute_distance(p1_x, p1_y, p2_x, p2_y):
    """2点間のユークリッド距離を計算し、四捨五入した整数値を返します。"""
    if np.isnan(p1_x) or np.isnan(p1_y) or np.isnan(p2_x) or np.isnan(p2_y):
        return 0
    dist = np.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)
    return int(round(dist))

def extract_features_and_bouts(pose_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    トラッキング済みのポーズデータから各種キネマティクス特徴量を計算し、
    ルールベース分類器によってスクラッチ判定（scratch_flag, scratch_prob）を行い、
    さらに Bout 情報を集計したデータフレームを出力します。

    Parameters
    ----------
    pose_df : pd.DataFrame
        'frame_idx', 'track', 'mouse_id' などのカラムに加え、
        各キーポイントの座標（例: 'nose_x', 'nose_y' 等）を持つデータフレーム。
        ※color_id.assign_track_ids により 'track' が分割済み、
          および 'mouse_id' (1-4) がアライメント済みであることを想定しています。

    Returns
    -------
    processed_pose_df : pd.DataFrame
        特徴量とスクラッチ判定結果が追加されたフレーム単位データフレーム。
    bouts_df : pd.DataFrame
        Bout単位の集計データフレーム。
    """
    # mouse_id が NaN の行は処理対象外、またはダミーを割り当てる
    # 基本的に有効な mouse_id (1-4) を持つデータ行のみ特徴量計算を行う
    pose_df = pose_df.copy()
    
    # 欠落している特徴量カラムを初期化
    feature_cols = [
        "body_length_px", "head_to_nose_dist",
        "hind_to_head_dist_L", "hind_to_head_dist_R",
        "hind_to_ear_dist_L", "hind_to_ear_dist_R",
        "hind_paw_speed_L", "hind_paw_speed_R",
        "hind_osc_x_L", "hind_osc_y_L", "hind_osc_x_R", "hind_osc_y_R",
        "scratch_flag", "scratch_prob"
    ]
    for col in feature_cols:
        pose_df[col] = 0.0

    # 個体（mouse_id）ごとに時系列ソートして計算
    # mouse_id が NaN のデータも track 単位で計算可能だが、CSV出力時の mouse_id 整合性のため
    # mouse_id を持つデータのみ、または track 単位で計算し、結果を反映する
    # ここでは、 track ごとに時系列で計算するのが速度の面でも最も安全
    grouped = pose_df.groupby("track")
    
    for track_id, track_grp in grouped:
        # 時系列順にソートしたインデックスを取得
        sorted_grp = track_grp.sort_values("frame_idx")
        indices = sorted_grp.index.tolist()
        
        # 前フレームの座標保持用
        prev_x_L, prev_y_L = None, None
        prev_x_R, prev_y_R = None, None
        
        # 振動数計算用のx座標履歴（最大15フレーム）
        x_history_L = []
        x_history_R = []
        
        for idx in indices:
            row = sorted_grp.loc[idx]
            
            # 各種キーポイントの座標取得
            # 存在しない場合や NaN の場合はデフォルト 0 となる
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
            
            # 1. 空間的な距離特徴量の計算 (丸めて整数にする)
            body_length = compute_distance(neck_x, neck_y, tail_x, tail_y)
            head_to_nose = compute_distance(nose_x, nose_y, head_x, head_y)
            
            hind_to_head_L = compute_distance(hind_l_x, hind_l_y, head_x, head_y)
            hind_to_head_R = compute_distance(hind_r_x, hind_r_y, head_x, head_y)
            
            hind_to_ear_L = compute_distance(hind_l_x, hind_l_y, ear_l_x, ear_l_y)
            hind_to_ear_R = compute_distance(hind_r_x, hind_r_y, ear_r_x, ear_r_y)
            
            # 2. 移動速度（変位量）の計算
            speed_L = 0
            speed_R = 0
            
            # 左後肢の速度
            if not np.isnan(hind_l_x) and not np.isnan(hind_l_y):
                if prev_x_L is not None and prev_y_L is not None:
                    speed_L = compute_distance(hind_l_x, hind_l_y, prev_x_L, prev_y_L)
                prev_x_L, prev_y_L = hind_l_x, hind_l_y
            else:
                prev_x_L, prev_y_L = None, None
                
            # 右後肢の速度
            if not np.isnan(hind_r_x) and not np.isnan(hind_r_y):
                if prev_x_R is not None and prev_y_R is not None:
                    speed_R = compute_distance(hind_r_x, hind_r_y, prev_x_R, prev_y_R)
                prev_x_R, prev_y_R = hind_r_x, hind_r_y
            else:
                prev_x_R, prev_y_R = None, None

            # 3. 振動数の計算 (15フレームのローリング窓)
            # 現在値を追加
            x_history_L.append(hind_l_x)
            if len(x_history_L) > SCRATCH_WINDOW_FRAMES:
                x_history_L.pop(0)
                
            x_history_R.append(hind_r_x)
            if len(x_history_R) > SCRATCH_WINDOW_FRAMES:
                x_history_R.pop(0)
                
            # 左肢の振動数計算
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
                    
            # 右肢の振動数計算
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
            
            # 特徴量の一括更新
            pose_df.at[idx, "body_length_px"] = body_length
            pose_df.at[idx, "head_to_nose_dist"] = head_to_nose
            pose_df.at[idx, "hind_to_head_dist_L"] = hind_to_head_L
            pose_df.at[idx, "hind_to_head_dist_R"] = hind_to_head_R
            pose_df.at[idx, "hind_to_ear_dist_L"] = hind_to_ear_L
            pose_df.at[idx, "hind_to_ear_dist_R"] = hind_to_ear_R
            pose_df.at[idx, "hind_paw_speed_L"] = speed_L
            pose_df.at[idx, "hind_paw_speed_R"] = speed_R
            pose_df.at[idx, "hind_osc_x_L"] = osc_L
            pose_df.at[idx, "hind_osc_x_R"] = osc_R
            
            # 4. ルールベースによるひっかき行動の自動分類判定
            # 左右のいずれかの後足がひっかき行動基準を満たすか
            scratch_L = (speed_L > SCRATCH_MIN_VELOCITY and
                         hind_to_ear_L < SCRATCH_MAX_HEAD_DIST and
                         osc_L >= SCRATCH_MIN_OSCILLATIONS)
                         
            scratch_R = (speed_R > SCRATCH_MIN_VELOCITY and
                         hind_to_ear_R < SCRATCH_MAX_HEAD_DIST and
                         osc_R >= SCRATCH_MIN_OSCILLATIONS)
                         
            if scratch_L or scratch_R:
                pose_df.at[idx, "scratch_flag"] = 1
                pose_df.at[idx, "scratch_prob"] = 0.95
            else:
                pose_df.at[idx, "scratch_flag"] = 0
                pose_df.at[idx, "scratch_prob"] = 0.0

    # 5. Bout (ひっかき一連動作) の統合処理
    # mouse_id (1-4) ごとにフレームを走査し、持続時間フィルタリングとマージを行う
    bouts_list = []
    
    # 有効な個体IDが存在する行のみ対象
    valid_mice_df = pose_df[pose_df["mouse_id"].notna()]
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
                # マージ基準: 隙間が BOUT_MERGE_GAP (10) フレーム以内なら同一の Bout とする
                if f - last_f <= BOUT_MERGE_GAP:
                    last_f = f
                else:
                    # Bout 終了判定
                    duration_f = last_f - start_f + 1
                    # 最小フレーム数 BOUT_MIN_FRAMES (5) 以上なら採用
                    if duration_f >= BOUT_MIN_FRAMES:
                        bouts.append((start_f, last_f))
                    start_f = f
                    last_f = f
            
            # 最後の区間
            duration_f = last_f - start_f + 1
            if duration_f >= BOUT_MIN_FRAMES:
                bouts.append((start_f, last_f))

        # Bout オブジェクトの作成
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
        # 空のデータフレームにスキーマを設定して作成
        bouts_df = pd.DataFrame(columns=[
            "mouse_id", "bout_id", "bout_start_frame", "bout_end_frame",
            "bout_start_time_s", "bout_end_time_s", "bout_duration_s", "bout_n_frames"
        ])
        
    return pose_df, bouts_df
