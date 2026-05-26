"""
Tail-Color → Individual ID
==========================
Samples the tail region around the `tail_base` keypoint in each video frame
and classifies the painted color (red / blue / yellow / black) to assign a
stable mouse_id (1–4) to each SLEAP track.

Strategy
--------
1. For each frame, extract a circular patch around tail_base.
2. Convert to HSV and count pixels matching each color range.
3. Accumulate votes over a rolling window (majority rule).
4. Resolve conflicts (two tracks claiming the same color) with the
   track that has the higher vote count winning.
"""

import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from config import MOUSE_COLORS, RED_WRAP_LOWER, RED_WRAP_UPPER

# ── Tunable parameters ────────────────────────────────────────────────────────
SAMPLE_RADIUS    = 12   # px — circle radius around tail_base keypoint
MIN_MATCH_PX     = 20   # minimum colored pixels to accept a classification
VOTE_WINDOW      = 30   # frames — rolling window for majority vote


# ── Per-frame color classifier ────────────────────────────────────────────────

def classify_tail_color(frame: np.ndarray,
                         tail_xy: tuple[float, float],
                         confidence: float = 1.0) -> int | None:
    """
    Classify the painted tail color in a single frame.

    Parameters
    ----------
    frame      : BGR image (from cv2.VideoCapture)
    tail_xy    : (x, y) pixel position of tail_base keypoint
    confidence : SLEAP confidence score — skip if too low

    Returns
    -------
    mouse_id (1–4) or None if classification fails
    """
    if confidence < 0.30 or np.isnan(tail_xy[0]) or np.isnan(tail_xy[1]):
        return None

    x, y = int(round(tail_xy[0])), int(round(tail_xy[1]))
    h, w = frame.shape[:2]

    # M-2: マップ端（ケージの壁際）対応のため境界チェックを緩和し、ROIクリッピングで対応
    x1 = max(0, x - SAMPLE_RADIUS)
    y1 = max(0, y - SAMPLE_RADIUS)
    x2 = min(w, x + SAMPLE_RADIUS)
    y2 = min(h, y + SAMPLE_RADIUS)

    # 有効なROIサイズが小さすぎる場合は無効とする
    if (x2 - x1) < 6 or (y2 - y1) < 6:
        return None

    # ROI の切り出し (L-1: GaussianBlurを画像全体でなくROIに対して先に適用するよう最適化)
    roi_bgr = frame[y1:y2, x1:x2]
    roi_blur = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
    roi_hsv = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2HSV)

    # ROI内における円形マスクの作成
    cx, cy = x - x1, y - y1
    roi_mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(roi_mask, (cx, cy), SAMPLE_RADIUS, 255, -1)

    best_id, best_count = None, MIN_MATCH_PX

    for mouse_id, info in MOUSE_COLORS.items():
        lower = np.array(info["hsv_lower"], dtype=np.uint8)
        upper = np.array(info["hsv_upper"], dtype=np.uint8)

        # Red wraps around the HSV hue circle
        if mouse_id == 1:
            m1 = cv2.inRange(roi_hsv, lower, upper)
            m2 = cv2.inRange(roi_hsv,
                             np.array(RED_WRAP_LOWER, dtype=np.uint8),
                             np.array(RED_WRAP_UPPER, dtype=np.uint8))
            color_mask = cv2.bitwise_or(m1, m2)
        else:
            color_mask = cv2.inRange(roi_hsv, lower, upper)

        hit_count = int(np.count_nonzero(cv2.bitwise_and(color_mask, roi_mask)))
        if hit_count > best_count:
            best_count, best_id = hit_count, mouse_id

    return best_id


# ── Batch assignment for a full video ────────────────────────────────────────

def assign_track_ids(video_path: str, pose_df: pd.DataFrame) -> pd.DataFrame:
    """
    動画の全フレームをスキャンし、スライディング窓を用いて各トラックのカラーID確率推移を評価します。
    急激な色の変化（Identity Swap / 混同）を検知した時点で、そのトラックを自動分割し、
    混同の伝播を防止した上で安定した mouse_id（1–4）を割り当てます。

    Conflict resolution
    -------------------
    If two subtracks claim the same color, the subtrack with more votes keeps it;
    the other subtrack is labeled as mouse_id = None.

    Parameters
    ----------
    video_path : ソース動画のパス (.mp4 / .avi)
    pose_df    : load_sleap_h5() でロードされたDataFrame

    Returns
    -------
    pose_df with updated 'track' columns and a new 'mouse_id' column (int 1–4 or NaN)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        video_ext = os.path.splitext(video_path)[1].lower()
        hint = ""
        if video_ext == '.mov':
            hint = (" MOVファイルの場合、OpenCVがH.264コーデックに対応している必要があります。"
                   " ffmpeg -i input.mov -c:v libx264 output.mp4 で変換できます。")
        raise FileNotFoundError(f"動画ファイルを開けませんでした: {video_path}{hint}")

    # C-3: 動画フレーム数と DataFrame の整合性確認
    max_frame = pose_df["frame_idx"].max() if not pose_df.empty else 0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames < max_frame:
        print(f"[Warning] 動画フレーム数({total_video_frames}) < データフレーム数({max_frame})")

    # 各元のトラック（original_track）の状態追跡用辞書
    # track_states[orig_track] = {
    #     "current_subtrack": orig_track,
    #     "split_count": 0,
    #     "stable_color": None,
    #     "history": [] # rolling window of color_ids
    # }
    track_states = {}
    
    # 各フレームの (frame_idx, orig_track) -> subtrack_id のマッピング
    frame_subtrack_map = {}
    
    # 各 subtrack_id ごとにフレームごとの投票結果を蓄積する
    subtrack_votes = defaultdict(list)

    # C-1: groupby で事前にフレームごとにグループ化し、フレームの全走査O(N^2)をO(N)に削減
    grouped = pose_df.groupby("frame_idx")
    last_frame_idx = -1

    for frame_idx, rows in sorted(grouped):
        # 必要なフレームにシークする。連続していない場合はそのまま read()
        if frame_idx != last_frame_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        if not ret:
            # 1 フレームの読み込み失敗で以降を全て破棄しないよう、当該フレームのみスキップして継続する。
            # last_frame_idx は更新しないため、次の有効フレームで自動的にシークし直される。
            print(f"[Warning] フレーム {frame_idx} の読み込みに失敗しました。当該フレームをスキップして継続します。")
            continue
        last_frame_idx = frame_idx

        for _, row in rows.iterrows():
            orig_track = row["track"]
            tx = row.get("tail_base_x", np.nan)
            ty = row.get("tail_base_y", np.nan)
            tc = row.get("tail_base_conf", 1.0)

            # 色の分類（1–4 または None）
            color_id = classify_tail_color(frame, (tx, ty), tc)

            # トラック状態の初期化
            if orig_track not in track_states:
                track_states[orig_track] = {
                    "current_subtrack": orig_track,
                    "split_count": 0,
                    "stable_color": None,
                    "history": []
                }

            state = track_states[orig_track]
            
            # 履歴に追加して窓サイズに制限
            state["history"].append(color_id)
            if len(state["history"]) > VOTE_WINDOW:
                state["history"].pop(0)

            # スライディング窓内での主要なカラーID判定
            valid_votes = [v for v in state["history"] if v is not None]
            if len(valid_votes) >= 10:  # 信頼できる判定に必要な最小フレーム数
                counts = Counter(valid_votes)
                major_color, major_count = counts.most_common(1)[0]
                ratio = major_count / len(valid_votes)

                # 判定色が70%以上を占める場合に主要色とみなす
                if ratio >= 0.70:
                    if state["stable_color"] is None:
                        # 初回の安定色決定
                        state["stable_color"] = major_color
                    elif state["stable_color"] != major_color:
                        # 安定色が変わった場合、ID取り違え（混同）とみなし分割を実行！
                        state["split_count"] += 1
                        old_subtrack = state["current_subtrack"]
                        new_subtrack = f"{orig_track}_split_{state['split_count']}"
                        state["current_subtrack"] = new_subtrack
                        state["stable_color"] = major_color
                        # 履歴をクリアして新たなサブトラックとして開始
                        state["history"] = [major_color]
                        print(f"[color_id] [Swap] トラック混同検知: {orig_track} がフレーム {frame_idx} で色変化 "
                              f"({old_subtrack} -> {new_subtrack}, 色: {major_color})")

            # 現在のフレームでの subtrack_id をマッピングに記録
            subtrack_id = state["current_subtrack"]
            frame_subtrack_map[(frame_idx, orig_track)] = subtrack_id
            
            # サブトラック単位でのカラー投票を蓄積
            if color_id is not None:
                subtrack_votes[subtrack_id].append(color_id)

    cap.release()

    # 各サブトラックの最終色（多数決）を決定
    subtrack_to_color = {}
    for subtrack_id, v_list in subtrack_votes.items():
        if v_list:
            subtrack_to_color[subtrack_id] = Counter(v_list).most_common(1)[0][0]
        else:
            subtrack_to_color[subtrack_id] = None

    # 最終的な DataFrame の更新
    pose_df = pose_df.copy()
    
    # 1. 各行の 'track' を分割後の 'subtrack_id' に更新 (B-6: zip内包表記による高速化)
    pose_df["track"] = [frame_subtrack_map.get((f, t), t) for f, t in zip(pose_df["frame_idx"], pose_df["track"])]

    # 2. 基本的な 'mouse_id' カラムの作成・マッピング
    pose_df["mouse_id"] = pose_df["track"].map(lambda t: subtrack_to_color.get(t, None))

    # 3. 同一フレーム内での個体ID重複（衝突）の解消 (C-2: `.loc` O(N^2) 検索を避けるため `.at` に最適化)
    for frame_idx, frame_grp in pose_df.groupby("frame_idx"):
        active_mice = frame_grp[frame_grp["mouse_id"].notna()]
        if len(active_mice) != len(active_mice["mouse_id"].unique()):
            counts = Counter(active_mice["mouse_id"])
            for m_id, count in counts.items():
                if count > 1:
                    conflicting_rows = active_mice[active_mice["mouse_id"] == m_id]
                    best_subtrack = max(conflicting_rows["track"],
                                        key=lambda s: Counter(subtrack_votes[s]).get(m_id, 0))
                    
                    for idx, crow in conflicting_rows.iterrows():
                        sub_t = crow["track"]
                        if sub_t != best_subtrack:
                            pose_df.at[idx, "mouse_id"] = None
                            print(f"[color_id] [Warning] フレーム {frame_idx} 内でID重複を解消: "
                                  f"サブトラック {best_subtrack} に ID {m_id} を割り当て、{sub_t} は解除しました。")

    unresolved = pose_df["mouse_id"].isna().sum()
    if unresolved:
        print(f"[color_id] [Warning] {unresolved} フレームで mouse_id が未解決です。")
        
    return pose_df

