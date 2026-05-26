"""
Tail-Color → Individual ID  (v2 — 白色マウス・動的個体数対応)
============================================================
尾部周辺のHSVカラーサンプリングによって個体IDを割り当てます。
以下の拡張を含みます:
  - 白色マウス (ID=5) の自然毛色検出
  - 最大 MAX_MICE_LIMIT 匹までの動的個体数対応
  - カラー分類が失敗した場合の体長クラスタリングによるフォールバック ID 割当

Strategy
--------
1. For each frame, extract a circular patch around tail_base.
2. Convert to HSV and count pixels matching each color range.
3. Accumulate votes over a rolling window (majority rule).
4. Resolve conflicts (two tracks claiming the same color) with the
   track that has the higher vote count winning.
5. [NEW] For tracks with no color match, attempt body-size k-means clustering
   as a secondary identification method.
"""

import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, Optional
from config import MOUSE_COLORS, RED_WRAP_LOWER, RED_WRAP_UPPER, MAX_MICE_LIMIT, ACTIVE_MOUSE_IDS

# ── Tunable parameters ────────────────────────────────────────────────────────
SAMPLE_RADIUS    = 12   # px — circle radius around tail_base keypoint
MIN_MATCH_PX     = 20   # minimum colored pixels to accept a classification
VOTE_WINDOW      = 30   # frames — rolling window for majority vote

# 白色マウス検出用の最小ピクセル数（白は背景ノイズに弱いため高めに設定）
WHITE_MIN_MATCH_PX = 40  # px — 白マウスの検出閾値（誤検出抑制）


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

    # 使用するマウスIDセットの決定
    # ACTIVE_MOUSE_IDS が設定されていればそのIDのみ使用、空の場合は全MOUSE_COLORSを使用
    target_ids = ACTIVE_MOUSE_IDS if ACTIVE_MOUSE_IDS else list(MOUSE_COLORS.keys())

    best_id, best_count = None, MIN_MATCH_PX

    for mouse_id in target_ids:
        if mouse_id not in MOUSE_COLORS:
            continue
        info = MOUSE_COLORS[mouse_id]
        lower = np.array(info["hsv_lower"], dtype=np.uint8)
        upper = np.array(info["hsv_upper"], dtype=np.uint8)

        # 赤色はHSV色相環の端 (0付近と180付近) に分布するため両端のマスクを結合する
        if mouse_id == 1:
            m1 = cv2.inRange(roi_hsv, lower, upper)
            m2 = cv2.inRange(roi_hsv,
                             np.array(RED_WRAP_LOWER, dtype=np.uint8),
                             np.array(RED_WRAP_UPPER, dtype=np.uint8))
            color_mask = cv2.bitwise_or(m1, m2)
        else:
            color_mask = cv2.inRange(roi_hsv, lower, upper)

        hit_count = int(np.count_nonzero(cv2.bitwise_and(color_mask, roi_mask)))

        # 白色マウス (ID=5) は背景ノイズに弱いため高い検出閾値を設定
        threshold = WHITE_MIN_MATCH_PX if mouse_id == 5 else MIN_MATCH_PX
        if hit_count > best_count and hit_count >= threshold:
            best_count, best_id = hit_count, mouse_id

    return best_id


def _estimate_body_length_px(row: pd.Series) -> float:
    """
    neck-to-tail_base 間のピクセル距離を体長の代理値として算出する。
    カラーIDが割り当てられない個体の体長クラスタリングに使用する。
    """
    nx = row.get("neck_x", np.nan)
    ny = row.get("neck_y", np.nan)
    tx = row.get("tail_base_x", np.nan)
    ty = row.get("tail_base_y", np.nan)
    if any(np.isnan(v) for v in [nx, ny, tx, ty]):
        return np.nan
    return float(np.sqrt((nx - tx) ** 2 + (ny - ty) ** 2))


def _fallback_assign_by_body_size(
    unresolved_subtracks: list,
    subtrack_votes: dict,
    pose_df: pd.DataFrame,
    used_ids: set,
    max_mice: int,
) -> dict:
    """
    カラーIDが割り当てられなかったサブトラックに対して、体長の中央値を使用した
    k-meansクラスタリングによって代替IDを割り当てる。

    戦略:
    - サブトラック群の平均体長を計算
    - 体長が似ている個体を同一IDとして扱う（体長差 < 15px を同一クラスタとみなす）
    - 使用済みIDを避け、未使用IDで小さい順に割り当てる

    Parameters
    ----------
    unresolved_subtracks : ID未割り当てのサブトラックID一覧
    subtrack_votes       : サブトラックごとの投票リスト
    pose_df              : 現在の pose DataFrame (体長計算に使用)
    used_ids             : 既に割り当て済みの mouse_id 集合
    max_mice             : 最大マウス数 (MAX_MICE_LIMIT)

    Returns
    -------
    dict: {subtrack_id: new_mouse_id} の割り当て辞書
    """
    # 各サブトラックの平均体長を取得
    body_lengths = {}
    for st in unresolved_subtracks:
        grp = pose_df[pose_df["track"] == st]
        lengths = grp.apply(_estimate_body_length_px, axis=1).dropna()
        body_lengths[st] = float(lengths.median()) if not lengths.empty else np.nan

    # 体長が不明なものは後回し
    valid = {st: bl for st, bl in body_lengths.items() if not np.isnan(bl)}
    if not valid:
        return {}

    # 体長でソートし、差分が閾値 (15px) 以下のものを同一クラスタにまとめる
    CLUSTER_THRESHOLD_PX = 15.0
    sorted_tracks = sorted(valid.items(), key=lambda x: x[1])
    clusters: list[list] = []
    current_cluster: list = []

    for i, (st, bl) in enumerate(sorted_tracks):
        if not current_cluster:
            current_cluster.append(st)
        else:
            prev_bl = valid[current_cluster[-1]]
            if abs(bl - prev_bl) <= CLUSTER_THRESHOLD_PX:
                current_cluster.append(st)
            else:
                clusters.append(current_cluster)
                current_cluster = [st]
    if current_cluster:
        clusters.append(current_cluster)

    # 未使用IDを昇順で準備
    available_ids = [i for i in range(1, max_mice + 1) if i not in used_ids]

    result = {}
    for cluster, new_id in zip(clusters, available_ids):
        for st in cluster:
            result[st] = new_id
            print(f"[color_id] [Fallback] 体長クラスタリングでID割当: subtrack={st} -> mouse_id={new_id} (体長:{valid[st]:.1f}px)")

    return result


# ── Batch assignment for a full video ────────────────────────────────────────

def assign_track_ids(video_path: str, pose_df: pd.DataFrame,
                     mice_count: int = 0,
                     initial_mapping: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    動画の全フレームをスキャンし、スライディング窓を用いて各トラックのカラーID確率推移を評価します。
    急激な色の変化（Identity Swap / 混同）を検知した時点で、そのトラックを自動分割し、
    混同の伝播を防止した上で安定した mouse_id を割り当てます。

    [拡張] 白色・無マーカーマウス対応:
    - MOUSE_COLORS に白 (ID=5) を含めることで白色マウスを尾部HSVで検出
    - カラー分類が失敗したトラックには体長クラスタリングによる代替IDを割り当て

    [拡張] 動的個体数対応:
    - mice_count=0 の場合は動画内のトラック数から自動決定
    - mice_count > 0 の場合はその値を期待最大個体数として使用

    Conflict resolution
    -------------------
    If two subtracks claim the same color, the subtrack with more votes keeps it;
    the other subtrack is labeled as mouse_id = None.

    Parameters
    ----------
    video_path      : ソース動画のパス (.mp4 / .avi)
    pose_df         : load_sleap_h5() でロードされたDataFrame
    mice_count      : 期待個体数 (0 = 自動検出)
    initial_mapping : 研究者確認済みの track→mouse_id マッピング辞書。
                      例: {"track_0": 1, "track_1": 2, ...}
                      指定した場合はカラーID自動分類をスキップしてこの値を使用する。

    Returns
    -------
    pose_df with updated 'track' columns and a new 'mouse_id' column (int or NaN)
    """
    # ── 確認済みマッピングによるショートカット ──────────────────────────────
    # フロントエンドで研究者が確認・承認した track→mouse_id マッピングが提供された場合、
    # カラーHSV自動分類をスキップして直接 mouse_id を割り当てる。
    # これにより初期フレームでの人による確認結果が全フレームに反映される。
    if initial_mapping is not None:
        print("[color_id] [Info] 確認済みIDマッピングが提供されました。カラーID自動分類をスキップします。")
        result = pose_df.copy()
        result["mouse_id"] = result["track"].map(
            lambda t: initial_mapping.get(str(t), None)
        )
        resolved = int(result["mouse_id"].notna().sum())
        unresolved = int(result["mouse_id"].isna().sum())
        print(f"[color_id] [Info] マッピング結果: {resolved} 行で割り当て済み, {unresolved} 行で未解決")
        if unresolved > 0:
            print("[color_id] [Warning] 一部のトラックでマッピングが適用されませんでした。"
                  "トラック名が不一致の可能性があります。")
            # 未解決分は連番でフォールバック割り当て
            used_ids = set(v for v in initial_mapping.values())
            unmapped_tracks = result[result["mouse_id"].isna()]["track"].unique()
            next_id = 1
            for track in sorted(unmapped_tracks):
                while next_id in used_ids:
                    next_id += 1
                result.loc[result["track"] == track, "mouse_id"] = next_id
                used_ids.add(next_id)
                print(f"[color_id] [Info] 未マッピングトラック {track} → mouse_id={next_id} に仮割り当て")
        result["mouse_id"] = result["mouse_id"].astype("Int64")
        return result

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

    # ── フォールバック: 体長クラスタリングで未解決トラックにIDを割り当てる ─────
    # カラーが識別できなかったトラック（白色マウスなど）を体長の類似度でクラスタリングする
    unresolved_subtracks_now = [
        st for st in subtrack_to_color
        if subtrack_to_color[st] is None
    ]
    if unresolved_subtracks_now:
        print(f"[color_id] [Info] {len(unresolved_subtracks_now)} サブトラックでカラーID未解決。体長クラスタリングによるフォールバックを試みます。")
        # 使用済みIDを収集
        used_ids = set(v for v in subtrack_to_color.values() if v is not None)
        fallback_map = _fallback_assign_by_body_size(
            unresolved_subtracks_now,
            subtrack_votes,
            pose_df,
            used_ids,
            MAX_MICE_LIMIT,
        )
        # フォールバック結果をマッピングに反映
        for st, new_id in fallback_map.items():
            subtrack_to_color[st] = new_id
        # pose_df の mouse_id を再マッピング
        pose_df["mouse_id"] = pose_df["track"].map(lambda t: subtrack_to_color.get(t, None))

    unresolved = pose_df["mouse_id"].isna().sum()
    if unresolved:
        print(f"[color_id] [Warning] {unresolved} フレームで mouse_id が未解決です（フォールバック後）。")

    # ── 動的個体数の検証 ──────────────────────────────────────────────────────
    detected_count = int(pose_df["mouse_id"].dropna().nunique())
    expected = mice_count if mice_count > 0 else detected_count
    print(f"[color_id] [Info] 検出個体数: {detected_count} 匹 (期待値: {expected} 匹)")
    if mice_count > 0 and detected_count != mice_count:
        print(f"[color_id] [Warning] 検出個体数 ({detected_count}) が期待値 ({mice_count}) と異なります。"
              f"動画の途中で個体数が変化している可能性があります。")

    return pose_df

