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
import re
from config import (
    MOUSE_COLORS, RED_WRAP_LOWER, RED_WRAP_UPPER, MAX_MICE_LIMIT, ACTIVE_MOUSE_IDS,
    # ── 色 + 帯本数 による個体識別 (color + band count) ──
    MOUSE_ID_BY_COLOR_COUNT, BAND_COUNT_COLOR_IDS,
    TAIL_SAMPLE_STEP_PX, TAIL_SAMPLE_RADIUS_PX, BAND_MIN_LEN_PX, BAND_MIN_GAP_PX,
)

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


# ── 色 + 帯本数 (color + band count) による個体識別 ───────────────────────────
# 尾に塗った「色」だけでなく「帯の本数」(赤1本/赤2本/青1本/青2本) も併用して
# 4 個体を一意に識別する。単点 HSV 分類 (classify_tail_color) では本数を区別できないため、
# DLC SuperAnimal-TopViewMouse が出力する尾チェーン (tail_base→tail_end) の折れ線に沿って
# 色を細かく標本化し、ランレングス符号化で帯の本数を数える方式を用いる。
#
# 注意: 本関数が返す mouse_id は MOUSE_ID_BY_COLOR_COUNT に基づく「色+本数」ID 空間 (1–4) で
#       あり、classify_tail_color が返す「色のみ」ID 空間 (1=赤, 2=青, …) とは意味が異なる。
#       両者を同一動画内で混在させてはならない（assign_track_ids でモードを排他選択する）。

# 標本点パッチが当該色とみなされる最小被覆率（パッチ面積の 1/4 以上）。
_PATCH_MIN_FRAC = 0.25


def _classify_sample_color(frame: np.ndarray,
                           cx: float, cy: float, radius: int) -> str | None:
    """
    尾上の単一標本点まわりの円形パッチを HSV 判定し、帯カウント対象色 (赤/青) の
    色名を返す。どの対象色にも十分一致しなければ None（皮膚・地色）。

    Parameters
    ----------
    frame  : BGR 画像
    cx, cy : 標本点のピクセル座標
    radius : 円形パッチ半径 (TAIL_SAMPLE_RADIUS_PX)
    """
    h, w = frame.shape[:2]
    x, y = int(round(cx)), int(round(cy))
    x1, y1 = max(0, x - radius), max(0, y - radius)
    x2, y2 = min(w, x + radius), min(h, y + radius)
    if (x2 - x1) < 3 or (y2 - y1) < 3:
        return None

    roi_bgr = frame[y1:y2, x1:x2]
    roi_hsv = cv2.cvtColor(cv2.GaussianBlur(roi_bgr, (3, 3), 0), cv2.COLOR_BGR2HSV)

    # ROI 内の円形マスク（パッチ実効領域）
    px, py = x - x1, y - y1
    mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (px, py), radius, 255, -1)
    total = int(np.count_nonzero(mask))
    if total == 0:
        return None

    best_name, best_frac = None, 0.0
    for mouse_id in BAND_COUNT_COLOR_IDS:
        info = MOUSE_COLORS.get(mouse_id)
        if not info:
            continue
        lower = np.array(info["hsv_lower"], dtype=np.uint8)
        upper = np.array(info["hsv_upper"], dtype=np.uint8)
        cm = cv2.inRange(roi_hsv, lower, upper)
        # 赤は色相環の両端 (0付近 / 180付近) に分布するため両端マスクを結合
        if mouse_id == 1:
            cm = cv2.bitwise_or(cm, cv2.inRange(
                roi_hsv,
                np.array(RED_WRAP_LOWER, dtype=np.uint8),
                np.array(RED_WRAP_UPPER, dtype=np.uint8)))
        frac = float(np.count_nonzero(cv2.bitwise_and(cm, mask))) / float(total)
        if frac > best_frac:
            best_frac, best_name = frac, info["name"]

    return best_name if best_frac >= _PATCH_MIN_FRAC else None


def _resample_polyline(points: list, step: float) -> list:
    """
    順序付き頂点列 points=[(x,y), ...] を弧長 step ごとに等間隔再標本化する。
    尾チェーンの各ノード間隔は不均一なため、帯長を一定尺度で数えられるよう距離等間隔化する。
    """
    pts = [(float(x), float(y)) for (x, y) in points
           if x is not None and y is not None
           and not (np.isnan(x) or np.isnan(y))]
    if len(pts) < 2:
        return pts[:]

    samples = [pts[0]]
    carry = 0.0  # 直前セグメントで標本化しきれなかった残り弧長
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = float(np.hypot(x1 - x0, y1 - y0))
        if seg <= 1e-6:
            continue
        d = step - carry
        while d <= seg:
            t = d / seg
            samples.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            d += step
        carry = seg - (d - step)
    return samples


def _count_bands(present: list, min_run: int, max_gap: int) -> int:
    """
    二値列 present (要素 1=対象色, 0=非対象色) から帯の本数を数える。

    - 連続する 1 の run 長が min_run 以上のものを 1 帯とみなす。
    - 帯と帯の間の 0 ギャップが max_gap 以上のときのみ帯の連続を打ち切る。
      （市松状の途切れや 1px の欠損で帯が過剰分割されるのを防ぐため、
        短いギャップは同一帯として結合する。）
    - min_run 未満の微小な 1 run はノイズとして無視する。
    """
    if not present:
        return 0

    # ランレングス符号化 (value, length)
    runs = []
    cur_v, cur_len = present[0], 1
    for v in present[1:]:
        if v == cur_v:
            cur_len += 1
        else:
            runs.append((cur_v, cur_len))
            cur_v, cur_len = v, 1
    runs.append((cur_v, cur_len))

    bands = 0
    pending_band = False  # 直近で帯を計上済みか（小ギャップ結合の判定用）
    for v, length in runs:
        if v == 1 and length >= min_run:
            if not pending_band:
                bands += 1
                pending_band = True
        elif v == 0 and length >= max_gap:
            # 大きなギャップで帯の連続を打ち切る → 次の帯は別本として計上される
            pending_band = False
        # それ以外 (微小な色 run / 小ギャップ) は状態を維持して無視
    return bands


def classify_tail_by_bands(frame: np.ndarray, tail_points: list) -> int | None:
    """
    尾の折れ線 (base→end 順の座標列) に沿って色を標本化し、「色 + 帯本数」から
    mouse_id (MOUSE_ID_BY_COLOR_COUNT, 1–4) を返す。判定不能なら None。

    手順:
      1. 折れ線を TAIL_SAMPLE_STEP_PX 間隔で等距離再標本化する。
      2. 各標本点の円形パッチ (半径 TAIL_SAMPLE_RADIUS_PX) を赤/青/None に分類する。
      3. 対象色ごとにラベル列をランレングス符号化し、帯の本数を数える
         (BAND_MIN_LEN_PX で 1 帯の最小長、BAND_MIN_GAP_PX で帯の分離を制御)。
      4. 標本数が最多の対象色を尾のマーカー色とし、その帯本数 (∈{1,2}) を採用する。
      5. (色名, 本数) → MOUSE_ID_BY_COLOR_COUNT で個体 ID へ対応付ける。
    """
    samples = _resample_polyline(tail_points, TAIL_SAMPLE_STEP_PX)
    if len(samples) < 2:
        return None

    labels = [_classify_sample_color(frame, x, y, TAIL_SAMPLE_RADIUS_PX)
              for (x, y) in samples]

    # px 閾値を「標本数」に換算（標本間隔 = TAIL_SAMPLE_STEP_PX）
    step = float(TAIL_SAMPLE_STEP_PX)
    min_run = max(1, int(round(BAND_MIN_LEN_PX / step)))  # 1 帯とみなす最小標本数
    max_gap = max(1, int(round(BAND_MIN_GAP_PX / step)))  # 帯を分離する最小ギャップ標本数

    best_id, best_score = None, 0
    for mouse_id in BAND_COUNT_COLOR_IDS:
        color_name = MOUSE_COLORS[mouse_id]["name"]
        present = [1 if lab == color_name else 0 for lab in labels]
        total_hits = sum(present)
        if total_hits == 0:
            continue
        bands = _count_bands(present, min_run, max_gap)
        if bands <= 0:
            continue
        band_count = min(2, bands)  # 3 本以上は 2 とみなす（対応表の都合）
        mid = MOUSE_ID_BY_COLOR_COUNT.get((color_name, band_count))
        if mid is None:
            continue
        # 支配色の決定: 当該色の総標本数が最多のものを尾のマーカー色として採用
        if total_hits > best_score:
            best_score, best_id = total_hits, mid

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

def _ordered_tail_keypoints(columns) -> list:
    """
    wide フォーマットの列名から尾チェーンのキーポイント基底名を base→end の順に並べて返す。
    DLC のバージョンやモデルにより尾ノード名は tail_base / tail1..tail5 / tail_end 等と
    異なるため、末尾の番号で頑健にソートする。

    順序規則:
      tail_base (始端) → tail1, tail2, … (番号昇順) → tail_end / tail_tip (終端)
    """
    names = set()
    for c in columns:
        if isinstance(c, str) and c.endswith("_x"):
            base = c[:-2]
            if base.lower().startswith("tail"):
                names.add(base)

    def sort_key(n: str):
        nl = n.lower()
        if nl in ("tail_base", "tailbase"):
            return (0, 0)
        if nl in ("tail_end", "tailend", "tail_tip", "tailtip"):
            return (2, 0)
        m = re.search(r"(\d+)", nl)
        return (1, int(m.group(1)) if m else 999)

    return sorted(names, key=sort_key)


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
        # [A-3] 人が確認・承認したマッピングのため ID 信頼度は最高 (1.0) とする。
        # 仮割り当て（未マッピングからの連番フォールバック）分は根拠が弱いため 0.0。
        mapped_tracks = set(initial_mapping.keys())
        result["id_confidence"] = result["track"].map(
            lambda t: 1.0 if str(t) in mapped_tracks else 0.0
        )
        per_mouse_cm = {}
        for m_id in sorted(result["mouse_id"].dropna().unique()):
            m_id_int = int(m_id)
            # mouse_id=0 は「誤検出（除外）」の特別値。個体QCの対象外とする。
            if m_id_int <= 0:
                continue
            mask = result["mouse_id"] == m_id
            n_frames = int(mask.sum())
            n_low = int((mask & (result["id_confidence"] < 0.70)).sum())
            mean_conf = float(result.loc[mask, "id_confidence"].mean()) if n_frames else 0.0
            per_mouse_cm[m_id_int] = {
                "mean_confidence":   round(mean_conf, 4),
                "n_frames":          n_frames,
                "n_low_conf_frames": n_low,
                "n_subtracks":       1,
            }
        result.attrs["id_quality"] = {
            "low_conf_threshold": 0.70,
            "swap_events":        [],
            "n_swap_events":      0,
            "per_mouse":          per_mouse_cm,
            "source":             "confirmed_mapping",
        }
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

    # ── 個体識別モードの決定: 尾チェーンの有無で「色+帯本数」/「単点色」を排他選択 ──
    # 尾チェーン (tail_base + tail1.. + tail_end) が wide df に存在すれば帯本数判定を使い、
    # MOUSE_ID_BY_COLOR_COUNT による「色+本数」ID 空間 (1–4) で投票する。
    # 尾チェーンが無い（tail_base 単点のみ）場合は従来の単点 HSV 分類 (色のみ ID) に後方互換する。
    # ※ 2 つの ID 空間は意味が異なるため、同一動画内で混在させない。
    tail_kps = _ordered_tail_keypoints(pose_df.columns)
    use_band_counting = len(tail_kps) >= 2
    if use_band_counting:
        print(f"[color_id] [Info] 尾チェーン {tail_kps} を検出。色+帯本数による個体識別を使用します。")
    else:
        print("[color_id] [Info] 尾チェーン未検出。単点尾色分類（色のみ・後方互換）を使用します。")

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

    # [A-3] ID品質管理: 検知した Identity Swap（色取り違え）イベントを記録する。
    # 各要素 = {orig_track, frame_idx, old_subtrack, new_subtrack, color}
    swap_events = []

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

            # 色の分類（モードにより ID 空間が異なる。混在させない）
            if use_band_counting:
                # 尾チェーンの折れ線を base→end の順に組み立て、低信頼ノードは除外する。
                tail_points = []
                for kp in tail_kps:
                    kx = row.get(f"{kp}_x", np.nan)
                    ky = row.get(f"{kp}_y", np.nan)
                    kc = row.get(f"{kp}_conf", 1.0)
                    if (not (np.isnan(kx) or np.isnan(ky))
                            and (kc is None or np.isnan(kc) or kc >= 0.30)):
                        tail_points.append((kx, ky))
                # 「色+帯本数」→ mouse_id (1–4)。判定不能なら None（そのフレームは投票しない）。
                # ※ 単点色分類への代替は ID 空間が異なるため行わない。
                color_id = (classify_tail_by_bands(frame, tail_points)
                            if len(tail_points) >= 2 else None)
            else:
                # 従来の単点 HSV 分類（色のみ ID, 1–4）
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
                        # [A-3] スワップイベントを品質レポート用に記録
                        swap_events.append({
                            "orig_track":   str(orig_track),
                            "frame_idx":    int(frame_idx),
                            "old_subtrack": str(old_subtrack),
                            "new_subtrack": str(new_subtrack),
                            "color":        int(major_color) if major_color is not None else None,
                        })
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
    # [A-3] あわせて ID 信頼度 = 採用色の得票数 / 総得票数 を算出する。
    #   信頼度 = (採用色票数) / (有効カラー票総数)
    #   1.0 に近いほどそのサブトラックの色は一貫しており、ID 割り当てが確実。
    #   0.5 付近は別個体の色票が多く混入しており、混同（Swap）の疑いが強い。
    subtrack_to_color = {}
    subtrack_confidence = {}
    for subtrack_id, v_list in subtrack_votes.items():
        if v_list:
            major_color, major_count = Counter(v_list).most_common(1)[0]
            subtrack_to_color[subtrack_id] = major_color
            subtrack_confidence[subtrack_id] = float(major_count) / float(len(v_list))
        else:
            subtrack_to_color[subtrack_id] = None
            # カラー票がまったく無いサブトラック（後段で体長フォールバック対象）は
            # 色由来の根拠が無いため信頼度 0.0（要確認）とする。
            subtrack_confidence[subtrack_id] = 0.0

    # 最終的な DataFrame の更新
    pose_df = pose_df.copy()
    
    # 1. 各行の 'track' を分割後の 'subtrack_id' に更新 (B-6: zip内包表記による高速化)
    pose_df["track"] = [frame_subtrack_map.get((f, t), t) for f, t in zip(pose_df["frame_idx"], pose_df["track"])]

    # 2. 基本的な 'mouse_id' カラムの作成・マッピング
    pose_df["mouse_id"] = pose_df["track"].map(lambda t: subtrack_to_color.get(t, None))

    # 3. 同一フレーム内での個体ID重複（衝突）の解消 (C-2: `.loc` O(N^2) 検索を避けるため `.at` に最適化)
    # 注: 重複解消はフレーム単位で多数発生しうる（数千フレーム×複数IDで数十万件規模）。
    #     各解消を逐一 print するとログが肥大化するため、件数のみ集計し末尾で1行に集約する。
    dedup_count = 0           # 解消した（ID を解除した）サブトラックの総件数
    dedup_frames = set()      # 解消が発生したフレームの集合（影響フレーム数の把握用）
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
                            dedup_count += 1
                            dedup_frames.add(frame_idx)

    if dedup_count > 0:
        print(f"[color_id] [Info] 同一フレーム内ID重複の解消: 計 {dedup_count} 件"
              f"（{len(dedup_frames)} フレームで発生）。各フレームで色票数が最多のサブトラックにIDを残し、"
              f"他のサブトラックのIDを解除しました。")

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

    # ── [A-3] ID信頼度カラムの付与と品質レポートの構築 ───────────────────────
    # 各フレーム行に、そのサブトラックのID信頼度（採用色得票率）を付与する。
    # frame_labels CSV に出力され、フロントエンドで低信頼フレームの可視化に使われる。
    pose_df["id_confidence"] = pose_df["track"].map(
        lambda t: round(subtrack_confidence.get(t, 0.0), 4)
    )

    # 個体（mouse_id）ごとに信頼度を集計する。投票数を重みとした加重平均を採用し、
    # 短いサブトラックのノイズが過度に影響しないようにする。
    per_mouse = {}
    for subtrack_id, color in subtrack_to_color.items():
        if color is None:
            continue
        n_votes = len(subtrack_votes.get(subtrack_id, []))
        conf = subtrack_confidence.get(subtrack_id, 0.0)
        acc = per_mouse.setdefault(int(color), {"weighted_sum": 0.0, "n_votes": 0, "n_subtracks": 0})
        acc["weighted_sum"] += conf * n_votes
        acc["n_votes"] += n_votes
        acc["n_subtracks"] += 1

    LOW_CONF_THRESHOLD = 0.70  # この値未満のフレームを「低信頼」として警告対象にする
    mouse_quality = {}
    for m_id, acc in per_mouse.items():
        mean_conf = (acc["weighted_sum"] / acc["n_votes"]) if acc["n_votes"] > 0 else 0.0
        mask = pose_df["mouse_id"] == m_id
        n_frames = int(mask.sum())
        n_low = int((mask & (pose_df["id_confidence"] < LOW_CONF_THRESHOLD)).sum())
        mouse_quality[int(m_id)] = {
            "mean_confidence":   round(float(mean_conf), 4),
            "n_frames":          n_frames,
            "n_low_conf_frames": n_low,
            "n_subtracks":       acc["n_subtracks"],
        }

    # 品質レポートを DataFrame の attrs に格納する（戻り値シグネチャを変えずに伝搬）。
    pose_df.attrs["id_quality"] = {
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "swap_events":        swap_events,
        "n_swap_events":      len(swap_events),
        "per_mouse":          mouse_quality,
    }

    return pose_df

