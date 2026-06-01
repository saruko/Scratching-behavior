"""
Mouse Scratch Detection System — Main Processing Pipeline (Revised)
==================================================================
本スクリプトは、SLEAPのトラッキングデータ（H5または自動実行ポーズデータ）と動画ファイルを読み込み、
1. 色マークによる個体IDの自動アライメント
2. キネマティクス特徴量の抽出およびルールベースでのひっかき行動の自動分類
3. ひっかき Bout の集計
を行い、ダッシュボードGUIにインポート可能な形式で CSV ファイルを出力するメインパイプラインです。
"""

import os
import sys
import json
import argparse
from typing import Optional
import numpy as np
import pandas as pd

from config import (
    KEYPOINTS, DEFAULT_FPS, CAGE_MICE_COUNT, MAX_MICE_LIMIT,
    BODY_LENGTH_MIN_MM, BODY_LENGTH_MAX_MM,
)
import color_id
import feature_extraction
import pose_engine

def create_mock_sleap_data(n_frames: int = 150, mice_count: int = 0) -> pd.DataFrame:
    """
    h5py が利用できない環境や、SLEAPデータが指定されていない場合のフォールバックとして、
    指定頭数のマウスが動くダミーのトラッキング座標 DataFrame を生成します。

    Parameters
    ----------
    n_frames   : 生成するフレーム数
    mice_count : 生成するマウスの頭数 (0 = デフォルト4匹)
    """
    n_mice = mice_count if mice_count > 0 else (CAGE_MICE_COUNT if CAGE_MICE_COUNT > 0 else 4)
    n_mice = min(n_mice, MAX_MICE_LIMIT)  # 最大数を超えないようにクランプ
    print(f"[pipeline] [Info] テスト用のダミーポーズデータ(Mock)を生成します。マウス数: {n_mice} 匹")
    records = []

    # n_mice 匹分のケージ内初期中心座標をグリッド状に配置
    # 配置例: 2列 × ceil(n_mice/2) 行
    import math
    cols = 2
    rows = math.ceil(n_mice / cols)
    base_centers_x = [100 + (i % cols) * 140 for i in range(n_mice)]
    base_centers_y = [100 + (i // cols) * 80  for i in range(n_mice)]

    centers = {
        f"track_{i}": (base_centers_x[i], base_centers_y[i])
        for i in range(n_mice)
    }

    # 各キーポイントの相対オフセット (簡易的なスケルトン用)
    kp_offsets = {
        "nose": (0, -35),
        "left_ear": (-8, -28),
        "right_ear": (8, -28),
        "head_center": (0, -25),
        "neck": (0, -20),
        "left_fore_paw": (-10, -10),
        "right_fore_paw": (10, -10),
        "spine_mid": (0, -10),
        "left_hind_paw": (-12, 5),
        "right_hind_paw": (12, 5),
        "tail_base": (0, 0),
    }

    for f in range(n_frames):
        for track_id, (cx, cy) in centers.items():
            row = {
                "frame_idx": f,
                "track": track_id,
                "is_mock": 1
            }

            # 各キーポイントにノイズと動きを付与
            for kp, (ox, oy) in kp_offsets.items():
                kp_x = cx + ox
                kp_y = cy + oy

                # 特定のフレーム（例えば 60-90フレーム目）で、左後ろ足を素早く振動させてひっかきをシミュレート
                if kp == "left_hind_paw" and track_id == "track_0" and 60 <= f <= 90:
                    # 耳に近づける
                    ear_x = cx + kp_offsets["left_ear"][0]
                    ear_y = cy + kp_offsets["left_ear"][1]
                    kp_x = ear_x + 5 + 35 * (f % 2) # X軸方向に大きく高速振動！
                    kp_y = ear_y + 10 + 10 * ((f // 2) % 2)
                
                row[f"{kp}_x"] = float(kp_x)
                row[f"{kp}_y"] = float(kp_y)
                row[f"{kp}_conf"] = 0.98
                
            records.append(row)
            
    return pd.DataFrame(records)

def load_sleap_h5(sleap_path: str) -> pd.DataFrame:
    """
    SLEAP で出力された H5 ファイルをパースし、キーポイント座標を含む pandas DataFrame に展開します。
    """
    if not os.path.exists(sleap_path):
        print(f"[pipeline] [Warning] SLEAPファイル {sleap_path} が見つかりません。")
        return create_mock_sleap_data()

    try:
        import h5py
    except ImportError:
        print("[pipeline] [Warning] h5py ライブラリがインストールされていません。")
        return create_mock_sleap_data()

    try:
        print(f"[pipeline] [Info] SLEAP H5データをロード中: {sleap_path}")
        # 次元正規化（SLEAP は転置が必要）と point_scores の読み込みは
        # pose_engine.parse_sleap_analysis_h5 に一元化し、両経路で同一処理を保証する。
        long_df = pose_engine.parse_sleap_analysis_h5(sleap_path)
        if long_df.empty:
            raise ValueError("H5ファイルから有効な座標レコードを抽出できませんでした。")
        return long_to_wide_format(long_df)

    except Exception as e:
        print(f"[pipeline] [Error] H5読み込み中にエラーが発生しました: {e}")
        return create_mock_sleap_data()

def long_to_wide_format(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    内部共通ロングフォーマット (frame, track, keypoint, x, y, conf, is_mock) から
    既存の処理系が期待するワイドフォーマット (frame_idx, track, is_mock, <kp>_x, <kp>_y, <kp>_conf)
    にデータをピボット変換します。
    """
    if long_df.empty:
        return pd.DataFrame(columns=["frame_idx", "track", "is_mock"])
        
    if "is_mock" not in long_df.columns:
        long_df = long_df.copy()
        long_df["is_mock"] = 0
    
    # is_mock を (frame, track) ごとに先に抽出 (ピボット前に保護)
    is_mock_map = long_df.groupby(["frame", "track"])["is_mock"].first().reset_index()
    
    # ピボット演算による展開 (is_mock はインデックスに含めない)
    pivoted = long_df.pivot(index=["frame", "track"], columns="keypoint", values=["x", "y", "conf"])
    
    # マルチインデックスカラムの平坦化 (<kp>_x, <kp>_y, <kp>_conf)
    flat_cols = [f"{kp}_{val_type}" for val_type, kp in pivoted.columns]
    pivoted.columns = flat_cols
    
    # インデックスをリセットして通常カラムに戻し、frame を frame_idx にリネーム
    wide_df = pivoted.reset_index()
    wide_df = wide_df.rename(columns={"frame": "frame_idx"})
    
    # is_mock を再結合
    wide_df = wide_df.merge(
        is_mock_map.rename(columns={"frame": "frame_idx"}),
        on=["frame_idx", "track"],
        how="left"
    )
    wide_df["is_mock"] = wide_df["is_mock"].fillna(0).astype(int)
    
    return wide_df


def load_cage_mask(mask_path: Optional[str]) -> Optional[dict]:
    """
    Layer 1: ケージマスク JSON を読み込んで返す。
    マスクが指定されていない、または読込失敗時は None を返し、本処理はマスクを使わない。

    返す dict 形式:
        {
            "polygons": [ [(x,y), (x,y), ...], ... ],  # 各多角形は (x,y) タプルのリスト
            "video_width": int,
            "video_height": int,
        }
    """
    if not mask_path:
        return None
    if not os.path.exists(mask_path):
        print(f"[pipeline] [Cage-Mask] [Warning] マスクファイルが見つかりません: {mask_path}")
        return None
    try:
        with open(mask_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        polys_raw = raw.get("exclude_polygons", [])
        # 各多角形は [[x,y], [x,y], ...] のリスト → (x,y) タプルのリストに変換
        polys = [[(float(p[0]), float(p[1])) for p in poly] for poly in polys_raw if len(poly) >= 3]
        if not polys:
            print(f"[pipeline] [Cage-Mask] [Warning] 有効な多角形が見つかりません: {mask_path}")
            return None
        print(f"[pipeline] [Cage-Mask] {len(polys)} 個の除外多角形を読み込みました: {mask_path}")
        return {
            "polygons": polys,
            "video_width": int(raw.get("video_width", 0)),
            "video_height": int(raw.get("video_height", 0)),
        }
    except Exception as e:
        print(f"[pipeline] [Cage-Mask] [Warning] マスク読込失敗 → マスク無効化: {e}")
        return None


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """
    Ray casting アルゴリズムによる点-多角形包含判定。
    polygon: [(x1, y1), (x2, y2), ...] (タプルのリスト)

    マスク領域は通常少数 (~1〜5個) なので numpy 化せずシンプル実装で十分。
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # 半直線が辺と交差するかチェック
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def is_point_in_mask(x: float, y: float, mask: Optional[dict]) -> bool:
    """点 (x, y) がマスクの除外多角形のいずれかに含まれるかを返す。マスクが None の場合は常に False。"""
    if mask is None:
        return False
    for poly in mask["polygons"]:
        if point_in_polygon(x, y, poly):
            return True
    return False


def build_preview_json(
    pose_df: pd.DataFrame,
    fps: float,
    video_path: str,
    preview_duration_s: float,
    sample_interval_s: float = 1.0,
    pixel_to_mm_ratio: Optional[float] = None,
    cage_mask: Optional[dict] = None,
) -> dict:
    """
    初期化確認モード用プレビュー JSON を生成する。

    動画冒頭 preview_duration_s 秒分のポーズデータについて、
    sample_interval_s 秒ごとにサンプリングし、各検出個体の
    バウンディングボックス（キーポイントの凸包 + パディング）と
    キーポイント座標をまとめて返す。

    フロントエンドはこの JSON をもとにシークバーを提供し、
    研究者が任意のフレームで各マウスに ID を割り当てて確認できる。
    """
    BBOX_PAD = 20  # バウンディングボックスのパディング (px)
    # ── 誤検出フィルタリング閾値（Layer 3: 解剖学的妥当性チェック） ─────────────
    # DLC SuperAnimal 等のゼロショットモデルは、ケージ装置（給水器・餌器・配管など）を
    # マウスとして誤認識することがある。これらの誤検出は以下の特徴がある:
    #   1. 平均キーポイント信頼度が低い (典型的に 0.3 未満)
    #   2. 有効キーポイント数が少ない (装置の一部しか「マウスっぽく」見えないため)
    #   3. bbox のアスペクト比がマウスらしくない (装置は正方形/円形が多い)
    #   4. 体長 (neck-tail_base 距離) が成体マウスの範囲を逸脱
    # 多段フィルタで明らかな誤検出をプレビューから除外する。
    MIN_AVG_KP_CONF      = 0.25  # 検出全体の平均キーポイント信頼度の下限
    MIN_VALID_KP_COUNT   = 4     # 有効キーポイント数の下限 (KEYPOINTS の約 1/3 以上)
    # bbox アスペクト比: マウスは細長い (1:1.2〜1:4 程度)。正方形に近い装置検出を除外する
    # 比率は max(w,h) / min(w,h) で計算。1.05 未満 = ほぼ正方形 → 装置の可能性大
    MIN_BBOX_ELONGATION  = 1.05  # 最低でもこの程度の縦横比を要求（緩めに設定）
    # 体長レンジ (px): config.BODY_LENGTH_*_MM を pixel_to_mm_ratio で換算した値。
    # build_preview_json は px_to_mm を引数で受けないので、引数で受け取った値を使う。
    # 値が指定されない場合は体長チェックをスキップ。
    _body_min_mm = BODY_LENGTH_MIN_MM
    _body_max_mm = BODY_LENGTH_MAX_MM

    # 動画の実解像度を取得（フロントエンドでの座標変換に使用）
    # NOTE: 正しい OpenCV プロパティ名は CAP_PROP_FRAME_WIDTH/HEIGHT。
    # CAP_PROP_WIDTH / CAP_PROP_HEIGHT は存在しないため AttributeError → 1920x1080 へ fallback していた。
    video_w, video_h = 1920, 1080
    try:
        import cv2 as _cv2
        cap = _cv2.VideoCapture(video_path)
        if cap.isOpened():
            w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                video_w, video_h = w, h
            cap.release()
    except Exception as e:
        print(f"[pipeline] [Warning] 動画解像度取得失敗 → フォールバック {video_w}x{video_h}: {e}")

    max_frame = int(preview_duration_s * fps)
    step = max(1, int(sample_interval_s * fps))
    sampled_frames = list(range(0, max_frame + 1, step))

    frames_dict = {}
    skipped_low_conf = 0  # 解剖学的妥当性チェックで除外された検出数（ログ用）
    skipped_mask     = 0  # ケージマスクで除外された検出数（ログ用）
    for frame_idx in sampled_frames:
        rows = pose_df[pose_df["frame_idx"] == frame_idx]
        if rows.empty:
            continue

        detections = []
        for _, row in rows.iterrows():
            # キーポイント座標を取得（NaN を除外）
            xs = [float(row[f"{kp}_x"]) for kp in KEYPOINTS
                  if f"{kp}_x" in row.index and not pd.isna(row.get(f"{kp}_x"))]
            ys = [float(row[f"{kp}_y"]) for kp in KEYPOINTS
                  if f"{kp}_y" in row.index and not pd.isna(row.get(f"{kp}_y"))]

            # バウンディングボックス：全キーポイントを囲む最小矩形 + パディング
            bbox = None
            if len(xs) >= 2 and len(ys) >= 2:
                bbox = [
                    round(max(0.0, min(xs) - BBOX_PAD), 1),
                    round(max(0.0, min(ys) - BBOX_PAD), 1),
                    round(max(xs) + BBOX_PAD, 1),
                    round(max(ys) + BBOX_PAD, 1),
                ]

            # ── Layer 1: ケージマスク除外 ───────────────────────────────────
            # bbox 中心がマスク多角形内にあれば、装置の誤検出とみなしてスキップ。
            # bbox が無い (キーポイント数<2) 場合は後段の解剖学チェックで弾かれる。
            if cage_mask is not None and bbox is not None:
                bcx = (bbox[0] + bbox[2]) * 0.5
                bcy = (bbox[1] + bbox[3]) * 0.5
                if is_point_in_mask(bcx, bcy, cage_mask):
                    skipped_mask += 1
                    continue

            # キーポイント辞書（x, y, conf）
            kps_out = {}
            for kp in KEYPOINTS:
                xcol, ycol, ccol = f"{kp}_x", f"{kp}_y", f"{kp}_conf"
                if xcol in row.index and not pd.isna(row.get(xcol)):
                    kps_out[kp] = {
                        "x": round(float(row[xcol]), 1),
                        "y": round(float(row[ycol]), 1),
                        "conf": round(float(row[ccol]) if (ccol in row.index and not pd.isna(row.get(ccol))) else 0.5, 3),
                    }

            # ── 誤検出フィルタ ─────────────────────────────────────────────
            # ケージ装置などをマウスと誤認識した検出を除外する。多段判定:
            #   (A) 有効キーポイント数: 装置の部分検出（数点のみ）を除外
            #   (B) 平均キーポイント信頼度: 低信頼度の誤検出を除外
            #   (C) bbox アスペクト比: 正方形/円形の装置検出を除外
            #   (D) 体長レンジ: 成体マウスの体長範囲外を除外
            reject_reason = None
            if len(kps_out) < MIN_VALID_KP_COUNT:
                reject_reason = f"few_kps({len(kps_out)}<{MIN_VALID_KP_COUNT})"
            else:
                avg_conf = sum(kp["conf"] for kp in kps_out.values()) / len(kps_out)
                if avg_conf < MIN_AVG_KP_CONF:
                    reject_reason = f"low_conf({avg_conf:.2f}<{MIN_AVG_KP_CONF})"
                elif bbox is not None:
                    # (C) bbox アスペクト比チェック
                    bw = max(1e-6, bbox[2] - bbox[0])
                    bh = max(1e-6, bbox[3] - bbox[1])
                    elongation = max(bw, bh) / min(bw, bh)
                    if elongation < MIN_BBOX_ELONGATION:
                        reject_reason = f"too_square(elong={elongation:.2f})"
                    # (D) 体長レンジチェック (pixel_to_mm_ratio が指定された場合のみ)
                    elif pixel_to_mm_ratio is not None and pixel_to_mm_ratio > 0 \
                            and "neck" in kps_out and "tail_base" in kps_out:
                        nx, ny = kps_out["neck"]["x"], kps_out["neck"]["y"]
                        tx, ty = kps_out["tail_base"]["x"], kps_out["tail_base"]["y"]
                        body_px = float(np.hypot(tx - nx, ty - ny))
                        body_mm = body_px * pixel_to_mm_ratio
                        if body_mm < _body_min_mm:
                            reject_reason = f"body_too_short({body_mm:.0f}mm<{_body_min_mm}mm)"
                        elif body_mm > _body_max_mm:
                            reject_reason = f"body_too_long({body_mm:.0f}mm>{_body_max_mm}mm)"
            if reject_reason is not None:
                skipped_low_conf += 1
                continue
            # avg_conf を後段の detection に含めるため再計算（reject されなかった場合のみ）
            avg_conf = sum(kp["conf"] for kp in kps_out.values()) / len(kps_out)

            mouse_id = None
            if "mouse_id" in row.index and not pd.isna(row.get("mouse_id")):
                mouse_id = int(row["mouse_id"])

            detections.append({
                "track": str(row["track"]),
                "mouse_id": mouse_id,          # カラーID自動推定値（参考）
                "bbox": bbox,
                "keypoints": kps_out,
                "avg_conf": round(avg_conf, 3),  # デバッグ用: 平均信頼度
            })

        frames_dict[str(frame_idx)] = {
            "timestamp_s": round(frame_idx / fps, 3),
            "detections": detections,
        }

    # フィルタ統計をログ出力（誤検出の発生頻度を研究者が把握できるように）
    if skipped_mask > 0:
        print(f"[pipeline] [Preview-Filter] ケージマスクで {skipped_mask} 件の装置誤検出を除外しました。")
    if skipped_low_conf > 0:
        print(f"[pipeline] [Preview-Filter] 解剖学的妥当性チェックで {skipped_low_conf} 件の検出を "
              f"プレビューから除外しました。閾値:")
        print(f"[pipeline]   - 有効KP数 >= {MIN_VALID_KP_COUNT}, 平均信頼度 >= {MIN_AVG_KP_CONF}")
        print(f"[pipeline]   - bbox 縦横比 >= {MIN_BBOX_ELONGATION} (正方形検出を除外)")
        if pixel_to_mm_ratio is not None and pixel_to_mm_ratio > 0:
            print(f"[pipeline]   - 体長 (neck→tail_base) ∈ [{_body_min_mm}, {_body_max_mm}] mm")
        else:
            print(f"[pipeline]   - 体長レンジチェック: 無効 (pixel_to_mm_ratio 未指定)")

    return {
        "video": os.path.basename(video_path),
        "fps": round(fps, 4),
        "video_width": video_w,
        "video_height": video_h,
        "preview_duration_s": preview_duration_s,
        "sample_interval_s": sample_interval_s,
        "sampled_frames": sampled_frames,
        "frames": frames_dict,
    }


def _collect_library_versions() -> dict:
    """解析に関与する主要ライブラリのバージョンを収集する（best-effort）。"""
    import platform
    versions = {"python": platform.python_version()}
    for mod_name, key in [
        ("numpy", "numpy"), ("pandas", "pandas"), ("cv2", "opencv"),
        ("torch", "torch"), ("ultralytics", "ultralytics"),
        ("deeplabcut", "deeplabcut"), ("sleap", "sleap"), ("h5py", "h5py"),
    ]:
        try:
            mod = __import__(mod_name)
            versions[key] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass  # 未インストールのライブラリは記録しない
    return versions


def _get_git_commit() -> Optional[str]:
    """現在の git コミットハッシュを取得する（リポジトリ外・git 未導入なら None）。"""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


# 各出力CSVの列の意味を機械可読・人間可読の両方で残すためのデータ辞書。
# 解析結果CSV単体では各列が物理的に何を表すか（単位・算出式）が判別できないため、
# 結果と同じディレクトリに列定義CSVを書き出し、学術的な再現性とレビュー容易性を担保する。
# ここでの定義は feature_extraction.py の実装（compute_summary_stats / detect_scratch_bouts）
# と厳密に一致させること。実装を変更した場合は本辞書も同時に更新する。
COLUMN_DEFINITIONS = [
    # file, column, unit, description(JA), formula
    # ── サマリー統計 (_summary.csv): 個体ごとに観察時間で正規化した集計値 ──
    ("summary", "mouse_id", "-", "個体識別ID（色＋尾の帯本数で割り当て）。0は未解決ID。", "color_id による割当"),
    ("summary", "observation_duration_s", "秒", "観察時間。姿勢データが存在するフレーム範囲の総時間。", "(最終frame_idx - 最初frame_idx + 1) / fps"),
    ("summary", "observation_duration_min", "分", "観察時間（分）。各種レートの分母。", "observation_duration_s / 60"),
    ("summary", "n_bouts", "回", "スクラッチbout（連続したひっかき動作のまとまり）の検出総数。", "その個体のbout行数"),
    ("summary", "total_scratch_duration_s", "秒", "全boutの合計時間。観察中に実際にひっかいていた総時間。", "Σ bout_duration_s"),
    ("summary", "scratch_rate_bouts_per_min", "回/分", "ひっかき頻度。1分あたり何回のboutが起きたか（観察時間で正規化）。", "n_bouts / observation_duration_min"),
    ("summary", "scratch_rate_s_per_min", "秒/分", "ひっかき強度。1分あたり何秒ひっかいていたか（観察時間で正規化）。", "total_scratch_duration_s / observation_duration_min"),
    ("summary", "mean_bout_duration_s", "秒", "1回のboutの平均持続時間。", "mean(bout_duration_s)"),
    ("summary", "sem_bout_duration_s", "秒", "bout持続時間の標準誤差（個体内ばらつきの指標）。", "std(bout_duration_s) / sqrt(n_bouts)"),
    ("summary", "median_bout_duration_s", "秒", "bout持続時間の中央値（外れ値に頑健な代表値）。", "median(bout_duration_s)"),
    ("summary", "longest_bout_s", "秒", "最も長く続いた単一boutの持続時間。", "max(bout_duration_s)"),
    ("summary", "fps_used", "フレーム/秒", "時間換算に用いた動画のフレームレート。", "動画メタデータ由来"),
    # ── Bout情報 (_bouts.csv): 検出された個々のスクラッチboutの明細 ──
    ("bouts", "mouse_id", "-", "個体識別ID。", "color_id による割当"),
    ("bouts", "bout_id", "-", "その個体内でのbout連番。", "0始まりの通し番号"),
    ("bouts", "bout_start_frame", "フレーム", "boutの開始フレーム番号。", "-"),
    ("bouts", "bout_end_frame", "フレーム", "boutの終了フレーム番号。", "-"),
    ("bouts", "bout_start_time_s", "秒", "bout開始時刻。", "bout_start_frame / fps"),
    ("bouts", "bout_end_time_s", "秒", "bout終了時刻。", "bout_end_frame / fps"),
    ("bouts", "bout_duration_s", "秒", "boutの持続時間。", "bout_n_frames / fps"),
    ("bouts", "bout_n_frames", "フレーム", "boutを構成するフレーム数。", "bout_end_frame - bout_start_frame + 1"),
    ("bouts", "fps_used", "フレーム/秒", "時間換算に用いたフレームレート。", "動画メタデータ由来"),
]


def write_column_definitions(path: str) -> None:
    """
    出力CSV（summary / bouts）各列の意味・単位・算出式を記したデータ辞書CSVを書き出す。

    解析結果CSVを後から人間が検証・査読する際に、各列が物理的に何を表すかを
    同梱ファイルで即座に参照できるようにするためのもの。データ本体のCSVスキーマは
    機械可読性を保つため変更せず、説明は本サイドカーCSVに分離する。
    """
    cols_df = pd.DataFrame(
        COLUMN_DEFINITIONS,
        columns=["出力ファイル", "列名", "単位", "意味", "算出式"],
    )
    cols_df.to_csv(path, index=False, encoding="utf-8-sig")


def write_provenance_manifest(manifest_path: str, args, fps: float,
                              output_files: dict,
                              id_quality: dict = None) -> None:
    """
    [A-4] 再現性のためのプロベナンス manifest を JSON で書き出す。

    解析結果 CSV 単体では「どの閾値・校正値・エンジン・バージョンで生成されたか」が
    判別できないため、結果と同じディレクトリに manifest を残す。学術論文の Methods
    記載および結果の再現に必要な全パラメータを 1 ファイルに集約する。
    """
    import datetime
    import config  # 実行時にオーバーライドされた可能性があるため live 値を読む

    manifest = {
        "schema": "mouse-scratch-provenance/v1",
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "input": {
            "video": os.path.basename(args.video),
            "video_abspath": os.path.abspath(args.video) if os.path.exists(args.video) else args.video,
            "sleap_h5": args.sleap,
        },
        "pose_estimation": {
            "engine": args.engine,
            "model_path": args.model_path,
            "device": args.device,
            "dlc_superanimal": args.dlc_superanimal if args.engine == "dlc" else None,
            "dlc_detector": args.dlc_detector if args.engine == "dlc" else None,
            "dlc_pose_model": args.dlc_pose_model if args.engine == "dlc" else None,
            "dlc_max_individuals": (args.dlc_max_individuals or args.mice_count or 4) if args.engine == "dlc" else None,
        },
        "calibration": {
            "pixel_to_mm_ratio": args.px_to_mm,
            "pixel_to_mm_is_default": bool(abs(args.px_to_mm - 0.25) < 1e-9),
            "fps": fps,
            "mice_count": args.mice_count,
            "roi": (
                {"x1": args.roi_x1, "y1": args.roi_y1, "x2": args.roi_x2, "y2": args.roi_y2}
                if all(v is not None for v in [args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2])
                else None
            ),
            "cage_mask": args.cage_mask,
        },
        "scratch_thresholds": {
            "SCRATCH_MIN_VELOCITY_MM_S":      config.SCRATCH_MIN_VELOCITY_MM_S,
            "SCRATCH_MAX_HEAD_DIST_MM":       config.SCRATCH_MAX_HEAD_DIST_MM,
            "SCRATCH_MIN_OSCILLATIONS":       config.SCRATCH_MIN_OSCILLATIONS,
            "SCRATCH_MIN_OSC_AMPLITUDE_MM":   config.SCRATCH_MIN_OSC_AMPLITUDE_MM,
            "SCRATCH_VELOCITY_SMOOTH_FRAMES": config.SCRATCH_VELOCITY_SMOOTH_FRAMES,
            "SCRATCH_WINDOW_FRAMES":          config.SCRATCH_WINDOW_FRAMES,
            "BOUT_MIN_FRAMES":                config.BOUT_MIN_FRAMES,
            "BOUT_MERGE_GAP":                 config.BOUT_MERGE_GAP,
            "INTERPOLATION_MAX_GAP":          config.INTERPOLATION_MAX_GAP,
            "MIN_CONFIDENCE":                 config.MIN_CONFIDENCE,
            "HIND_PAW_MIN_CONF":              config.HIND_PAW_MIN_CONF,
        },
        "anatomical_checks": {
            "ENABLE_ANATOMICAL_CHECKS":  config.ENABLE_ANATOMICAL_CHECKS,
            "BODY_LENGTH_MIN_MM":        config.BODY_LENGTH_MIN_MM,
            "BODY_LENGTH_MAX_MM":        config.BODY_LENGTH_MAX_MM,
            "MAX_TELEPORT_BODY_LENGTHS": config.MAX_TELEPORT_BODY_LENGTHS,
        },
        "environment": {
            "library_versions": _collect_library_versions(),
            "git_commit": _get_git_commit(),
            "working_directory": os.getcwd(),
        },
        "outputs": output_files,
        # [A-3] カラーID品質サマリー（個体別平均信頼度・Swap件数など）
        "color_id_quality": id_quality if id_quality else None,
    }

    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[pipeline] プロベナンス manifest 保存完了: {manifest_path}")
        if manifest["calibration"]["pixel_to_mm_is_default"]:
            print("[pipeline] [Warning] ⚠️ manifest に記録された px-to-mm はデフォルト値です（未校正）。")
    except Exception as e:
        print(f"[pipeline] [Warning] manifest 書き出しに失敗しました: {e}")


def main():
    parser = argparse.ArgumentParser(description="Mouse Scratch Detection System — End-to-End Pipeline")
    parser.add_argument("--video", type=str, required=True, help="解析対象の動画ファイルパス (.mp4 / .mov / .avi など OpenCV対応形式)")
    parser.add_argument("--sleap", type=str, default=None, help="SLEAPから出力された H5 ファイルパス (省略した場合は動画から自動的に姿勢推定を実行)")
    parser.add_argument("--engine", type=str, default="yolov8", choices=["sleap", "yolov8", "dlc", "mock"], help="姿勢推定の自動実行に使用するエンジン (デフォルト: yolov8)")
    parser.add_argument("--model_path", type=str, default=None, help="姿勢推定モデルの重みファイル/設定パス (DLC ではカスタム config.yaml を指定可)")
    parser.add_argument("--dlc-superanimal", type=str, default="superanimal_topviewmouse",
                        choices=["superanimal_topviewmouse", "superanimal_quadruped"],
                        help="DLC エンジン使用時の Model Zoo モデル名 (デフォルト: superanimal_topviewmouse)")
    parser.add_argument("--dlc-detector", type=str, default="fasterrcnn_resnet50_fpn_v2",
                        help="DLC SuperAnimal の物体検出器モデル名 (デフォルト: fasterrcnn_resnet50_fpn_v2)")
    parser.add_argument("--dlc-pose-model", type=str, default="hrnet_w32",
                        help="DLC SuperAnimal のポーズ推定モデル名 (デフォルト: hrnet_w32)")
    parser.add_argument("--dlc-max-individuals", type=int, default=0,
                        help="DLC SuperAnimal multi-animal 推論の最大個体数。"
                             "0 = --mice-count を流用 (それも 0 なら 4)。集団飼育の頭数と一致させること。")
    parser.add_argument("--out_dir", type=str, default="outputs", help="CSV結果を出力するディレクトリ")
    parser.add_argument("--fps", type=int, default=None, help="動画のFPS設定 (configのデフォルト値をオーバーライド)")
    parser.add_argument("--px-to-mm", type=float, default=0.25, help="ピクセルからミリメートルへのキャリブレーション比率 (mm/pixel)")
    parser.add_argument("--gt", type=str, default=None, help="グラウンドトゥルース CSV パス (省略時は評価をスキップ)")
    parser.add_argument("--gt-format", type=str, default="frame", choices=["frame", "bout"],
                        help="GT CSV のフォーマット (frame または bout, デフォルト: frame)")
    parser.add_argument("--iou", type=float, default=0.5, help="Bout 評価の Temporal IoU 閾値 (デフォルト: 0.5)")
    parser.add_argument("--mice-count", type=int, default=0,
                        help="ケージ内マウスの頭数 (0 = 自動検出。動画途中から増減する場合も 0 を推奨)")
    # ROI(関心領域): 指定すると矩形外の検出は学術判定から除外される (px 単位)
    parser.add_argument("--roi-x1", type=int, default=None, help="ROI 矩形 左上 X 座標 (px, 任意)")
    parser.add_argument("--roi-y1", type=int, default=None, help="ROI 矩形 左上 Y 座標 (px, 任意)")
    parser.add_argument("--roi-x2", type=int, default=None, help="ROI 矩形 右下 X 座標 (px, 任意)")
    parser.add_argument("--roi-y2", type=int, default=None, help="ROI 矩形 右下 Y 座標 (px, 任意)")

    # ── フェーズ1 初期化確認モード ─────────────────────────────────────────────
    # --preview-only: 動画冒頭のみ解析してバウンディングボックス付き JSON を出力する。
    #                 出力後にフロントエンドで研究者が各マウスに ID を割り当て、
    #                 /api/confirm_ids で本解析を開始する。
    parser.add_argument("--preview-only", action="store_true",
                        help="プレビューモード: 最初の --preview-duration 秒のみ解析し、"
                             "_preview.json を出力して終了する（本解析は実行しない）")
    parser.add_argument("--preview-duration", type=float, default=300.0,
                        help="プレビュー解析の長さ（秒）。デフォルト: 300秒（5分）")
    # --confirmed-mapping: フロントエンドで研究者が確認済みの track→mouse_id マッピング。
    #                      JSON 文字列形式 {"track_0": 1, "track_1": 2, ...}
    #                      指定するとカラーID自動分類をスキップしてこのマッピングを使用する。
    parser.add_argument("--confirmed-mapping", type=str, default=None,
                        help='研究者確認済み track→mouse_id マッピング (JSON文字列: {"track_0": 1, ...})')
    parser.add_argument("--cage-mask", type=str, default=None,
                        help="Layer 1 ケージマスク JSON (outputs/<videoname>_cage_mask.json) のパス。"
                             "多角形内に bbox 中心を持つ検出は装置誤検出として除外される。")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                        help="推論デバイスを明示指定 (cuda / cpu)。省略時は GPU 互換性を自動判定して選択。"
                             "RTX 5080 など新世代 GPU を確実に使いたい場合は --device cuda を明示することで"
                             "GPU 互換性チェックをスキップできます（非互換の場合は CUDA エラーで停止）。")

    args = parser.parse_args()

    # ── ROI 値を環境変数経由で feature_extraction に伝達 ──
    # config.py の動的読み込みパターンを踏襲（モジュール間のグローバル共有を避ける）
    if all(v is not None for v in [args.roi_x1, args.roi_y1, args.roi_x2, args.roi_y2]):
        os.environ["ROI_X1"] = str(args.roi_x1)
        os.environ["ROI_Y1"] = str(args.roi_y1)
        os.environ["ROI_X2"] = str(args.roi_x2)
        os.environ["ROI_Y2"] = str(args.roi_y2)
        print(f"[pipeline] [Info] ROI 指定を有効化: ({args.roi_x1},{args.roi_y1})-({args.roi_x2},{args.roi_y2})")

    # 環境変数または引数でFPSなどを一時的にオーバーライドする設定
    if args.fps is not None:
        os.environ["DEFAULT_FPS"] = str(args.fps)

    # 出力先ディレクトリの作成
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n========================================================")
    print(" Mouse Scratch Detection Pipeline")
    print("========================================================\n")

    # ⚠️ デフォルトの pixel→mm キャリブレーション値を使用している場合、学術論文用途では校正必須を警告
    if abs(args.px_to_mm - 0.25) < 1e-9:
        print("[pipeline] [Warning] ⚠️ pixel-to-mm 比率がデフォルト値 0.25 mm/px で実行されています。")
        print("[pipeline]            学術論文用途では、各カメラ設置位置で物差し撮影による実測校正を行い")
        print("[pipeline]            --px-to-mm <実測値> オプションで指定してください。未校正の場合、")
        print("[pipeline]            mm/s 速度・mm 距離の絶対値は無意味となり、実験間比較に使用できません。")

    # 動画から実際の FPS を読み取る (cv2 が使えない場合は DEFAULT_FPS にフォールバック)
    fps = float(args.fps) if args.fps is not None else None
    if fps is None:
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(args.video)
            if _cap.isOpened():
                fps = _cap.get(_cv2.CAP_PROP_FPS)
                _cap.release()
                if fps <= 0:
                    fps = None
        except Exception:
            pass
    if fps is None or fps <= 0:
        fps = float(DEFAULT_FPS)
        print(f"[pipeline] [Info] FPS を動画から取得できませんでした。デフォルト値を使用します: {fps} fps")
    else:
        print(f"[pipeline] [Info] 動画 FPS: {fps}")

    # 1. 姿勢推定ポーズデータのロード（または自動実行）
    if args.sleap is not None:
        if args.sleap.lower() == "dummy":
            pose_df = create_mock_sleap_data()
        else:
            pose_df = load_sleap_h5(args.sleap)
    else:
        print(f"[pipeline] [Info] SLEAPファイルが指定されていません。動画 {args.video} から姿勢推定を自動実行します。")
        print(f"[pipeline] 使用エンジン: {args.engine.upper()} (モデルパス: {args.model_path})")
        
        # エンジンのインスタンス化
        # --device が明示指定された場合はそれを使用。省略時は detect_gpu_device() が互換性チェック後に選択。
        device_override = args.device  # None = 自動 / "cuda" or "cpu" = 強制
        if args.engine == "sleap":
            engine = pose_engine.SleapEngine(model_path=args.model_path, device=device_override)
        elif args.engine == "yolov8":
            engine = pose_engine.Yolov8PoseEngine(model_path=args.model_path, device=device_override)
        elif args.engine == "dlc":
            # multi-animal 上限の決定: --dlc-max-individuals 優先、未指定(0)なら --mice-count、
            # それも 0 なら 4（典型的な集団飼育頭数）にフォールバック。
            _dlc_max_ind = args.dlc_max_individuals or args.mice_count or 4
            engine = pose_engine.DeepLabCutEngine(
                model_path=args.model_path,
                device=device_override,
                superanimal_name=args.dlc_superanimal,
                detector_name=args.dlc_detector,
                pose_model_name=args.dlc_pose_model,
                max_individuals=_dlc_max_ind,
            )
        else:
            engine = pose_engine.MockPoseEngine(model_path=args.model_path, device=device_override)
            
        # ── プレビューモード時は動画をトリミングして処理量を削減 ──────────────────
        # preview_only が True の場合、冒頭 preview_duration 秒分だけを一時ファイルに
        # 切り出してエンジンに渡す。DLC は動画ファイル全体をシーケンシャルに読むため、
        # フレーム数を絞ることで処理時間を大幅に短縮する（71,961フレーム → 約9,000フレーム）。
        video_for_engine = args.video
        _trimmed_tmp = None
        if args.preview_only and args.preview_duration and os.path.exists(args.video):
            try:
                import cv2 as _cv2
                _cap_trim = _cv2.VideoCapture(args.video)
                _trim_fps   = _cap_trim.get(_cv2.CAP_PROP_FPS) or fps
                _trim_total = int(_cap_trim.get(_cv2.CAP_PROP_FRAME_COUNT))
                _max_frames = min(int(args.preview_duration * _trim_fps) + 1, _trim_total)
                _cap_trim.release()

                # トリミング先の一時ファイルを作成 (常に .mp4 を使用。cv2 の mp4v コーデックは
                # .mov/.avi 等の元形式によらず確実に書き出せる。DLC も .mp4 を確実に読める)
                _trimmed_tmp = os.path.join(args.out_dir, "__preview_trim.mp4")
                _cap_src = _cv2.VideoCapture(args.video)
                _w = int(_cap_src.get(_cv2.CAP_PROP_FRAME_WIDTH))
                _h = int(_cap_src.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                _fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
                _writer = _cv2.VideoWriter(_trimmed_tmp, _fourcc, _trim_fps, (_w, _h))
                _written = 0
                while _written < _max_frames:
                    _ret, _frm = _cap_src.read()
                    if not _ret:
                        break
                    _writer.write(_frm)
                    _written += 1
                    # 進捗を10秒ごとに出力
                    if _written % max(1, int(_trim_fps * 10)) == 0:
                        _pct = int(_written / _max_frames * 100)
                        print(f"[pipeline] [Preview-Trim] 動画トリミング中: {_written}/{_max_frames} フレーム ({_pct}%)")
                _cap_src.release()
                _writer.release()
                print(f"[pipeline] [Preview-Trim] プレビュー用トリム動画を作成: "
                      f"{_trimmed_tmp} ({_written} フレーム / {_written/_trim_fps:.1f} 秒)")
                video_for_engine = _trimmed_tmp
            except Exception as _e:
                print(f"[pipeline] [Preview-Trim] トリミング失敗（元動画をそのまま使用）: {_e}")
                video_for_engine = args.video

        # 姿勢推定の実行 (ロングフォーマットが返る)
        print(f"\n[Progress: 10%] 姿勢推定を開始します...")
        long_df = engine.run(video_for_engine)
        print(f"\n[Progress: 75%] 姿勢推定完了 — 特徴量抽出中...")

        # トリミング一時ファイルを削除
        if _trimmed_tmp and os.path.exists(_trimmed_tmp):
            try:
                os.remove(_trimmed_tmp)
                print(f"[pipeline] [Preview-Trim] 一時ファイルを削除しました: {_trimmed_tmp}")
            except Exception:
                pass

        # ワイドフォーマットへの変換
        pose_df = long_to_wide_format(long_df)
        
    print(f"[pipeline] 初期ロードデータ数: {len(pose_df)} 行")

    # ── Layer 1: ケージマスクをロード ─────────────────────────────────────
    # マスクが指定されていれば、ポーズデータからマスク内のキーポイント中心を持つ
    # 検出行を除外する。これにより装置誤検出が後段の特徴量抽出にも残らない。
    cage_mask = load_cage_mask(args.cage_mask)
    if cage_mask is not None and not pose_df.empty:
        # bbox 中心の代理として spine_mid (なければ head_center, さらに無ければ neck) を使う
        # 各行のキーポイント中心を計算 → 多角形内なら除外
        def _row_in_mask(row):
            for kp in ("spine_mid", "head_center", "neck", "tail_base"):
                xc = row.get(f"{kp}_x"); yc = row.get(f"{kp}_y")
                if xc is not None and yc is not None and not pd.isna(xc) and not pd.isna(yc):
                    return is_point_in_mask(float(xc), float(yc), cage_mask)
            return False
        before_n = len(pose_df)
        in_mask = pose_df.apply(_row_in_mask, axis=1)
        pose_df = pose_df[~in_mask].copy()
        removed = before_n - len(pose_df)
        if removed > 0:
            print(f"[pipeline] [Cage-Mask] {removed} 行 ({removed/before_n*100:.1f}%) の "
                  f"マスク内検出を本解析データから除外しました。")

    # ── フェーズ1: プレビューモード（--preview-only 指定時） ─────────────────
    # 動画冒頭 preview_duration 秒分のみを解析し、バウンディングボックス付き JSON を
    # 出力して終了する。フロントエンドで研究者が確認後、/api/confirm_ids で本解析を開始する。
    if args.preview_only:
        print(f"\n--- [Preview Mode] 初期化確認用プレビュー生成中 ---")
        preview_max_frame = int(args.preview_duration * fps)
        pose_df_preview = pose_df[pose_df["frame_idx"] <= preview_max_frame].copy()
        print(f"[pipeline] プレビュー対象: 最初 {args.preview_duration:.0f} 秒 "
              f"({preview_max_frame} フレーム) / {len(pose_df_preview)} 行のポーズデータ")

        # プレビュー用カラーID割り当て（本解析と同一ロジックを最初 N フレームに適用）
        try:
            if os.path.exists(args.video):
                pose_df_preview = color_id.assign_track_ids(
                    args.video, pose_df_preview, mice_count=args.mice_count
                )
            else:
                pose_df_preview["mouse_id"] = pose_df_preview["track"].map(
                    lambda t: int(t.split("_")[-1]) + 1 if t.split("_")[-1].isdigit() else 1
                )
        except Exception as e:
            print(f"[pipeline] [Warning] プレビュー用カラーID割り当て中にエラー: {e}")
            pose_df_preview["mouse_id"] = pose_df_preview["track"].map(
                lambda t: int(t.split("_")[-1]) + 1 if t.split("_")[-1].isdigit() else 1
            )

        # バウンディングボックス付きプレビュー JSON を生成・保存
        video_basename_p = os.path.splitext(os.path.basename(args.video))[0]
        preview_data = build_preview_json(
            pose_df_preview, fps, args.video,
            preview_duration_s=args.preview_duration,
            sample_interval_s=1.0,  # 1 秒ごとにサンプリング
            pixel_to_mm_ratio=args.px_to_mm,  # 体長レンジチェックに使用
            cage_mask=cage_mask,              # Layer 1: ケージ装置除外
        )
        preview_json_path = os.path.join(args.out_dir, f"{video_basename_p}_preview.json")
        with open(preview_json_path, "w", encoding="utf-8") as fp:
            json.dump(preview_data, fp, ensure_ascii=False, separators=(",", ":"))

        print(f"[pipeline] [Preview] プレビューJSON 保存完了: {preview_json_path}")
        print(f"[pipeline] [Preview] サンプリングフレーム数: {len(preview_data['sampled_frames'])} フレーム "
              f"(1秒間隔 / 最初 {args.preview_duration:.0f}秒)")
        print(f"[pipeline] [Preview] フロントエンドで各マウスに ID を割り当て後、/api/confirm_ids を呼び出してください。")
        print("\n[Preview] プレビュー生成が完了しました。本解析は実行しません。")
        return  # プレビューモードはここで終了

    # ── 確認済みIDマッピングの解析（--confirmed-mapping 指定時） ──────────────
    # フロントエンドで研究者が確認したマッピングを JSON 文字列から辞書に変換する。
    confirmed_mapping = None
    if args.confirmed_mapping:
        try:
            raw_map = json.loads(args.confirmed_mapping)
            confirmed_mapping = {str(k): int(v) for k, v in raw_map.items()}
            print(f"[pipeline] [Info] 確認済みIDマッピングを適用します: {confirmed_mapping}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[pipeline] [Warning] --confirmed-mapping の解析に失敗しました: {e}")
            print("[pipeline] [Warning] カラーID自動分類にフォールバックします。")

    # 2. 動画のカラーIDに基づく個体トラッキング自動アライメント
    print("\n--- [Step 1] カラーIDの自動アライメント実行中 ---")
    try:
        # 動画が存在しない場合はダミー動作に切り替え
        if not os.path.exists(args.video):
            print(f"[pipeline] [Warning] 動画ファイル {args.video} が見つかりません。")
            print("[pipeline] カラーID割り当てをスキップし、元のトラックIDをそのままmouse_idにアライメントします。")
            
            # 代替: track の連番を mouse_id にマップ
            pose_df["mouse_id"] = pose_df["track"].map(
                lambda t: int(t.split("_")[-1]) + 1 if t.split("_")[-1].isdigit() else 1
            )
            aligned_df = pose_df
        else:
            # MOVファイルを含む、OpenCVが対応する主要動画形式のリスト
            SUPPORTED_VIDEO_EXTS = {
                '.mp4', '.mov', '.avi', '.mpeg', '.mpg', '.mkv',
                '.wmv', '.m4v', '.ts', '.mts', '.m2ts', '.flv'
            }
            video_ext = os.path.splitext(args.video)[1].lower()
            if video_ext not in SUPPORTED_VIDEO_EXTS:
                print(f"[pipeline] [Warning] 動画形式 '{video_ext}' は公式テスト済み形式外の可能性がありますが、OpenCVでの読み込みを試みます。")
            if video_ext == '.mov':
                print(f"[pipeline] [Info] MOV形式を検出しました。OpenCVはMOVをサポートしていますが、H.264コーデックでエンコードされている必要があります。")
            aligned_df = color_id.assign_track_ids(
                args.video, pose_df,
                mice_count=args.mice_count,
                initial_mapping=confirmed_mapping,
            )
    except Exception as e:
        print(f"[pipeline] [Error] カラーID割り当て中に予期しないエラーが発生しました: {e}")
        print("[pipeline] トラックIDから暫定IDを割り当てて処理を続行します。")
        pose_df["mouse_id"] = pose_df["track"].map(
            lambda t: int(t.split("_")[-1]) + 1 if t.split("_")[-1].isdigit() else 1
        )
        aligned_df = pose_df

    # [A-3] カラーID品質レポートを取得する。feature_extraction が DataFrame をコピーすると
    # .attrs が失われるため、特徴量抽出の前にここで退避しておく。
    id_quality = dict(getattr(aligned_df, "attrs", {}).get("id_quality", {}))

    # 3. 特徴量抽出およびスクラッチ・Bout検出の実行
    print(f"\n[Progress: 80%] 特徴量抽出およびスクラッチ判定実行中...")
    print("\n--- [Step 2] 特徴量抽出およびスクラッチ判定実行中 ---")
    processed_df, bouts_df, summary_df = feature_extraction.extract_features_and_bouts(
        aligned_df, pixel_to_mm_ratio=args.px_to_mm, fps=fps
    )
    print(f"\n[Progress: 95%] 特徴量抽出完了 — CSV書き出し中...")

    # 4. CSVファイルの出力
    video_basename = os.path.splitext(os.path.basename(args.video))[0]

    frame_csv_path   = os.path.join(args.out_dir, f"{video_basename}_frame_labels.csv")
    bouts_csv_path   = os.path.join(args.out_dir, f"{video_basename}_bouts.csv")
    summary_csv_path = os.path.join(args.out_dir, f"{video_basename}_summary.csv")

    print("\n--- [Step 3] 結果のCSV書き出し中 ---")

    # 学術的な正確性を保持するため、一律 0 で埋める fillna(0) は廃止。
    # 代わりに未割り当ての mouse_id (NaNなど) のみ、明示的に 0 (未解決ID) として割り振る
    if "mouse_id" in processed_df.columns:
        processed_df["mouse_id"] = processed_df["mouse_id"].fillna(0).astype(int)

    # 保存時に欠損座標や欠損特徴量は "" (空) になるように na_rep を指定
    processed_df.to_csv(frame_csv_path, index=False, na_rep="")
    bouts_df.to_csv(bouts_csv_path, index=False, na_rep="")
    summary_df.to_csv(summary_csv_path, index=False, na_rep="")

    print(f"[pipeline] フレームラベルCSV保存完了: {frame_csv_path}")
    print(f"[pipeline] Bout情報CSV保存完了: {bouts_csv_path}")
    print(f"[pipeline] サマリー統計CSV保存完了: {summary_csv_path}")
    print(f"[pipeline] 総検出 Bout 数: {len(bouts_df)} 件")

    # 各出力CSVの列の意味（単位・算出式）を記したデータ辞書CSVを同梱出力する。
    columns_csv_path = os.path.join(args.out_dir, f"{video_basename}_columns.csv")
    try:
        write_column_definitions(columns_csv_path)
        print(f"[pipeline] 列定義（データ辞書）CSV保存完了: {columns_csv_path}")
    except Exception as e:
        print(f"[pipeline] [Warning] 列定義CSVの書き出しに失敗しました: {e}")

    # ── [A-3] カラーID品質レポート JSON 出力 ─────────────────────────────────
    # フロントエンドが個体ごとのID信頼度・低信頼フレーム数・Swap（混同）イベントを
    # 可視化できるよう、専用 JSON を結果CSVと同じディレクトリに書き出す。
    id_quality_path = os.path.join(args.out_dir, f"{video_basename}_id_quality.json")
    try:
        with open(id_quality_path, "w", encoding="utf-8") as fp:
            json.dump(id_quality, fp, ensure_ascii=False, indent=2)
        n_swaps = id_quality.get("n_swap_events", 0)
        print(f"[pipeline] カラーID品質レポート保存完了: {id_quality_path} (Swap検知: {n_swaps} 件)")
    except Exception as e:
        print(f"[pipeline] [Warning] カラーID品質レポート書き出しに失敗しました: {e}")

    # ── [A-4] プロベナンス manifest 出力 ─────────────────────────────────────
    # 結果CSVと同じディレクトリに、再現に必要な全パラメータを JSON で残す。
    manifest_path = os.path.join(args.out_dir, f"{video_basename}_manifest.json")
    write_provenance_manifest(
        manifest_path, args, fps,
        output_files={
            "frame_labels_csv": os.path.basename(frame_csv_path),
            "bouts_csv":        os.path.basename(bouts_csv_path),
            "summary_csv":      os.path.basename(summary_csv_path),
            "id_quality_json":  os.path.basename(id_quality_path),
            "n_bouts":          int(len(bouts_df)),
        },
        id_quality=id_quality,
    )

    # 5. グラウンドトゥルース評価（--gt 指定時のみ）
    if args.gt:
        print("\n--- [Step 4] グラウンドトゥルース評価実行中 ---")
        try:
            import evaluate as ev
            ev.run_evaluation(
                auto_frame_csv=frame_csv_path,
                gt_csv=args.gt,
                gt_format=args.gt_format,
                fps=fps,
                iou_threshold=args.iou,
                output_dir=args.out_dir,
                auto_bouts_csv=bouts_csv_path,
            )
        except Exception as e:
            print(f"[pipeline] [Warning] 評価処理中にエラーが発生しました: {e}")

    print("\n========================================================")
    print(" パイプライン処理が正常に完了しました。")
    print("========================================================\n")

if __name__ == "__main__":
    main()
