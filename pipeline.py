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
import numpy as np
import pandas as pd

from config import KEYPOINTS, DEFAULT_FPS, CAGE_MICE_COUNT, MAX_MICE_LIMIT
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
            dx = 0
            dy = 0
            
            row = {
                "frame_idx": f,
                "track": track_id,
                "is_mock": 1
            }
            
            # 各キーポイントにノイズと動きを付与
            for kp, (ox, oy) in kp_offsets.items():
                kp_x = cx + dx + ox
                kp_y = cy + dy + oy
                
                # 特定のフレーム（例えば 60-90フレーム目）で、左後ろ足を素早く振動させてひっかきをシミュレート
                if kp == "left_hind_paw" and track_id == "track_0" and 60 <= f <= 90:
                    # 耳に近づける
                    ear_x = cx + dx + kp_offsets["left_ear"][0]
                    ear_y = cy + dy + kp_offsets["left_ear"][1]
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


def build_preview_json(
    pose_df: pd.DataFrame,
    fps: float,
    video_path: str,
    preview_duration_s: float,
    sample_interval_s: float = 1.0,
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

            mouse_id = None
            if "mouse_id" in row.index and not pd.isna(row.get("mouse_id")):
                mouse_id = int(row["mouse_id"])

            detections.append({
                "track": str(row["track"]),
                "mouse_id": mouse_id,          # カラーID自動推定値（参考）
                "bbox": bbox,
                "keypoints": kps_out,
            })

        frames_dict[str(frame_idx)] = {
            "timestamp_s": round(frame_idx / fps, 3),
            "detections": detections,
        }

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
            engine = pose_engine.DeepLabCutEngine(
                model_path=args.model_path,
                device=device_override,
                superanimal_name=args.dlc_superanimal,
                detector_name=args.dlc_detector,
                pose_model_name=args.dlc_pose_model,
            )
        else:
            engine = pose_engine.MockPoseEngine(model_path=args.model_path, device=device_override)
            
        # 姿勢推定の実行 (ロングフォーマットが返る)
        long_df = engine.run(args.video)
        
        # ワイドフォーマットへの変換
        pose_df = long_to_wide_format(long_df)
        
    print(f"[pipeline] 初期ロードデータ数: {len(pose_df)} 行")

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

    # 3. 特徴量抽出およびスクラッチ・Bout検出の実行
    print("\n--- [Step 2] 特徴量抽出およびスクラッチ判定実行中 ---")
    processed_df, bouts_df, summary_df = feature_extraction.extract_features_and_bouts(
        aligned_df, pixel_to_mm_ratio=args.px_to_mm, fps=fps
    )

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
