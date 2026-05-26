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
import argparse
import numpy as np
import pandas as pd

from config import KEYPOINTS, DEFAULT_FPS, CAGE_MICE_COUNT
import color_id
import feature_extraction
import pose_engine

def create_mock_sleap_data(n_frames: int = 150) -> pd.DataFrame:
    """
    h5py が利用できない環境や、SLEAPデータが指定されていない場合のフォールバックとして、
    4匹のマウスが動くダミーのトラッキング座標 DataFrame を生成します。
    """
    print("[pipeline] [Info] テスト用のダミーポーズデータ(Mock)を生成します。")
    records = []
    
    # 4匹のケージ内初期中心座標 (動画 test_swap.mp4 のサークル座標 100,120 および 200,120 にアライメント)
    centers = {
        "track_0": (100, 120),
        "track_1": (200, 120),
        "track_2": (100, 180),
        "track_3": (220, 180),
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
        with h5py.File(sleap_path, "r") as f:
            if "node_names" not in f or "tracks" not in f:
                raise ValueError("SLEAP H5ファイルのスキーマが期待と異なります（node_names または tracks がありません）。")

            node_names = [n.decode("utf-8") for n in f["node_names"][:]]
            tracks = f["tracks"][:] # 形状: (n_frames, n_nodes, 2, n_tracks)
            
            if len(tracks.shape) != 4:
                raise ValueError(f"想定外のトラックデータ形状です: {tracks.shape}")
                
            n_frames, n_nodes, n_coords, n_tracks = tracks.shape
            print(f"[pipeline] ロード完了: フレーム数={n_frames}, ノード数={n_nodes}, トラック数={n_tracks}")

            records = []
            for t_idx in range(n_tracks):
                track_name = f"track_{t_idx}"
                for f_idx in range(n_frames):
                    frame_track_coords = tracks[f_idx, :, :, t_idx] # (n_nodes, 2)
                    if np.isnan(frame_track_coords).all():
                        continue
                        
                    row = {
                        "frame_idx": f_idx,
                        "track": track_name,
                        "is_mock": 0
                    }
                    
                    # 各ノードの座標を展開
                    for n_idx, node in enumerate(node_names):
                        if node in KEYPOINTS:
                            x = frame_track_coords[n_idx, 0]
                            y = frame_track_coords[n_idx, 1]
                            row[f"{node}_x"] = float(x) if not np.isnan(x) else np.nan
                            row[f"{node}_y"] = float(y) if not np.isnan(y) else np.nan
                            row[f"{node}_conf"] = 1.0 if not (np.isnan(x) or np.isnan(y)) else 0.0
                            
                    records.append(row)
                    
            df = pd.DataFrame(records)
            if df.empty:
                raise ValueError("H5ファイルから有効な座標レコードを抽出できませんでした。")
            return df

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


def main():
    parser = argparse.ArgumentParser(description="Mouse Scratch Detection System — End-to-End Pipeline")
    parser.add_argument("--video", type=str, required=True, help="解析対象の動画ファイルパス (.mp4 / .mov / .avi など OpenCV対応形式)")
    parser.add_argument("--sleap", type=str, default=None, help="SLEAPから出力された H5 ファイルパス (省略した場合は動画から自動的に姿勢推定を実行)")
    parser.add_argument("--engine", type=str, default="yolov8", choices=["sleap", "yolov8", "mock"], help="姿勢推定の自動実行に使用するエンジン (デフォルト: yolov8)")
    parser.add_argument("--model_path", type=str, default=None, help="姿勢推定モデルの重みファイルパス")
    parser.add_argument("--out_dir", type=str, default="outputs", help="CSV結果を出力するディレクトリ")
    parser.add_argument("--fps", type=int, default=None, help="動画のFPS設定 (configのデフォルト値をオーバーライド)")
    parser.add_argument("--px-to-mm", type=float, default=0.25, help="ピクセルからミリメートルへのキャリブレーション比率 (mm/pixel)")
    
    args = parser.parse_args()

    # 環境変数または引数でFPSなどを一時的にオーバーライドする設定
    if args.fps is not None:
        os.environ["DEFAULT_FPS"] = str(args.fps)

    # 出力先ディレクトリの作成
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("\n========================================================")
    print(" Mouse Scratch Detection Pipeline")
    print("========================================================\n")
    
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
        if args.engine == "sleap":
            engine = pose_engine.SleapEngine(model_path=args.model_path)
        elif args.engine == "yolov8":
            engine = pose_engine.Yolov8PoseEngine(model_path=args.model_path)
        else:
            engine = pose_engine.MockPoseEngine(model_path=args.model_path)
            
        # 姿勢推定の実行 (ロングフォーマットが返る)
        long_df = engine.run(args.video)
        
        # ワイドフォーマットへの変換
        pose_df = long_to_wide_format(long_df)
        
    print(f"[pipeline] 初期ロードデータ数: {len(pose_df)} 行")

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
            aligned_df = color_id.assign_track_ids(args.video, pose_df)
    except Exception as e:
        print(f"[pipeline] [Error] カラーID割り当て中に予期しないエラーが発生しました: {e}")
        print("[pipeline] トラックIDから暫定IDを割り当てて処理を続行します。")
        pose_df["mouse_id"] = pose_df["track"].map(
            lambda t: int(t.split("_")[-1]) + 1 if t.split("_")[-1].isdigit() else 1
        )
        aligned_df = pose_df

    # 3. 特徴量抽出およびスクラッチ・Bout検出の実行
    print("\n--- [Step 2] 特徴量抽出およびスクラッチ判定実行中 ---")
    processed_df, bouts_df = feature_extraction.extract_features_and_bouts(
        aligned_df, pixel_to_mm_ratio=args.px_to_mm
    )

    # 4. CSVファイルの出力
    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    
    frame_csv_path = os.path.join(args.out_dir, f"{video_basename}_frame_labels.csv")
    bouts_csv_path = os.path.join(args.out_dir, f"{video_basename}_bouts.csv")
    
    print("\n--- [Step 3] 結果のCSV書き出し中 ---")
    
    # 学術的な正確性を保持するため、一律 0 で埋める fillna(0) は廃止。
    # 代わりに未割り当ての mouse_id (NaNなど) のみ、明示的に 0 (未解決ID) として割り振る
    if "mouse_id" in processed_df.columns:
        processed_df["mouse_id"] = processed_df["mouse_id"].fillna(0).astype(int)
        
    # 保存時に欠損座標や欠損特徴量は "" (空) になるように na_rep を指定
    processed_df.to_csv(frame_csv_path, index=False, na_rep="")
    bouts_df.to_csv(bouts_csv_path, index=False, na_rep="")
    
    print(f"[pipeline] フレームラベルCSV保存完了: {frame_csv_path}")
    print(f"[pipeline] Bout情報CSV保存完了: {bouts_csv_path}")
    print(f"[pipeline] 総検出 Bout 数: {len(bouts_df)} 件")
    
    print("\n========================================================")
    print(" パイプライン処理が正常に完了しました。")
    print("========================================================\n")

if __name__ == "__main__":
    main()
