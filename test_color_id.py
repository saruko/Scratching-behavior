import os
import sys
import numpy as np
import pandas as pd
import cv2

from color_id import assign_track_ids
from config import MOUSE_COLORS

def create_dummy_video_and_df(video_path: str, n_frames: int = 100):
    # 動画サイズ
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

    # トラック0とトラック1のtail_base位置
    # track_0 は (100, 120), track_1 は (200, 120) に固定
    t0_xy = (100, 120)
    t1_xy = (200, 120)

    # カラー定義
    # HSV値からBGRに変換するヘルパー
    def hsv_to_bgr(h, s, v):
        hsv_img = np.uint8([[[h, s, v]]])
        bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return tuple(map(int, bgr_img[0][0]))

    # MOUSE_COLORS から HSV の中間値を取得
    # mouse_1: 赤, mouse_2: 青
    red_hsv = MOUSE_COLORS[1]["hsv_lower"]
    blue_hsv = MOUSE_COLORS[2]["hsv_lower"]
    
    red_bgr = hsv_to_bgr((red_hsv[0] + MOUSE_COLORS[1]["hsv_upper"][0]) // 2, 200, 200)
    blue_bgr = hsv_to_bgr((blue_hsv[0] + MOUSE_COLORS[2]["hsv_upper"][0]) // 2, 200, 200)

    records = []

    for f in range(n_frames):
        # 黒い背景フレーム
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 50フレーム目で Identity Swap（交差・混同）をシミュレート
        if f < 50:
            # track_0 = 赤, track_1 = 青
            cv2.circle(frame, t0_xy, 10, red_bgr, -1)
            cv2.circle(frame, t1_xy, 10, blue_bgr, -1)
        else:
            # track_0 = 青, track_1 = 赤 (入れ替わり！)
            cv2.circle(frame, t0_xy, 10, blue_bgr, -1)
            cv2.circle(frame, t1_xy, 10, red_bgr, -1)

        out.write(frame)

        # DataFrameのレコード作成
        records.append({
            "frame_idx": f,
            "track": "track_0",
            "tail_base_x": float(t0_xy[0]),
            "tail_base_y": float(t0_xy[1]),
            "tail_base_conf": 1.0
        })
        records.append({
            "frame_idx": f,
            "track": "track_1",
            "tail_base_x": float(t1_xy[0]),
            "tail_base_y": float(t1_xy[1]),
            "tail_base_conf": 1.0
        })

    out.release()
    return pd.DataFrame(records)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "test_swap.mp4")

    print("--- 1. ダミー動画とポーズデータの生成中 ---")
    pose_df = create_dummy_video_and_df(video_path, n_frames=100)
    print(f"ダミー動画生成完了: {video_path}")
    print(f"ポーズデータ行数: {len(pose_df)}")

    print("\n--- 2. assign_track_ids() を実行して自動分割テスト ---")
    processed_df = assign_track_ids(video_path, pose_df)

    print("\n--- 3. テスト結果の検証 ---")
    # 分割後のユニークトラック名を確認
    unique_tracks = processed_df["track"].unique()
    print(f"分割後のユニークトラック: {unique_tracks}")
    
    # 0フレーム目と99フレーム目の track_0 と track_1 の mouse_id を確認
    t0_f0 = processed_df[(processed_df["frame_idx"] == 0) & (processed_df["track"].str.startswith("track_0"))]
    t0_f99 = processed_df[(processed_df["frame_idx"] == 99) & (processed_df["track"].str.startswith("track_0"))]
    t1_f0 = processed_df[(processed_df["frame_idx"] == 0) & (processed_df["track"].str.startswith("track_1"))]
    t1_f99 = processed_df[(processed_df["frame_idx"] == 99) & (processed_df["track"].str.startswith("track_1"))]

    print(f"フレーム0  の track_0 分割後トラック名: {t0_f0['track'].values[0]}, 割り当てID: {t0_f0['mouse_id'].values[0]}")
    print(f"フレーム99 の track_0 分割後トラック名: {t0_f99['track'].values[0]}, 割り当てID: {t0_f99['mouse_id'].values[0]}")
    print(f"フレーム0  の track_1 分割後トラック名: {t1_f0['track'].values[0]}, 割り当てID: {t1_f0['mouse_id'].values[0]}")
    print(f"フレーム99 の track_1 分割後トラック名: {t1_f99['track'].values[0]}, 割り当てID: {t1_f99['mouse_id'].values[0]}")

    # クリーンアップ
    # if os.path.exists(video_path):
    #     os.remove(video_path)

if __name__ == "__main__":
    main()
