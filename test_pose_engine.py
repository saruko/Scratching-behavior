"""
Unit Tests for Pose Estimation Engine Module
===========================================
pose_engine.py および pipeline.py の姿勢推定データ変換処理に関するテスト。
"""

import os
import unittest
import numpy as np
import pandas as pd

import pose_engine
from pipeline import long_to_wide_format
from config import KEYPOINTS

class TestPoseEngine(unittest.TestCase):
    def setUp(self):
        self.dummy_video = "dummy_non_existent_video.mp4"
        self.real_video = "test_swap.mp4" # リポジトリ内に実在する動画ファイル

    def test_gpu_detection(self):
        """GPU自動判定機能がエラーなく動作し、期待されるデバイス名を返すか検証します。"""
        device = pose_engine.detect_gpu_device()
        self.assertIn(device, ["cuda", "cpu"])
        print(f"[Test] 自動検知されたデバイス: {device}")

    def test_mock_pose_engine_dummy_video(self):
        """MockPoseEngine が存在しない動画に対しても堅牢に動作し、デフォルト仕様のダミーデータを生成できるか検証します。"""
        engine = pose_engine.MockPoseEngine()
        df = engine.run(self.dummy_video)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        # ロングフォーマットのカラム検証
        expected_cols = ["frame", "track", "keypoint", "x", "y", "conf"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # デフォルトフレーム数は 150 フレーム
        max_frame = df["frame"].max()
        self.assertEqual(max_frame, 149) # 0-indexed で 150フレーム
        
        # トラック数は 4つ (track_0 to track_3)
        tracks = df["track"].unique()
        self.assertEqual(len(tracks), 4)
        
        # キーポイント数が 11個
        kps = df["keypoint"].unique()
        self.assertEqual(len(kps), len(KEYPOINTS))

    def test_mock_pose_engine_real_video(self):
        """MockPoseEngine が存在する動画から仕様を正しく取得してダミーデータを生成できるか検証します。"""
        if not os.path.exists(self.real_video):
            self.skipTest(f"実動画ファイル {self.real_video} が見つからないためテストをスキップします。")
            
        engine = pose_engine.MockPoseEngine()
        df = engine.run(self.real_video)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        
        # 実動画 test_swap.mp4 に合わせたフレーム数が生成されているはず
        # （cv2.VideoCaptureが正しくフレーム数を検出できたか）
        self.assertTrue(df["frame"].max() > 0)

    def test_long_to_wide_format_conversion(self):
        """ロングフォーマットからワイドフォーマットへの変換が、アライメントと値の正確性を維持して完了するか検証します。"""
        # テスト用ロングフォーマットデータの作成
        records = [
            {"frame": 0, "track": "track_0", "keypoint": "nose", "x": 10.0, "y": 20.0, "conf": 0.9},
            {"frame": 0, "track": "track_0", "keypoint": "tail_base", "x": 100.0, "y": 200.0, "conf": 0.85},
            {"frame": 0, "track": "track_1", "keypoint": "nose", "x": 30.0, "y": 40.0, "conf": 0.95},
            {"frame": 1, "track": "track_0", "keypoint": "nose", "x": 11.0, "y": 21.0, "conf": 0.92},
        ]
        long_df = pd.DataFrame(records)
        
        # 変換実行
        wide_df = long_to_wide_format(long_df)
        
        self.assertIsInstance(wide_df, pd.DataFrame)
        self.assertIn("frame_idx", wide_df.columns)
        self.assertIn("track", wide_df.columns)
        
        # ピボット展開された座標カラムの検証
        self.assertIn("nose_x", wide_df.columns)
        self.assertIn("nose_y", wide_df.columns)
        self.assertIn("nose_conf", wide_df.columns)
        self.assertIn("tail_base_x", wide_df.columns)
        
        # 行数の検証 (フレーム0のtrack_0, track_1、フレーム1のtrack_0 の計 3行)
        self.assertEqual(len(wide_df), 3)
        
        # 具体的な値の整合性検証
        row_f0_t0 = wide_df[(wide_df["frame_idx"] == 0) & (wide_df["track"] == "track_0")].iloc[0]
        self.assertEqual(row_f0_t0["nose_x"], 10.0)
        self.assertEqual(row_f0_t0["nose_y"], 20.0)
        self.assertEqual(row_f0_t0["nose_conf"], 0.9)
        self.assertEqual(row_f0_t0["tail_base_x"], 100.0)

    def test_engine_fallback(self):
        """SLEAP または YOLOv8 が未インストール等の理由で失敗した場合に、MockPoseEngineへフォールバックされるか検証します。"""
        # 意図的に無効なモデルパスやコマンドを叩かせ、フォールバックの動作を確認
        # SleapEngine は sleap-track がない場合自動的に MockPoseEngine にフォールバックする
        sleap_engine = pose_engine.SleapEngine(model_path="non_existent_model.h5")
        
        # 存在しない動画を渡しても、フォールバックされれば正常に MockPoseEngine が起動してデータを返すはず
        df = sleap_engine.run(self.dummy_video)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(df["frame"].max(), 149) # MockPoseEngineのデフォルトフレーム数

        # Yolov8PoseEngine も同様にフォールバックされるはず
        yolo_engine = pose_engine.Yolov8PoseEngine(model_path="non_existent_model.pt")
        df_yolo = yolo_engine.run(self.dummy_video)
        
        self.assertIsInstance(df_yolo, pd.DataFrame)
        self.assertFalse(df_yolo.empty)
        self.assertEqual(df_yolo["frame"].max(), 149)

if __name__ == "__main__":
    unittest.main()
