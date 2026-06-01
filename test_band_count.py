"""
色 + 帯本数 (color + band count) 個体識別のユニットテスト
========================================================
DLC や実動画に依存せず、合成した「尾の帯画像」に対して
classify_tail_by_bands() が (色, 本数) → mouse_id を正しく返すかを検証する。

対応表 (config.MOUSE_ID_BY_COLOR_COUNT):
    赤1本 → 1, 赤2本 → 2, 青1本 → 3, 青2本 → 4

実行: python -m unittest test_band_count.py
"""

import unittest
import numpy as np
import cv2

from color_id import (
    classify_tail_by_bands,
    _count_bands,
    _resample_polyline,
)
from config import MOUSE_COLORS


def _hsv_to_bgr(h, s, v):
    """HSV 単色を BGR タプルへ変換するヘルパー。"""
    px = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0][0])


# MOUSE_COLORS の範囲中央付近の代表色 (彩度・明度は十分高く設定)
_RED_BGR = _hsv_to_bgr(
    (MOUSE_COLORS[1]["hsv_lower"][0] + MOUSE_COLORS[1]["hsv_upper"][0]) // 2, 220, 220)
_BLUE_BGR = _hsv_to_bgr(
    (MOUSE_COLORS[2]["hsv_lower"][0] + MOUSE_COLORS[2]["hsv_upper"][0]) // 2, 220, 220)


def _make_tail_frame(bands):
    """
    水平な尾 (y=100, x=40..160) に沿って帯を描いた黒背景フレームを生成する。

    bands : [(x_start, x_end, color_bgr), ...]
            各帯は y=88..112 (高さ24px) の塗り矩形として描画する
            （半径 7px のサンプリング円が帯内に十分収まる太さ）。

    戻り値: (frame, tail_points)  tail_points は base→end の折れ線端点。
    """
    frame = np.zeros((200, 220, 3), dtype=np.uint8)
    for x0, x1, color in bands:
        cv2.rectangle(frame, (x0, 88), (x1, 112), color, thickness=-1)
    tail_points = [(40.0, 100.0), (160.0, 100.0)]  # 端点のみ（再標本化で内挿される）
    return frame, tail_points


class TestCountBands(unittest.TestCase):
    """_count_bands の純ロジック検証（画像非依存）。"""

    def test_single_run(self):
        # 連続した1の塊が1つ → 1帯
        present = [0, 0, 1, 1, 1, 1, 0, 0]
        self.assertEqual(_count_bands(present, min_run=2, max_gap=3), 1)

    def test_two_runs_large_gap(self):
        # 大きなギャップ(>=max_gap)で分離 → 2帯
        present = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
        self.assertEqual(_count_bands(present, min_run=2, max_gap=3), 2)

    def test_small_gap_merged(self):
        # 小さなギャップ(<max_gap)は結合 → 1帯
        present = [1, 1, 1, 0, 1, 1, 1]
        self.assertEqual(_count_bands(present, min_run=2, max_gap=3), 1)

    def test_noise_run_ignored(self):
        # min_run 未満の微小runはノイズとして無視 → 0帯
        present = [0, 1, 0, 0, 1, 0]
        self.assertEqual(_count_bands(present, min_run=2, max_gap=3), 0)


class TestResamplePolyline(unittest.TestCase):
    def test_even_spacing(self):
        pts = [(0.0, 0.0), (40.0, 0.0)]
        out = _resample_polyline(pts, step=4.0)
        # 0..40 を 4px 間隔 → 11 点 (0,4,...,40)
        self.assertGreaterEqual(len(out), 10)
        # x が単調増加し、間隔がおよそ step
        xs = [p[0] for p in out]
        self.assertTrue(all(b - a > 0 for a, b in zip(xs[:-1], xs[1:])))

    def test_degenerate(self):
        self.assertEqual(_resample_polyline([(1.0, 1.0)], 4.0), [(1.0, 1.0)])
        self.assertEqual(_resample_polyline([], 4.0), [])


class TestClassifyTailByBands(unittest.TestCase):
    """合成画像から (色,本数) → mouse_id を検証。"""

    def test_red_one_band(self):
        frame, tp = _make_tail_frame([(80, 115, _RED_BGR)])  # 赤1本
        self.assertEqual(classify_tail_by_bands(frame, tp), 1)

    def test_red_two_bands(self):
        frame, tp = _make_tail_frame([(60, 90, _RED_BGR), (120, 150, _RED_BGR)])  # 赤2本
        self.assertEqual(classify_tail_by_bands(frame, tp), 2)

    def test_blue_one_band(self):
        frame, tp = _make_tail_frame([(80, 115, _BLUE_BGR)])  # 青1本
        self.assertEqual(classify_tail_by_bands(frame, tp), 3)

    def test_blue_two_bands(self):
        frame, tp = _make_tail_frame([(60, 90, _BLUE_BGR), (120, 150, _BLUE_BGR)])  # 青2本
        self.assertEqual(classify_tail_by_bands(frame, tp), 4)

    def test_no_marker_returns_none(self):
        # 帯なし（黒背景のみ）→ 判定不能 None
        frame, tp = _make_tail_frame([])
        self.assertIsNone(classify_tail_by_bands(frame, tp))

    def test_too_few_points_returns_none(self):
        frame, _ = _make_tail_frame([(80, 115, _RED_BGR)])
        self.assertIsNone(classify_tail_by_bands(frame, [(40.0, 100.0)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
