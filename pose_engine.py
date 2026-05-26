"""
Pose Estimation Engine Module
=============================
本モジュールは、マウス姿勢推定プロセスの実行を抽象化するクラス群を提供します。
SLEAP、YOLOv8-pose、および環境に依存しないフォールバック用モックエンジンをサポートします。
出力データは一律で内部共通ロングフォーマット（frame, track, keypoint, x, y, conf）に統一されます。
"""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import cv2
from typing import Optional
from config import KEYPOINTS, DEFAULT_FPS

# ── GPU 自動認識・対応 ────────────────────────────────────────────────────────

def detect_gpu_device() -> str:
    """
    実行環境のGPU可用性を自動判定します。
    PyTorchが利用可能で、CUDA対応GPUが認識できる場合は "cuda" を返し、
    それ以外の場合は "cpu" を返します（CPUフォールバック）。
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[pose_engine] [GPU-Detect] CUDA対応GPUが検出されました: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            print("[pose_engine] [GPU-Detect] GPUが検出されませんでした。CPU推論を使用します。")
            return "cpu"
    except ImportError:
        print("[pose_engine] [GPU-Detect] PyTorchがインストールされていません。CPU推論を使用します。")
        return "cpu"


# ── SLEAP analysis HDF5 の正準パーサ ─────────────────────────────────────────
# SLEAP の analysis HDF5 は、ディスク上では tracks=(n_tracks, 2, n_nodes, n_frames)、
# point_scores=(n_tracks, n_nodes, n_frames) というレイアウトで格納される。
# SLEAP 公式の解析例でも f["tracks"][:].T のように転置して使うのが正しい。
# （旧実装は転置せず (n_frames, n_nodes, 2, n_tracks) と誤って仮定していたため、
#   実 SLEAP ファイルでは次元が入れ替わり IndexError → モックへ静かにフォールバックしていた）

def _normalize_sleap_track_array(tracks: np.ndarray, n_nodes_expected: int) -> np.ndarray:
    """tracks を内部標準の (n_frames, n_nodes, 2, n_tracks) に正規化する。"""
    if tracks.ndim != 4:
        raise ValueError(f"想定外のトラックデータ形状です: {tracks.shape}")
    # 既に (frames, nodes, 2, tracks) の並びならそのまま使う
    if tracks.shape[2] == 2 and tracks.shape[1] == n_nodes_expected:
        return tracks
    # SLEAP 標準の (tracks, 2, nodes, frames) → 完全転置で (frames, nodes, 2, tracks)
    if tracks.shape[1] == 2 and tracks.shape[2] == n_nodes_expected:
        return tracks.T
    # どちらにも当てはまらない場合でも、coords 軸が末尾 2 になるよう SLEAP 公式に従い転置を試みる
    transposed = tracks.T
    if transposed.shape[2] == 2:
        return transposed
    raise ValueError(
        f"SLEAP tracks 配列の軸を解釈できませんでした: shape={tracks.shape}, n_nodes={n_nodes_expected}"
    )


def _normalize_sleap_scores(scores, n_frames: int, n_nodes_expected: int, n_tracks: int):
    """point_scores を (n_frames, n_nodes, n_tracks) に正規化する。解釈不能なら None。"""
    if scores is None or getattr(scores, "ndim", 0) != 3:
        return None
    if scores.shape == (n_frames, n_nodes_expected, n_tracks):
        return scores
    transposed = scores.T  # (tracks, nodes, frames) → (frames, nodes, tracks)
    if transposed.shape == (n_frames, n_nodes_expected, n_tracks):
        return transposed
    return None  # 形状が一致しない場合は誤対応を避けるため使用しない


def parse_sleap_analysis_h5(h5_path: str) -> pd.DataFrame:
    """
    SLEAP analysis HDF5 を読み込み、共通ロングフォーマット
    (frame, track, keypoint, x, y, conf, is_mock) の DataFrame を返す。
    conf には可能なら SLEAP の point_scores（キーポイント信頼度）を使用する。
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "node_names" not in f or "tracks" not in f:
            raise ValueError("SLEAP H5ファイルのスキーマが期待と異なります（node_names または tracks がありません）。")

        node_names = [n.decode("utf-8") for n in f["node_names"][:]]
        tracks = _normalize_sleap_track_array(f["tracks"][:], len(node_names))
        n_frames, n_nodes, _n_coords, n_tracks = tracks.shape

        scores_raw = f["point_scores"][:] if "point_scores" in f else None
        scores = _normalize_sleap_scores(scores_raw, n_frames, len(node_names), n_tracks)
        if scores_raw is not None and scores is None:
            print("[pose_engine] [Warning] point_scores の形状を解釈できなかったため、"
                  "信頼度は座標の有無から推定します。")
        elif scores is None:
            print("[pose_engine] [Warning] point_scores データセットが存在しないため、"
                  "信頼度は座標の有無から推定します（全有効点を conf=1.0 とみなします）。")

        print(f"[pose_engine] [Info] SLEAP ロード完了: フレーム数={n_frames}, "
              f"ノード数={n_nodes}, トラック数={n_tracks}, 信頼度={'point_scores' if scores is not None else '座標有無'}")

        records = []
        for t_idx in range(n_tracks):
            track_name = f"track_{t_idx}"
            for f_idx in range(n_frames):
                frame_coords = tracks[f_idx, :, :, t_idx]  # (n_nodes, 2)
                if np.isnan(frame_coords).all():
                    continue
                for n_idx, node in enumerate(node_names):
                    if node not in KEYPOINTS:
                        continue
                    x = frame_coords[n_idx, 0]
                    y = frame_coords[n_idx, 1]
                    valid = not (np.isnan(x) or np.isnan(y))
                    if scores is not None:
                        s = scores[f_idx, n_idx, t_idx]
                        conf = float(s) if (valid and not np.isnan(s)) else 0.0
                    else:
                        conf = 1.0 if valid else 0.0
                    records.append({
                        "frame": int(f_idx),
                        "track": track_name,
                        "keypoint": node,
                        "x": float(x) if valid else np.nan,
                        "y": float(y) if valid else np.nan,
                        "conf": conf,
                        "is_mock": 0,
                    })
        return pd.DataFrame(records)


# ── 基底クラス（インターフェース） ───────────────────────────────────────────

class BasePoseEngine:
    """
    すべての姿勢推定エンジンの基底クラス。
    """
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path
        self.device = device if device is not None else detect_gpu_device()
        self.is_mock_engine = False
        
        # models/ ディレクトリの自動作成
        os.makedirs("models", exist_ok=True)
        
        # モデルパスが未指定の場合の自動検索
        if not self.model_path:
            self.model_path = self._auto_detect_model()

    def _auto_detect_model(self) -> str:
        """サブクラスでデフォルトモデルを自動検出するためのメソッド。"""
        return None

    def run(self, video_path: str) -> pd.DataFrame:
        """
        指定された動画に対して姿勢推定を実行し、ロングフォーマットのDataFrameを返します。
        
        Parameters
        ----------
        video_path : str
            動画ファイルへのパス。
            
        Returns
        -------
        pd.DataFrame
            カラム: ['frame', 'track', 'keypoint', 'x', 'y', 'conf', 'is_mock']
        """
        raise NotImplementedError("サブクラスで run() メソッドを実装する必要があります。")


# ── SLEAP 姿勢推定エンジン ───────────────────────────────────────────────────

class SleapEngine(BasePoseEngine):
    """
    SLEAP (sleap-track) コマンドラインツールを内部で自動呼び出しして姿勢推定を実行するエンジン。
    """
    def _auto_detect_model(self) -> Optional[str]:
        """models/ ディレクトリから .slp モデルを検索・自動選択します。"""
        if os.path.exists("models"):
            for item in os.listdir("models"):
                path = os.path.join("models", item)
                if item.endswith(".slp") or (os.path.isdir(path) and "training_config.json" in os.listdir(path)):
                    print(f"[pose_engine] [Auto-Detect] SLEAPモデルを自動検出しました: {path}")
                    return path
        return None

    def run(self, video_path: str) -> pd.DataFrame:
        if not self.model_path:
            # モデルパスが指定されていない場合、フォールバックまたはエラー
            print("[pose_engine] [Warning] SLEAPモデルのパスが指定されていません。")
            # 環境内のテスト用としてモックエンジンに委譲するか、エラーを投げます。
            # ここでは堅牢性のため、コマンドが呼べない場合はモックへフォールバックします。

        # コマンドの存在チェック
        sleap_available = False
        try:
            # Windows/Mac/Linux共通で実行ファイルパスをチェック
            result = subprocess.run(["sleap-track", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            sleap_available = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("[pose_engine] [Warning] SLEAP ('sleap-track') がシステム環境に見つからないか、実行できません。")

        if not sleap_available:
            print("[pose_engine] [Fallback] SLEAPが利用できないため、自動的にMockPoseEngineにフォールバックします。")
            fallback_engine = MockPoseEngine(self.model_path, self.device)
            return fallback_engine.run(video_path)

        # 本物の推論を実行するため、動画ファイルが存在することを確認
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        # 一時出力先としてのH5ファイルを生成
        temp_dir = tempfile.gettempdir()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_h5 = os.path.join(temp_dir, f"sleap_temp_{video_name}.h5")

        # コマンドラインの組み立て
        cmd = [
            "sleap-track",
            video_path,
            "--output", temp_h5
        ]
        
        if self.model_path:
            cmd.extend(["--model", self.model_path])
            
        # GPUデバイスの指定を反映
        if self.device == "cpu":
            # SLEAPが環境変数やオプションでCPU動作を強制できる場合の対応
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
        print(f"[pose_engine] [Info] SLEAPによる姿勢推定を開始します: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"[pose_engine] [Info] SLEAPの実行が完了しました。出力をパースします: {temp_h5}")
            
            # 生成された H5 ファイルをパースしてロングフォーマットへ変換
            df = self._load_and_parse_h5(temp_h5)
            
            # 一時ファイルのクリーンアップ
            if os.path.exists(temp_h5):
                os.remove(temp_h5)
                
            return df
        except subprocess.CalledProcessError as e:
            print(f"[pose_engine] [Error] SLEAPの実行に失敗しました: {e}")
            print("[pose_engine] [Fallback] エラーが発生したため、MockPoseEngineにフォールバックします。")
            fallback_engine = MockPoseEngine(self.model_path, self.device)
            return fallback_engine.run(video_path)

    def _load_and_parse_h5(self, h5_path: str) -> pd.DataFrame:
        """
        SLEAP H5ファイルをロードし、共通ロングフォーマットに展開します。
        次元正規化と point_scores（信頼度）の読み込みは parse_sleap_analysis_h5 に委譲します。
        """
        return parse_sleap_analysis_h5(h5_path)


# ── YOLOv8-pose 姿勢推定エンジン ─────────────────────────────────────────────

class Yolov8PoseEngine(BasePoseEngine):
    """
    YOLOv8-pose を内部で自動呼び出しして姿勢推定を実行するエンジン。
    """
    def _auto_detect_model(self) -> Optional[str]:
        """models/ ディレクトリから .pt モデルを検索・自動選択します。"""
        if os.path.exists("models"):
            for item in os.listdir("models"):
                path = os.path.join("models", item)
                if item.endswith(".pt") and "pose" in item.lower():
                    print(f"[pose_engine] [Auto-Detect] YOLOv8ポーズモデルを自動検出しました: {path}")
                    return path
        return None

    def run(self, video_path: str) -> pd.DataFrame:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[pose_engine] [Warning] ultralytics (YOLOv8) ライブラリがインストールされていません。")
            print("[pose_engine] [Fallback] YOLOv8が利用できないため、自動的にMockPoseEngineにフォールバックします。")
            fallback_engine = MockPoseEngine(self.model_path, self.device)
            return fallback_engine.run(video_path)

        # モデルのロード（指定がない場合は標準モデル）。
        # モデルファイルの欠損・破損や推論失敗でパイプライン全体が落ちないよう、
        # ロード・推論を例外保護し、失敗時は ImportError と同様に Mock へフォールバックする。
        model_name = self.model_path if self.model_path else "yolov8n-pose.pt"
        print(f"[pose_engine] [Info] YOLOv8-pose モデルをロード中: {model_name}")
        try:
            model = YOLO(model_name)
            print(f"[pose_engine] [Info] YOLOv8-poseによる姿勢推定を開始します: {video_path} (デバイス: {self.device})")
            results = model.track(source=video_path, device=self.device, persist=True, stream=True)
        except Exception as e:
            print(f"[pose_engine] [Error] YOLOv8モデルのロードまたは推論に失敗しました: {e}")
            print("[pose_engine] [Fallback] 自動的にMockPoseEngineにフォールバックします。")
            fallback_engine = MockPoseEngine(self.model_path, self.device)
            return fallback_engine.run(video_path)

        records = []
        try:
            for f_idx, r in enumerate(results):
                # キーポイント (N, 17, 3) などの形状。YOLOv8標準は17点
                keypoints_data = r.keypoints
                if keypoints_data is None:
                    continue

                # 各検出個体の座標情報を展開
                xy = keypoints_data.xy.cpu().numpy()     # (num_instances, num_kps, 2)
                conf = keypoints_data.conf.cpu().numpy() if keypoints_data.conf is not None else None # (num_instances, num_kps)

                # トラックIDの取得
                if r.boxes is not None and r.boxes.id is not None:
                    track_ids = r.boxes.id.int().cpu().tolist()
                else:
                    track_ids = list(range(len(xy)))

                for idx, track_id in enumerate(track_ids):
                    track_name = f"track_{track_id}"
                    inst_xy = xy[idx]
                    inst_conf = conf[idx] if conf is not None else [1.0] * len(inst_xy)

                    # キーポイント数が本システムの定義と一致するかチェック
                    if len(inst_xy) == len(KEYPOINTS):
                        # カスタムモデルとしてそのままマッピング
                        for kp_idx, kp_name in enumerate(KEYPOINTS):
                            px, py = inst_xy[kp_idx]
                            p_conf = inst_conf[kp_idx]
                            records.append({
                                "frame": int(f_idx),
                                "track": track_name,
                                "keypoint": kp_name,
                                "x": float(px),
                                "y": float(py),
                                "conf": float(p_conf),
                                "is_mock": 0
                            })
                    else:
                        # ⚠️ 学術利用警告：標準 17 点 (COCO 人間用) → マウス 11 点の近似マッピング。
                        # nose=0 を 'nose' と 'head_center' の両方に流用するなど解剖学的に厳密でないため、
                        # 距離特徴量 (`head_to_nose_dist` 等) は退化値となる。学術論文用途では
                        # マウス専用学習モデル (custom yolov8 / SLEAP) を必ず使用すること。
                        if f_idx == 0:
                            print("[pose_engine] [Warning] ⚠️ 学術利用注意：標準 17 点 (人間 COCO) モデルを近似マッピングで使用しています。"
                                  "解剖学的に正確ではないため、結果は学術論文に使用しないでください。"
                                  "マウス専用に学習したモデル (custom YOLOv8 / SLEAP) を使用してください。")
                        mapping = {
                            "nose": 0,
                            "left_ear": 3,
                            "right_ear": 4,
                            "head_center": 0,      # 近似 (= nose) — ⚠️ head_to_nose_dist が常に 0 になる
                            "neck": 5,             # 肩を首の代わりとする
                            "left_fore_paw": 9,    # 手首
                            "right_fore_paw": 10,  # 手首
                            "spine_mid": 11,       # 臀部近傍
                            "left_hind_paw": 15,   # 足首
                            "right_hind_paw": 16,  # 足首
                            "tail_base": 12,       # 臀部
                        }
                        for kp_name, yolo_idx in mapping.items():
                            if yolo_idx < len(inst_xy):
                                px, py = inst_xy[yolo_idx]
                                p_conf = inst_conf[yolo_idx]
                                records.append({
                                    "frame": int(f_idx),
                                    "track": track_name,
                                    "keypoint": kp_name,
                                    "x": float(px),
                                    "y": float(py),
                                    "conf": float(p_conf),
                                    "is_mock": 0
                                })
        except Exception as e:
            # ストリーム推論の途中で失敗した場合: 部分結果があればそれを返し、無ければ Mock へ。
            print(f"[pose_engine] [Error] YOLOv8推論の処理中にエラーが発生しました: {e}")
            if not records:
                print("[pose_engine] [Fallback] 有効な検出が無いため、MockPoseEngineにフォールバックします。")
                fallback_engine = MockPoseEngine(self.model_path, self.device)
                return fallback_engine.run(video_path)
            print("[pose_engine] [Warning] 取得済みの部分的な検出結果を返します。")

        if not records:
            # 検出できなかった場合はモックにフォールバック
            print("[pose_engine] [Warning] YOLOv8での姿勢検出座標が空でした。MockPoseEngineに切り替えます。")
            fallback_engine = MockPoseEngine(self.model_path, self.device)
            return fallback_engine.run(video_path)

        return pd.DataFrame(records)


# ── Mock 姿勢推定エンジン (インテリジェントフォールバック) ────────────────────

class MockPoseEngine(BasePoseEngine):
    """
    外部の姿勢推定モデルを使用せず、動画ファイルの情報（フレーム数、サイズ）を取得し、
    アライメント可能な自然な4匹のマウス軌跡座標を生成するモックエンジン。
    """
    def __init__(self, model_path: str = None, device: str = None):
        super().__init__(model_path, device)
        self.is_mock_engine = True

    def _auto_detect_model(self) -> Optional[str]:
        return None

    def run(self, video_path: str) -> pd.DataFrame:
        print("[pose_engine] [Mock] [警告] 学習済み姿勢推定モデルが検出されないため、ダミーデータの生成を開始します。学術論文等には使用しないでください。")
        print(f"[pose_engine] [Mock] MockPoseEngineを起動します: {video_path}")
        
        # 動画からフレーム情報などを取得
        n_frames = 150
        fps = DEFAULT_FPS
        width, height = 640, 480
        
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or np.isnan(fps):
                    fps = DEFAULT_FPS
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                print(f"[pose_engine] [Mock] 動画情報の読み込み成功: フレーム数={n_frames}, FPS={fps:.1f}, 解像度={width}x{height}")
            else:
                print(f"[pose_engine] [Mock] [Warning] 動画ファイルを開けませんでした。標準解像度でダミーデータを生成します。")
        else:
            print(f"[pose_engine] [Mock] [Warning] 動画ファイルが見つかりません。標準解像度でダミーデータを生成します。")

        # 4匹のマウスの初期中心座標（動画のサイズに合わせて配置）
        centers = {
            "track_0": (width * 0.25, height * 0.35),
            "track_1": (width * 0.65, height * 0.35),
            "track_2": (width * 0.25, height * 0.65),
            "track_3": (width * 0.70, height * 0.65),
        }

        # 各キーポイントの相対オフセット
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

        records = []
        for f in range(n_frames):
            for track_id, (cx, cy) in centers.items():
                dx = 0
                dy = 0
                
                # 各キーポイントの座標を展開してロングフォーマットに変換
                for kp_name, (ox, oy) in kp_offsets.items():
                    px = cx + dx + ox
                    py = cy + dy + oy
                    
                    # 特定のフレーム範囲で、ひっかき行動（左後ろ肢の高速振動）をシミュレート
                    # track_0は中盤、track_1は終盤などでひっかきを発生させる
                    if kp_name == "left_hind_paw" and track_id == "track_0" and int(n_frames * 0.4) <= f <= int(n_frames * 0.6):
                        # 耳に近づける
                        ear_ox, ear_oy = kp_offsets["left_ear"]
                        ear_x = cx + dx + ear_ox
                        ear_y = cy + dy + ear_oy
                        px = ear_x + 5 + 35 * (f % 2) # X軸方向に大きく高速振動！
                        py = ear_y + 10 + 10 * ((f // 2) % 2)

                    records.append({
                        "frame": int(f),
                        "track": track_id,
                        "keypoint": kp_name,
                        "x": float(px),
                        "y": float(py),
                        "conf": 0.98,
                        "is_mock": 1
                    })
                    
        print(f"[pose_engine] [Mock] {len(records)} 行の姿勢推定データを生成しました。")
        return pd.DataFrame(records)
