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

    互換性チェック: 小さなテンソルを実際に GPU で実行し、カーネルが利用可能か確認します。
    RTX 5080 (sm_120 / Blackwell) + PyTorch 2.5以前 のような非互換組み合わせを
    クリーンに検出し、黙って CPU にフォールバックします。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("[pose_engine] [GPU-Detect] GPUが検出されませんでした。CPU推論を使用します。")
            return "cpu"

        device_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        cap_str = f"sm_{cap[0]}{cap[1]}"

        # 実際にカーネルを起動して互換性を検証する（torch.cuda.is_available() は
        # ドライバ検出のみで、カーネル互換性まではチェックしない）
        try:
            _t = torch.tensor([1.0], device="cuda")
            _ = _t * 2.0  # 演算も実行してカーネルロードを確認
            del _t
            print(f"[pose_engine] [GPU-Detect] GPU使用可能: {device_name} ({cap_str})")
            return "cuda"
        except Exception as e:
            # CUDA error: no kernel image は PyTorch と GPU のアーキテクチャ非互換を示す
            print(f"[pose_engine] [GPU-Detect] GPU ({device_name}, {cap_str}) は現在の PyTorch ビルドでは"
                  f"利用できません: {type(e).__name__}")
            print(f"[pose_engine] [GPU-Detect] → CPU にフォールバックします。")
            print(f"[pose_engine] [GPU-Detect] 修正: pip install torch --index-url "
                  f"https://download.pytorch.org/whl/cu128 で PyTorch を再インストールしてください。")
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


# ── DeepLabCut Model Zoo 姿勢推定エンジン ───────────────────────────────────

class DeepLabCutEngine(BasePoseEngine):
    """
    DeepLabCut (DLC) Model Zoo を使用した姿勢推定エンジン。

    対応モデル:
      - "superanimal_topviewmouse" : 真上視点用 (DLC 公式・27点骨格・マウス専用)
      - "superanimal_quadruped"    : 側面/斜め視点用 (paws を含む 39点骨格)
      - 任意のカスタム学習済み DLC config.yaml

    DLC は zero-shot 推論で SuperAnimal モデルを使えるため、追加学習なしで
    マウス専用骨格を取得可能。学術論文用途には推奨選択肢の一つ。

    パラメータ
    ----------
    model_path : str | None
        - None: SuperAnimal モデルを Model Zoo からダウンロードして使用
        - 文字列 ("superanimal_topviewmouse" / "superanimal_quadruped"): 指定 Model Zoo モデル
        - 既存 config.yaml への絶対パス: カスタム学習モデルを使用
    superanimal_name : str
        Model Zoo の名前。デフォルトは "superanimal_topviewmouse"。
        model_path が config.yaml の場合は無視される。
    detector_name : str
        SuperAnimal の物体検出器。デフォルト "fasterrcnn_resnet50_fpn_v2"。
    pose_model_name : str
        SuperAnimal のポーズモデル。デフォルト "hrnet_w32"。
    max_individuals : int
        フレーム内の最大個体数（multi-animal 推論の上限）。SuperAnimal の
        top-down 検出器はこの値で「1フレームあたり何個体まで検出するか」を決める。
        集団飼育ケージでは頭数を必ず指定すること（例: 4匹なら 4）。
        0 以下を渡した場合はデフォルト 4 として扱う。
        ⚠️ 学術的注意: 過小指定すると一部個体が欠落し、過大指定すると床のシミ等を
        誤検出して偽個体が混入する。実験のケージ内頭数と厳密に一致させること。
    """

    # SuperAnimal-TopViewMouse (27 点) → 内部 11 点マッピング
    # ⚠️ SuperAnimal-TopViewMouse は俯瞰視点向けのため paw キーポイントが含まれない。
    #    left_shoulder / left_hip を fore_paw / hind_paw の代替に使う近似マッピング。
    #    ひっかき動作の検出感度はやや低下するため、Limitations への明記を推奨。
    SUPERANIMAL_TOPVIEW_MAP = {
        "nose":           "nose",
        "left_ear":       "left_ear",
        "right_ear":      "right_ear",
        "mouse_center":   "head_center",     # ⚠️ approximated
        "neck":           "neck",
        "left_shoulder":  "left_fore_paw",   # ⚠️ approximated (shoulder ≠ paw)
        "right_shoulder": "right_fore_paw",  # ⚠️ approximated
        "mid_back":       "spine_mid",
        "left_hip":       "left_hind_paw",   # ⚠️ approximated (hip ≠ paw)
        "right_hip":      "right_hind_paw",  # ⚠️ approximated
        "tail_base":      "tail_base",
    }

    # SuperAnimal-Quadruped (39 点) → 内部 11 点マッピング (paws を直接含む)
    # 側面/斜め視点で撮影された動画に対しては、こちらの方が解剖学的に正確。
    # DLC バージョンによってキーポイント名の表記が異なるため、
    # 両方の命名規則をまとめて定義しておく。
    SUPERANIMAL_QUADRUPED_MAP = {
        # ─ DLC 3.x 標準名 ─
        "nose":              "nose",
        "left_earbase":      "left_ear",
        "right_earbase":     "right_ear",
        "throat_base":       "neck",
        "back_middle":       "spine_mid",
        "front_left_paw":    "left_fore_paw",
        "front_right_paw":   "right_fore_paw",
        "back_left_paw":     "left_hind_paw",
        "back_right_paw":    "right_hind_paw",
        "tail_base":         "tail_base",
        # ─ 代替表記 (一部の DLC バージョン / ファインチューン済みモデル) ─
        "Nose":              "nose",
        "L_EarBase":         "left_ear",
        "R_EarBase":         "right_ear",
        "Throat":            "neck",
        "Back":              "spine_mid",
        "L_F_Paw":           "left_fore_paw",
        "R_F_Paw":           "right_fore_paw",
        "L_B_Paw":           "left_hind_paw",
        "R_B_Paw":           "right_hind_paw",
        "TailBase":          "tail_base",
        # ─ 別表記パターン ─
        "left_front_paw":    "left_fore_paw",
        "right_front_paw":   "right_fore_paw",
        "left_back_paw":     "left_hind_paw",
        "right_back_paw":    "right_hind_paw",
        # head_center: 推論時に left_ear と right_ear の中点として補間
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        superanimal_name: str = "superanimal_topviewmouse",
        detector_name: str = "fasterrcnn_resnet50_fpn_v2",
        pose_model_name: str = "hrnet_w32",
        max_individuals: int = 4,
    ):
        super().__init__(model_path, device)
        self.superanimal_name = superanimal_name
        self.detector_name = detector_name
        self.pose_model_name = pose_model_name
        # 集団飼育の頭数。0 以下はデフォルト 4 とみなす（誤指定でパイプラインが落ちないように）。
        self.max_individuals = max_individuals if max_individuals and max_individuals > 0 else 4

        # model_path の解釈:
        #   - None             → SuperAnimal Model Zoo (zero-shot)
        #   - "superanimal_*"  → 指定された Model Zoo モデルへの切り替え
        #   - 既存 *.yaml      → カスタム学習済み DLC プロジェクト
        if self.model_path and self.model_path.startswith("superanimal_"):
            self.superanimal_name = self.model_path
            self.use_modelzoo = True
            self.model_path = None
        elif self.model_path and os.path.exists(self.model_path) and self.model_path.endswith((".yaml", ".yml")):
            self.use_modelzoo = False
        else:
            self.use_modelzoo = True

    def _auto_detect_model(self) -> Optional[str]:
        """models/ から DLC config.yaml を検索。SuperAnimal の場合は None でも問題ない。"""
        if os.path.exists("models"):
            for item in os.listdir("models"):
                if item.endswith(".yaml") or item.endswith(".yml"):
                    path = os.path.join("models", item)
                    print(f"[pose_engine] [Auto-Detect] DLC config を自動検出しました: {path}")
                    return path
            # SuperAnimal用にDLCモデルキャッシュフォルダがあるか
            sa_cache = os.path.join("models", "dlc_modelzoo")
            if os.path.isdir(sa_cache):
                print(f"[pose_engine] [Auto-Detect] SuperAnimal キャッシュフォルダを検出: {sa_cache}")
        return None

    def _keypoint_map(self) -> dict:
        """現在の SuperAnimal モデルに対応するキーポイントマッピングを返す。"""
        if "topview" in self.superanimal_name.lower():
            return self.SUPERANIMAL_TOPVIEW_MAP
        elif "quadruped" in self.superanimal_name.lower():
            return self.SUPERANIMAL_QUADRUPED_MAP
        else:
            # カスタムモデルではマッピングなし → DLC bodypart 名が KEYPOINTS と一致する前提
            return {k: k for k in KEYPOINTS}

    def run(self, video_path: str) -> pd.DataFrame:
        # 1. DLC のインポート（重い依存のため遅延ロード）
        try:
            import deeplabcut as dlc
        except ImportError:
            print("[pose_engine] [Warning] deeplabcut ライブラリがインストールされていません。")
            print("[pose_engine]            インストール: pip install 'deeplabcut[modelzoo,pytorch]'")
            print("[pose_engine] [Fallback] MockPoseEngine にフォールバックします。")
            fallback = MockPoseEngine(self.model_path, self.device)
            return fallback.run(video_path)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"動画ファイルが見つかりません: {video_path}")

        # 2. 出力ディレクトリの準備
        dest_folder = tempfile.mkdtemp(prefix="dlc_")
        print(f"[pose_engine] [Info] DLC 推論用一時フォルダ: {dest_folder}")

        # 3. 推論の実行
        try:
            if self.use_modelzoo:
                # SuperAnimal Model Zoo (zero-shot inference)
                print(f"[pose_engine] [Info] DeepLabCut SuperAnimal による推論を開始: "
                      f"{self.superanimal_name} (detector={self.detector_name}, pose={self.pose_model_name})")
                print(f"[pose_engine] [Info] 初回実行時は SuperAnimal モデル (~数百MB) をダウンロードします。")

                # DLC 3.x の SuperAnimal API
                # videotype は拡張子のドットなし文字列を渡す (".mp4" → "mp4")
                videotype = os.path.splitext(video_path)[1].lstrip(".")
                # GPU デバイス指定: DLC 3.x は PyTorch を使うため CUDA_VISIBLE_DEVICES で制御する。
                # CPU モードが明示指定された場合は "-1" で GPU を無効化する。
                # DLC 2.x 系は gputouse パラメータを持つが 3.x では不要・無視されるため
                # 環境変数方式をメインとし、gputouse は後方互換のためオプションで渡す。
                if self.device == "cpu":
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                    print(f"[pose_engine] [Info] DLC 推論デバイス: CPU (CUDA_VISIBLE_DEVICES=-1)")
                else:
                    # "cuda" または自動: 既存の環境変数を尊重し、未設定なら GPU 0 を選択
                    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
                    print(f"[pose_engine] [Info] DLC 推論デバイス: GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '0')})")

                # multi-animal 上限: 集団飼育の頭数を SuperAnimal の top-down 検出器に渡す。
                # これが無いと既定値（1 もしくは過大値）で推論され、4匹中1匹しか検出されない／
                # 偽個体が混入する等、ID 割当と頭数検証（color_id.py）が破綻する。
                # DLC のバージョン差で引数名が異なるため、複数候補を順に試行する。
                print(f"[pose_engine] [Info] multi-animal 上限 max_individuals={self.max_individuals} で推論します。")
                _gpu_to_use = 0 if self.device != "cpu" else None
                _dlc_kwargs = dict(
                    videos=[video_path],
                    superanimal_name=self.superanimal_name,
                    model_name=self.pose_model_name,
                    detector_name=self.detector_name,
                    videotype=videotype,
                    dest_folder=dest_folder,
                    max_individuals=self.max_individuals,
                )
                # 引数の互換性を段階的に縮退させて呼び出す（gputouse → max_individuals の順に外す）。
                # DLC 3.x は gputouse 非対応、一部バージョンは max_individuals 非対応のため。
                _attempts = [
                    dict(gputouse=_gpu_to_use),  # 2.x 互換 + multi-animal
                    dict(),                      # 3.x: 環境変数で GPU 制御 + multi-animal
                ]
                _last_err = None
                _succeeded = False
                for _extra in _attempts:
                    try:
                        dlc.video_inference_superanimal(**_dlc_kwargs, **_extra)
                        _succeeded = True
                        break
                    except TypeError as _te:
                        _last_err = _te
                        # gputouse が原因なら次の試行（gputouse なし）で解消する
                        continue
                if not _succeeded:
                    # max_individuals 自体が非対応の可能性 → これを外して最後の試行を行う
                    print(f"[pose_engine] [Info] max_individuals 非対応バージョン検出 — 当該引数を外して再試行します。")
                    _dlc_kwargs.pop("max_individuals", None)
                    dlc.video_inference_superanimal(**_dlc_kwargs)

                # [注意] TopView モデルは paw キーポイントを持たないことを警告
                # NOTE: 推論は上で既に完了している。この print は情報提供のみ。
                # Windows の cp932 コンソール等で絵文字をエンコードできず UnicodeEncodeError が
                # 発生すると、外側の except に捕捉されて「成功済みの DLC 結果を破棄して Mock へ
                # フォールバック」する致命的な誤動作を起こす。よって (1) cp932 で表現可能な
                # 文字のみ使用し、(2) print 自体を try でガードして推論結果を絶対に巻き戻さない。
                if "topview" in self.superanimal_name.lower():
                    try:
                        print("[pose_engine] [Warning] [学術利用注意] SuperAnimal-TopViewMouse は paw キーポイントを"
                              "含まないため、left_fore_paw / left_hind_paw 等は left_shoulder / left_hip の近似値です。"
                              "解剖学的に厳密な解析には SuperAnimal-Quadruped (側面視点) または paw を含む"
                              "カスタム学習モデルを使用してください。")
                    except Exception:
                        pass
            else:
                # カスタム学習済み DLC プロジェクト
                print(f"[pose_engine] [Info] DLC analyze_videos を実行: config={self.model_path}")
                dlc.analyze_videos(
                    config=self.model_path,
                    videos=[video_path],
                    destfolder=dest_folder,
                    save_as_csv=False,
                )
        except Exception as e:
            print(f"[pose_engine] [Error] DLC の推論実行に失敗しました: {e}")
            print("[pose_engine] [Fallback] MockPoseEngine にフォールバックします。")
            fallback = MockPoseEngine(self.model_path, self.device)
            return fallback.run(video_path)

        # 4. 出力 H5 ファイルの探索
        h5_candidates = [f for f in os.listdir(dest_folder) if f.endswith(".h5")]
        if not h5_candidates:
            print(f"[pose_engine] [Warning] DLC の H5 出力が見つかりません: {dest_folder}")
            print("[pose_engine] [Fallback] MockPoseEngine にフォールバックします。")
            fallback = MockPoseEngine(self.model_path, self.device)
            return fallback.run(video_path)

        h5_path = os.path.join(dest_folder, h5_candidates[0])
        print(f"[pose_engine] [Info] DLC 結果 H5 を読み込み中: {h5_path}")

        # 5. H5 → 内部ロングフォーマットへの変換
        try:
            df = self._parse_dlc_h5(h5_path)
        except Exception as e:
            print(f"[pose_engine] [Error] DLC H5 のパースに失敗しました: {e}")
            print("[pose_engine] [Fallback] MockPoseEngine にフォールバックします。")
            fallback = MockPoseEngine(self.model_path, self.device)
            return fallback.run(video_path)

        if df.empty:
            print("[pose_engine] [Warning] DLC の検出座標が空でした。Mock にフォールバックします。")
            fallback = MockPoseEngine(self.model_path, self.device)
            return fallback.run(video_path)

        print(f"[pose_engine] [Info] DLC 推論完了: {len(df)} 行 (キーポイント数={df['keypoint'].nunique()}, "
              f"トラック数={df['track'].nunique()})")
        return df

    def _parse_dlc_h5(self, h5_path: str) -> pd.DataFrame:
        """
        DLC が出力した HDF5 ファイル (pandas MultiIndex 形式) を内部ロングフォーマットに変換する。
        DLC の列構造:
          - 単一個体    : (scorer, bodyparts, coords)            coords ∈ {x, y, likelihood}
          - 複数個体    : (scorer, individuals, bodyparts, coords)
        """
        # DLC は pandas を必ず読める形式で書き出す
        dlc_df = pd.read_hdf(h5_path)
        if not isinstance(dlc_df.columns, pd.MultiIndex):
            raise ValueError(f"DLC H5 の列形式が予期せぬ形式です: {dlc_df.columns}")

        col_levels = dlc_df.columns.names
        has_individuals = "individuals" in col_levels

        kp_map = self._keypoint_map()

        records = []
        # 単一個体の場合は擬似的に 1 個体として処理
        if has_individuals:
            individuals = dlc_df.columns.get_level_values("individuals").unique().tolist()
        else:
            individuals = ["individual_0"]

        for ind_idx, individual in enumerate(individuals):
            track_name = f"track_{ind_idx}"

            if has_individuals:
                ind_df = dlc_df.xs(individual, axis=1, level="individuals")
            else:
                ind_df = dlc_df.copy()

            # ── Bug Fix: scorer レベルを除去して常に (bodyparts, coords) 2 レベルに統一 ──
            # has_individuals=True の場合、xs() 後も (scorer, bodyparts, coords) の
            # 3 レベルが残る。droplevel(0) しないと get_level_values(0) がスコアラー名を
            # 返してしまい、bodyparts ループで全キーポイントが kp_map にヒットしなくなる。
            if ind_df.columns.nlevels > 2:
                ind_df.columns = ind_df.columns.droplevel(0)

            # bodyparts のリスト（ドットなし coords "x","y","likelihood" を除く）
            bodyparts = ind_df.columns.get_level_values(0).unique().tolist()

            # ── デバッグ: 最初の個体についてキーポイント名のヒット率をログ出力 ──
            if ind_idx == 0:
                matched = [bp for bp in bodyparts if kp_map.get(bp) is not None]
                print(f"[pose_engine] [Info] DLC bodyparts 検出数={len(bodyparts)}, "
                      f"内部マップヒット数={len(matched)}/{len(bodyparts)}")
                unmatched = [bp for bp in bodyparts if kp_map.get(bp) is None]
                if unmatched:
                    print(f"[pose_engine] [Info]   マップなし bodypart: {unmatched[:10]}"
                          + (" ..." if len(unmatched) > 10 else ""))
                if len(matched) == 0:
                    print("[pose_engine] [Warning] ⚠️ キーポイント名が内部マップと一致しません。"
                          "モデルのキーポイント名を確認し、SUPERANIMAL_*_MAP を更新してください。")

            # head_center を ear の中点として補間する必要があるか?
            need_head_center_interp = (
                "head_center" not in [kp_map.get(bp) for bp in bodyparts]
                and "left_ear" in bodyparts
                and "right_ear" in bodyparts
            )

            for f_idx in ind_df.index:
                # 行全体が NaN ならスキップ
                row = ind_df.loc[f_idx]
                if row.isna().all():
                    continue

                # head_center 補間用に ear の座標を一時保持
                lear_x = lear_y = rear_x = rear_y = lear_c = rear_c = None

                for bp in bodyparts:
                    # この bp が内部 11 点のどれにマップされるか
                    mapped_kp = kp_map.get(bp)
                    bp_norm = str(bp).lower()
                    if mapped_kp is not None and mapped_kp in KEYPOINTS:
                        # 通常の 11 点スケルトンキーポイント
                        out_kp = mapped_kp
                    elif bp_norm.startswith("tail"):
                        # [B] 尾チェーン透過保持: DLC SuperAnimal-TopViewMouse は
                        #     tail_base / tail1..tail5 / tail_end など複数の尾ノードを出力する。
                        #     これらを「色＋帯本数」個体識別 (color_id.py) の尾折れ線サンプリングに
                        #     使うため、元の名前 (小文字正規化) のままロングフォーマットへ透過させる。
                        #     11 点スケルトンには含まれないため feature_extraction では無視され、
                        #     スクラッチ判定の物理計算には一切干渉しない（非干渉プラミング）。
                        out_kp = bp_norm
                    else:
                        continue

                    try:
                        x = float(row[(bp, "x")])
                        y = float(row[(bp, "y")])
                        conf = float(row[(bp, "likelihood")]) if (bp, "likelihood") in row.index else 1.0
                    except (KeyError, ValueError, TypeError):
                        continue

                    valid = not (np.isnan(x) or np.isnan(y))
                    rec_conf = float(conf) if (valid and not np.isnan(conf)) else 0.0

                    records.append({
                        "frame":    int(f_idx),
                        "track":    track_name,
                        "keypoint": out_kp,
                        "x":        float(x) if valid else np.nan,
                        "y":        float(y) if valid else np.nan,
                        "conf":     rec_conf,
                        "is_mock":  0,
                    })

                    # ear 座標を retain (head_center 補間用)
                    if need_head_center_interp:
                        if mapped_kp == "left_ear" and valid:
                            lear_x, lear_y, lear_c = x, y, rec_conf
                        elif mapped_kp == "right_ear" and valid:
                            rear_x, rear_y, rear_c = x, y, rec_conf

                # head_center を ear の中点として補間
                if (
                    need_head_center_interp
                    and lear_x is not None and rear_x is not None
                ):
                    records.append({
                        "frame":    int(f_idx),
                        "track":    track_name,
                        "keypoint": "head_center",
                        "x":        (lear_x + rear_x) / 2.0,
                        "y":        (lear_y + rear_y) / 2.0,
                        "conf":     min(lear_c, rear_c),
                        "is_mock":  0,
                    })

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
