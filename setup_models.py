"""
モデル自動準備スクリプト
========================
起動前に利用可能な姿勢推定エンジンとモデルを確認・準備します。
- YOLOv8: ultralytics が自動ダウンロード (yolov8n-pose.pt)
- SLEAP  : 別途インストールが必要（H5ファイルのインポートは常に可能）
"""

import os
import sys
import shutil


MODELS_DIR = "models"
YOLOV8_DEFAULT_MODEL = "yolov8n-pose.pt"


def check_yolov8() -> bool:
    """YOLOv8 エンジンの確認とデフォルトモデルの準備"""
    try:
        from ultralytics import YOLO
        import torch

        gpu_info = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )

        model_dest = os.path.join(MODELS_DIR, YOLOV8_DEFAULT_MODEL)
        if not os.path.exists(model_dest):
            print(f"[models] YOLOv8 モデル '{YOLOV8_DEFAULT_MODEL}' をダウンロードします...")
            # ultralytics は初回ロード時に自動ダウンロードする
            model = YOLO(YOLOV8_DEFAULT_MODEL)
            # ダウンロード後のファイルを models/ へ移動
            if os.path.exists(YOLOV8_DEFAULT_MODEL):
                shutil.move(YOLOV8_DEFAULT_MODEL, model_dest)
                print(f"[models] ✅ モデルを配置しました: {model_dest}")
            else:
                # ultralytics がキャッシュに保存した場合の探索
                candidates = [
                    os.path.join(os.path.expanduser("~"), ".cache", "ultralytics", "assets", YOLOV8_DEFAULT_MODEL),
                    os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Ultralytics", YOLOV8_DEFAULT_MODEL),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        shutil.copy2(c, model_dest)
                        print(f"[models] ✅ キャッシュからモデルを配置しました: {model_dest}")
                        break
                else:
                    print(f"[models] ✅ YOLOv8 は利用可能です (モデルは実行時に自動取得)")
        else:
            print(f"[models] ✅ YOLOv8 モデルを確認しました: {model_dest}")

        print(f"[models]    推論デバイス: {gpu_info}")
        return True

    except ImportError:
        print("[models] ⚠️  ultralytics がインストールされていません。")
        print("           pip install ultralytics  でインストールしてください。")
        return False
    except Exception as e:
        print(f"[models] ⚠️  YOLOv8 の確認中にエラーが発生しました: {e}")
        return False


def check_sleap() -> bool:
    """SLEAP エンジンの確認 (CLI / Python パッケージ両方)"""
    # CLI コマンド確認
    try:
        import subprocess
        result = subprocess.run(
            ["sleap-track", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        if result.returncode == 0:
            print("[models] ✅ SLEAP CLI (sleap-track) が利用可能です")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Python パッケージ確認
    try:
        import sleap
        print(f"[models] ✅ SLEAP Python パッケージが利用可能です (v{sleap.__version__})")
        return True
    except ImportError:
        pass

    print("[models] ℹ️  SLEAP は未インストールです。")
    print("           SLEAP の姿勢推定を使用したい場合:")
    print("             conda create -n sleap -c sleap -c nvidia -c conda-forge sleap")
    print("           ※ SLEAP .h5 ファイルのインポートは常に使用可能です。")
    return False


def check_h5py() -> bool:
    """SLEAP H5 ファイル読み込みに必要な h5py の確認"""
    try:
        import h5py
        print(f"[models] ✅ h5py が利用可能です (SLEAP .h5 インポート対応)")
        return True
    except ImportError:
        print("[models] ⚠️  h5py がインストールされていません。SLEAP .h5 の読み込みに必要です。")
        return False


def list_available_models():
    """models/ ディレクトリのモデルファイルを一覧表示"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)

    model_files = []
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if f.endswith((".pt", ".slp", ".h5", ".pkl", ".json")):
                rel = os.path.relpath(os.path.join(root, f), MODELS_DIR)
                size_mb = os.path.getsize(os.path.join(root, f)) / 1_048_576
                model_files.append((rel, size_mb))

    if model_files:
        print(f"\n[models] 配置済みモデル ({len(model_files)} 件):")
        for name, size in model_files:
            print(f"  - {name}  ({size:.1f} MB)")
    else:
        print("\n[models] ℹ️  models/ にモデルファイルはありません。")
        print("           YOLOv8 は初回実行時に自動ダウンロードされます。")


if __name__ == "__main__":
    print("=" * 58)
    print(" モデル準備チェック (setup_models.py)")
    print("=" * 58)

    os.makedirs(MODELS_DIR, exist_ok=True)

    print()
    yolo_ok = check_yolov8()
    print()
    sleap_ok = check_sleap()
    print()
    h5_ok = check_h5py()

    list_available_models()

    print()
    print("=" * 58)
    print(" 推奨起動コマンド:")
    if yolo_ok:
        print("  python pipeline.py --video <動画> --engine yolov8")
    elif sleap_ok:
        print("  python pipeline.py --video <動画> --engine sleap")
    else:
        print("  python pipeline.py --video <動画> --engine mock  (デモ用)")
    print()
    print("  Web ダッシュボード: python app_server.py")
    print("=" * 58)
    sys.exit(0)
