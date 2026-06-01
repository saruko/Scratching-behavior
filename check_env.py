# -*- coding: utf-8 -*-
"""
check_env.py — 実行環境の依存パッケージを検査し、不足していれば自動インストールする。

start.bat（および Anaconda Prompt からの手動起動）から、対象 Python インタプリタ
（conda env もしくは .venv）上で呼び出される想定。本スクリプト自身が動いている
`sys.executable` に対してパッケージを検査・導入するため、どちらの環境でも同じロジックで動く。

導入方針（ユーザー合意済み）:
  - GPU（nvidia-smi 検出時）は PyTorch を **cu128** ホイールで導入する。
    RTX 5080（Blackwell / compute capability sm_120）は CUDA 12.8 以降のビルドでないと
    GPU カーネルが無く動作しないため、cu121/cu124 ではなく cu128 を用いる。
  - DeepLabCut（SuperAnimal Model Zoo）も含め、不足分は確認なしで全自動インストールする。

使い方:
  python check_env.py              # 検査し、不足を自動インストール（既定）
  python check_env.py --no-install # 検査・報告のみ（インストールしない）
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys

# ── 軽量依存（PyTorch / DeepLabCut 以外）。(import名, pip指定) の対で定義 ──
# requirements.txt と一致させること（変更時は両方を同時更新）。
BASE_PACKAGES = [
    ("fastapi",   "fastapi>=0.111.0"),
    ("uvicorn",   "uvicorn[standard]>=0.29.0"),
    ("pydantic",  "pydantic>=2.7.0"),
    ("cv2",       "opencv-python-headless>=4.9.0"),
    ("numpy",     "numpy>=1.23.0,<2.0"),   # DeepLabCut が numpy<2 を要求するため上限を固定
    ("pandas",    "pandas>=2.0.0"),
    ("h5py",      "h5py>=3.9.0"),
    ("ultralytics", "ultralytics>=8.2.0"),
    ("httpx",     "httpx>=0.27.0"),
]

# GPU 用 PyTorch ホイールの CUDA タグ。RTX 5080(sm_120) 対応のため cu128 を使用。
TORCH_CUDA_TAG = "cu128"


def _has_module(import_name: str) -> bool:
    """import せずにモジュールの存在のみ判定（DLC 等の重い import 副作用を避ける）。"""
    try:
        return importlib.util.find_spec(import_name) is not None
    except Exception:
        # 壊れたパッケージで find_spec が例外を投げる場合は「無い」とみなす
        return False


def _pip_install(*args: str) -> None:
    """現在のインタプリタの pip で installを実行。失敗時は CalledProcessError を送出。"""
    cmd = [sys.executable, "-m", "pip", "install", *args]
    print(f"[check_env] 実行: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def detect_gpu() -> bool:
    """nvidia-smi が存在し正常終了するかで GPU の有無を判定する。"""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def torch_status(gpu: bool):
    """
    PyTorch の導入状態を判定して返す:
      None        … 問題なし（再インストール不要）
      "missing"   … 未導入
      "reinstall" … 導入済みだが GPU で使えない / sm 非対応ビルド → 入れ直し要
    """
    if not _has_module("torch"):
        return "missing"
    if not gpu:
        return None  # CPU 環境では import できれば良い

    # GPU 環境では「CUDA が使えるか」「このGPUの sm がビルドに含まれるか」まで検査する。
    try:
        import torch
        if not torch.cuda.is_available():
            return "reinstall"
        major, minor = torch.cuda.get_device_capability()
        sm = f"sm_{major}{minor}"  # 例: RTX 5080 → sm_120
        arch_list = torch.cuda.get_arch_list()  # 例(cu128): [..., 'sm_90', 'sm_100', 'sm_120']
        # 当該 sm がビルド済みアーキ一覧に含まれない場合、カーネル不在で実行時に失敗するため要再導入。
        if sm not in arch_list:
            print(f"[check_env] [Warning] 現GPU {sm} が PyTorch ビルド {arch_list} に含まれません。"
                  f"cu128 ホイールへ入れ直します。")
            return "reinstall"
        return None
    except Exception as e:
        print(f"[check_env] [Warning] PyTorch GPU 検査中に例外: {e} → 入れ直します。")
        return "reinstall"


def install_torch(gpu: bool, force: bool) -> None:
    """GPU 有無に応じた index-url で torch / torchvision を導入する。"""
    if gpu:
        index = f"https://download.pytorch.org/whl/{TORCH_CUDA_TAG}"
        print(f"[check_env] GPU 検出: PyTorch を {TORCH_CUDA_TAG} で導入します。")
    else:
        index = "https://download.pytorch.org/whl/cpu"
        print("[check_env] GPU 未検出: PyTorch を CPU 版で導入します。")
    args = ["torch", "torchvision", "--index-url", index]
    if force:
        # cu121 等の非対応ビルドを cu128 に置き換える場合は強制再インストール
        args.append("--force-reinstall")
    _pip_install(*args)


def ensure_base_packages() -> None:
    """軽量依存のうち未導入のものだけまとめて導入する。"""
    missing = [spec for imp, spec in BASE_PACKAGES if not _has_module(imp)]
    if missing:
        print(f"[check_env] 不足している基本パッケージ: {missing}")
        _pip_install(*missing)
    else:
        print("[check_env] 基本パッケージはすべて導入済みです。")


def ensure_deeplabcut() -> None:
    """DeepLabCut（SuperAnimal Model Zoo + PyTorch エンジン）を導入する。"""
    if _has_module("deeplabcut"):
        print("[check_env] DeepLabCut は導入済みです。")
        return
    print("[check_env] DeepLabCut が未導入のため導入します（数百MB〜のダウンロードあり）。")
    _pip_install("deeplabcut[modelzoo,pytorch]>=3.0.0")


def verify(gpu: bool) -> bool:
    """最終確認。必須モジュールが import 可能か、GPU 環境なら CUDA が有効かを表示・判定する。"""
    ok = True
    print("\n[check_env] ── 最終確認 ──")
    for imp, _ in BASE_PACKAGES:
        present = _has_module(imp)
        ok = ok and present
        print(f"  {'OK ' if present else 'NG '} {imp}")
    torch_ok = _has_module("torch")
    print(f"  {'OK ' if torch_ok else 'NG '} torch")
    ok = ok and torch_ok
    if torch_ok and gpu:
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            name = torch.cuda.get_device_name(0) if cuda_ok else "なし"
            print(f"  {'OK ' if cuda_ok else 'NG '} CUDA 利用可: {cuda_ok} / デバイス: {name} / torch {torch.__version__}")
            ok = ok and cuda_ok
        except Exception as e:
            print(f"  NG  CUDA 検査で例外: {e}")
            ok = False
    dlc_ok = _has_module("deeplabcut")
    # DeepLabCut は任意機能（DLC エンジン使用時のみ必須）。未導入でも致命的ではないが状態は表示する。
    print(f"  {'OK ' if dlc_ok else '-- '} deeplabcut（DLCエンジン使用時に必要）")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="依存環境の検査と自動インストール")
    parser.add_argument("--no-install", action="store_true",
                        help="検査・報告のみ行い、インストールはしない")
    parser.add_argument("--skip-dlc", action="store_true",
                        help="DeepLabCut の導入をスキップする（軽量起動用）")
    args = parser.parse_args()

    gpu = detect_gpu()
    print(f"[check_env] Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"[check_env] GPU 検出: {gpu}")

    if args.no_install:
        # 報告のみモード: 何が足りないかを表示して終了
        print("[check_env] --no-install 指定: 検査のみ行います。")
        verify(gpu)
        return 0

    # 1. 軽量依存
    ensure_base_packages()

    # 2. PyTorch（GPU 対応を厳密検査して必要なら入れ直し）
    status = torch_status(gpu)
    if status == "missing":
        install_torch(gpu, force=False)
    elif status == "reinstall":
        install_torch(gpu, force=True)
    else:
        print("[check_env] PyTorch は要件を満たしています。")

    # 3. DeepLabCut（指定がなければ導入）
    if not args.skip_dlc:
        ensure_deeplabcut()
        # DLC 導入が torch を別ビルドに差し替えてしまった場合に備え、GPU 環境では再検査して必要なら直す。
        if gpu:
            status2 = torch_status(gpu)
            if status2 == "reinstall":
                print("[check_env] DeepLabCut 導入後に PyTorch GPU 構成が崩れたため cu128 で入れ直します。")
                install_torch(gpu, force=True)

    all_ok = verify(gpu)
    if all_ok:
        print("\n[check_env] [OK] 環境準備完了。")
        return 0
    print("\n[check_env] [Warning] 一部のパッケージが揃っていません。上記 NG 項目を確認してください。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
