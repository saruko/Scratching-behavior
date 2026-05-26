@echo off
chcp 65001 > nul
echo.
echo =======================================================
echo  Mouse Scratch Detection System
echo  初回セットアップ (Windows + Anaconda + Nvidia GPU)
echo =======================================================
echo.

:: ── Anaconda の確認 ───────────────────────────────────────
where conda > nul 2>&1
if errorlevel 1 (
    echo [Error] Anaconda が見つかりません。
    echo.
    echo 以下のURLから Anaconda または Miniconda をインストールしてください:
    echo   https://www.anaconda.com/download
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)

:: ── GPU 確認 ──────────────────────────────────────────────
echo [Step 0] Nvidia GPU を確認しています...
nvidia-smi > nul 2>&1
if errorlevel 1 (
    echo [Warning] nvidia-smi が見つかりません。CPU 動作モードになります。
    echo           GPU を使用したい場合は Nvidia ドライバーをインストールしてください。
    set CUDA_VER=cpu
) else (
    echo [OK] Nvidia GPU が検出されました。
    echo.
    echo ※ CUDA バージョンを確認してください (nvidia-smi の右上 "CUDA Version"):
    echo    - CUDA 12.x の場合: environment.yml の pytorch-cuda=12.1 のままで OK
    echo    - CUDA 11.x の場合: environment.yml の pytorch-cuda=12.1 を 11.8 に変更してください
    echo.
    pause
)

:: ── conda 環境の作成 ─────────────────────────────────────
echo.
echo [Step 1/4] conda 環境 'mouse-scratch' を作成しています...
echo            (初回は数分かかります。インターネット接続が必要です)
echo.

call conda env create -f environment.yml
if errorlevel 1 (
    echo.
    echo [Warning] 新規作成に失敗しました。既存環境の更新を試みます...
    call conda env update -n mouse-scratch -f environment.yml --prune
    if errorlevel 1 (
        echo.
        echo [Error] 環境の作成/更新に失敗しました。
        echo         CUDA バージョンが合っているか environment.yml を確認してください。
        pause
        exit /b 1
    )
)

echo.
echo [Step 2/4] GPU / PyTorch 動作確認中...
call conda run -n mouse-scratch python -c ^
    "import torch; ^
     gpu = torch.cuda.is_available(); ^
     name = torch.cuda.get_device_name(0) if gpu else 'なし'; ^
     print('[GPU] CUDA:', gpu, '/ デバイス:', name)"

:: ── models/ フォルダ確認 ──────────────────────────────────
echo.
echo [Step 3/4] models/ フォルダを確認しています...
if not exist "models" mkdir models
echo [OK] models/ フォルダを確認しました。

:: ── モデルの準備確認 ─────────────────────────────────────
echo.
echo [Step 4/4] 推定エンジンとデフォルトモデルを確認しています...
call conda run -n mouse-scratch python setup_models.py

echo.
echo =======================================================
echo  セットアップが完了しました！
echo =======================================================
echo.
echo 次回からは start.bat をダブルクリックするだけでシステムが起動します。
echo.
pause
