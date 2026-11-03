@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: 文字コードを UTF-8 に固定（既定 cp932 だと日本語ログで UnicodeEncodeError が発生し、
:: 解析が Mock に誤フォールバックすることがあるため）
set PYTHONUTF8=1

:: バッチ自身の場所をプロジェクトルートとして使用（非ASCIIパス・別ドライブでも安全）
cd /d "%~dp0"

echo.
echo =======================================================
echo  Mouse Scratch Detection System — 起動中
echo =======================================================
echo.

:: ── 環境の自動判別（conda 優先・無ければ .venv） ───────────────────
::   USE_CONDA=1 なら conda 環境 mouse-scratch を、そうでなければ .venv を使う
set "USE_CONDA="
where conda > nul 2>&1
if not errorlevel 1 (
    set "USE_CONDA=1"
)

if defined USE_CONDA (
    echo [Info] conda を検出しました。conda 環境 'mouse-scratch' を使用します。
    :: 環境が無ければ作成（python 3.14。PyTorch cu128 等は check_env.py が pip で導入）
    conda env list | findstr /C:"mouse-scratch" > nul 2>&1
    if errorlevel 1 (
        echo [Step 1/3] conda 環境 'mouse-scratch' を作成しています...
        call conda create -n mouse-scratch python=3.14 -y
        if errorlevel 1 (
            echo [Error] conda 環境の作成に失敗しました。
            pause
            exit /b 1
        )
    ) else (
        echo [Step 1/3] conda 環境 'mouse-scratch' を確認しました。
    )
    :: conda run は activate フックに依存せず実行できる（プレーンな cmd でも動作）
    set "PYRUN=conda run -n mouse-scratch --no-capture-output python"
) else (
    echo [Info] conda が見つかりません。プロジェクト直下の .venv を使用します。
    if not exist ".venv\Scripts\python.exe" (
        echo [Step 1/3] .venv を作成しています...
        py -V:3.14 -m venv .venv
        if errorlevel 1 (
            echo [Warning] py ランチャーで Python 3.14 が見つかりません。'python' で再試行します...
            python -m venv .venv
            if errorlevel 1 (
                echo [Error] .venv の作成に失敗しました。Python 3.14 をインストールしてください。
                pause
                exit /b 1
            )
        )
    ) else (
        echo [Step 1/3] .venv を確認しました。
    )
    set "PYRUN=.venv\Scripts\python.exe"
)

:: ── 依存パッケージの検査と自動インストール ─────────────────────────
echo.
echo [Step 2/3] 依存パッケージを検査し、不足分を自動インストールします...
echo            （初回は PyTorch cu128 / DeepLabCut の取得に時間がかかります）
echo.
%PYRUN% check_env.py
if errorlevel 1 (
    echo.
    echo [Error] 依存環境の準備に失敗しました。上記ログを確認してください。
    pause
    exit /b 1
)

:: ── 起動メッセージ ────────────────────────────────────────
echo.
echo [Step 3/3] FastAPI バックエンドサーバーを起動しています...
echo [Info] 数秒後にブラウザが自動で開きます
echo [Info] 手動アクセス: http://127.0.0.1:8000
echo.
echo サーバーを停止するには、このウィンドウで Ctrl+C を押してください。
echo =======================================================
echo.

:: ── 3秒後にブラウザを自動で開く ───────────────────────────
start "" /B cmd /C "timeout /t 3 /nobreak > nul && start http://127.0.0.1:8000"

:: ── FastAPI サーバー起動 ──────────────────────────────────
%PYRUN% app_server.py

echo.
echo [Info] サーバーが停止しました。
pause

endlocal
