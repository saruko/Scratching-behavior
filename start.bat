@echo off
chcp 65001 > nul
echo.
echo =======================================================
echo  Mouse Scratch Detection System — 起動中
echo =======================================================
echo.

:: ── 環境の確認 ────────────────────────────────────────────
call conda run -n mouse-scratch python --version > nul 2>&1
if errorlevel 1 (
    echo [Error] conda 環境 'mouse-scratch' が見つかりません。
    echo         初回は setup.bat を実行してください。
    echo.
    pause
    exit /b 1
)

:: ── 起動メッセージ ────────────────────────────────────────
echo [Info] FastAPI バックエンドサーバーを起動しています...
echo [Info] 数秒後にブラウザが自動で開きます
echo [Info] 手動アクセス: http://127.0.0.1:8000
echo.
echo サーバーを停止するには、このウィンドウで Ctrl+C を押してください。
echo =======================================================
echo.

:: ── 3秒後にブラウザを自動で開く ───────────────────────────
start "" /B cmd /C "timeout /t 3 /nobreak > nul && start http://127.0.0.1:8000"

:: ── FastAPI サーバー起動 ──────────────────────────────────
call conda run -n mouse-scratch python app_server.py

echo.
echo [Info] サーバーが停止しました。
pause
