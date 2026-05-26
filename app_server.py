"""
Mouse Scratch Detection System - FastAPI Backend Server
======================================================
本スクリプトは、静的なダッシュボード（index.html等）をブラウザへ配信し、
GUIからのリクエストに応じて非同期で動画解析パイプラインを実行するバックエンドサーバーです。
"""

import os
import sys
import uuid
import json
import subprocess
import asyncio
import threading
from typing import Dict, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="Mouse Scratch Analyzer API",
    description="FastAPI backend to run the behavior detection pipeline and serve dashboard files."
)

# --- Task DB Persistence Helper [B-4] ---
TASKS_DB_FILE = "tasks_db.json"
TASKS_DB_LOCK = threading.Lock()

def load_tasks_db() -> Dict[str, dict]:
    with TASKS_DB_LOCK:
        if os.path.exists(TASKS_DB_FILE):
            try:
                with open(TASKS_DB_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load tasks db from file: {e}")
        return {}

def save_tasks_db(db: Dict[str, dict]):
    with TASKS_DB_LOCK:
        try:
            temp_file = TASKS_DB_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, TASKS_DB_FILE)
        except Exception as e:
            print(f"[Error] Failed to save tasks db to file: {e}")

# タスクの進捗ステータス保持用
# 構造: { task_id: { "status": "running"|"completed"|"failed", "video": str, "log": str, "warning": str } }
TASKS_DB: Dict[str, dict] = load_tasks_db()

class AnalysisRequest(BaseModel):
    video_path: str
    engine: str = "yolov8"     # sleap | yolov8 | mock
    model_path: Optional[str] = None
    fps: Optional[int] = None
    px_to_mm: Optional[float] = 0.25

    model_config = {
        "protected_namespaces": ()
    }

# --- 1. Dashboard Static Files & Direct Routing ---

@app.get("/")
async def get_index():
    """ルートパスでメインダッシュボード index.html を返します。"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    raise HTTPException(status_code=404, detail="index.html が見つかりません。")

# NOTE: FastAPIはルートを登録順に評価するため、
# より具体的なパス (/outputs/{filename}) を先に定義しておく必要がある。
@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """解析完了後に出力されたCSVなどの結果ファイルを配信します。"""
    path = os.path.join("outputs", filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="text/csv")
    raise HTTPException(status_code=404, detail=f"出力ファイルが見つかりません: {filename}")

@app.get("/outputs/{subdir}/{filename}")
async def get_output_subdir_file(subdir: str, filename: str):
    """モデル別結果フォルダなどのサブディレクトリ配下にある結果ファイルを安全に配信します。"""
    if ".." in subdir or ".." in filename:
        raise HTTPException(status_code=400, detail="不正なパスが検出されました。")
    path = os.path.join("outputs", subdir, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="text/csv")
    raise HTTPException(status_code=404, detail=f"出力ファイルが見つかりません: {subdir}/{filename}")

@app.get("/api/config")
async def get_config():
    """
    Python側で設定された閾値（速度、距離、振動回数）などの定数情報をフロントエンドへ提供します。
    """
    import config
    return {
        # 物理単位 (mm/s, mm) — 推奨
        "SCRATCH_MIN_VELOCITY_MM_S":      config.SCRATCH_MIN_VELOCITY_MM_S,
        "SCRATCH_MAX_HEAD_DIST_MM":       config.SCRATCH_MAX_HEAD_DIST_MM,
        "SCRATCH_MIN_OSCILLATIONS":       config.SCRATCH_MIN_OSCILLATIONS,
        "SCRATCH_MIN_OSC_AMPLITUDE_PX":   config.SCRATCH_MIN_OSC_AMPLITUDE_PX,
        "SCRATCH_VELOCITY_SMOOTH_FRAMES": config.SCRATCH_VELOCITY_SMOOTH_FRAMES,
        "SCRATCH_WINDOW_FRAMES":          config.SCRATCH_WINDOW_FRAMES,
        # Bout 後処理パラメータ（フロント GUI 側の再 Bout 化で同一閾値を使用する）
        "BOUT_MIN_FRAMES":                config.BOUT_MIN_FRAMES,
        "BOUT_MERGE_GAP":                 config.BOUT_MERGE_GAP,
        # 補間パラメータ
        "INTERPOLATION_MAX_GAP":          config.INTERPOLATION_MAX_GAP,
        # カメラ・FPS
        "PIXEL_TO_MM_RATIO":              config.PIXEL_TO_MM_RATIO,
        "DEFAULT_FPS":                    config.DEFAULT_FPS,
        # 後方互換 (px 単位, DEPRECATED)
        "SCRATCH_MIN_VELOCITY":  config.SCRATCH_MIN_VELOCITY,
        "SCRATCH_MAX_HEAD_DIST": config.SCRATCH_MAX_HEAD_DIST,
    }

@app.get("/api/results")
async def list_results():
    """
    outputs/ ディレクトリ配下にある、ロード可能な結果CSVファイルの一覧を再帰的にスキャンして返します。
    """
    results = []
    if os.path.exists("outputs"):
        for root, dirs, files in os.walk("outputs"):
            for file in files:
                if file.endswith("_frame_labels.csv"):
                    rel_dir = os.path.relpath(root, "outputs")
                    subdir = "" if rel_dir == "." else rel_dir
                    
                    # 関連する bouts.csv もペアとして特定
                    video_name = file.replace("_frame_labels.csv", "")
                    bouts_file = f"{video_name}_bouts.csv"
                    has_bouts = os.path.exists(os.path.join(root, bouts_file))
                    
                    results.append({
                        "name": f"{subdir}/{file}" if subdir else file,
                        "subdir": subdir,
                        "filename": file,
                        "frame_csv_url": f"/outputs/{subdir}/{file}" if subdir else f"/outputs/{file}",
                        "bouts_csv_url": f"/outputs/{subdir}/{bouts_file}" if subdir and has_bouts else (f"/outputs/{bouts_file}" if has_bouts else None)
                    })
    return {"results": results}


class EvaluateRequest(BaseModel):
    auto_frame_csv: str          # outputs/ 配下の _frame_labels.csv パス (サーバー相対)
    gt_csv: str                  # グラウンドトゥルース CSV の絶対パスまたはサーバー相対パス
    gt_format: str = "frame"     # "frame" | "bout"
    fps: float = 30.0
    iou_threshold: float = 0.5
    auto_bouts_csv: Optional[str] = None


@app.post("/api/evaluate")
async def run_evaluation(req: EvaluateRequest):
    """
    自動検出 CSV とグラウンドトゥルース CSV を比較してフレーム/Bout レベルの精度指標を返します。
    """
    try:
        import evaluate as ev
    except ImportError:
        raise HTTPException(status_code=500, detail="evaluate モジュールが見つかりません。")

    # パスの安全解決 (outputs/ 配下を想定)
    def resolve(p: str) -> str:
        if os.path.exists(p):
            return p
        candidate = os.path.join("outputs", p)
        if os.path.exists(candidate):
            return candidate
        raise HTTPException(status_code=400, detail=f"ファイルが見つかりません: {p}")

    auto_frame_path = resolve(req.auto_frame_csv)
    gt_path         = resolve(req.gt_csv)
    auto_bouts_path = resolve(req.auto_bouts_csv) if req.auto_bouts_csv else None

    try:
        result = ev.run_evaluation(
            auto_frame_csv=auto_frame_path,
            gt_csv=gt_path,
            gt_format=req.gt_format,
            fps=req.fps,
            iou_threshold=req.iou_threshold,
            output_dir="outputs",
            auto_bouts_csv=auto_bouts_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"評価処理中にエラーが発生しました: {e}")

    return {
        "frame_metrics": result["frame_metrics"].to_dict(orient="records") if not result["frame_metrics"].empty else [],
        "bout_metrics":  result["bout_metrics"].to_dict(orient="records")  if result["bout_metrics"] is not None and not result["bout_metrics"].empty else [],
        "output_files":  result["output_files"],
    }


@app.get("/{filename}")
async def get_static_file(filename: str):
    """HTML5ダッシュボードに必要なCSS、JavaScript、および検証用動画ファイルを安全に配信します。"""
    allowed_exts = {".js", ".css", ".mp4", ".mov", ".avi", ".html", ".ico"}
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in allowed_exts and os.path.exists(filename):
        # JavaScriptやCSSのMIMEタイプを明示
        media_types = {
            ".js": "application/javascript",
            ".css": "text/css",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo"
        }
        return FileResponse(filename, media_type=media_types.get(ext))
        
    raise HTTPException(status_code=404, detail=f"ファイルが見つかりません: {filename}")

# --- 2. Background task for running pipeline.py ---

def run_pipeline_task(task_id: str, req: AnalysisRequest):
    """
    バックグラウンドで pipeline.py を呼び出し、ログを監視するタスク。
    """
    TASKS_DB[task_id]["status"] = "running"
    save_tasks_db(TASKS_DB)
    
    # 確実に同じPythonインタプリタの実行環境を使用するため、sys.executableを使用する
    cmd = [sys.executable, "pipeline.py"]
        
    cmd.extend([
        "--video", req.video_path,
        "--engine", req.engine,
        "--px-to-mm", str(req.px_to_mm)
    ])
    if req.model_path:
        cmd.extend(["--model_path", req.model_path])
    if req.fps:
        cmd.extend(["--fps", str(req.fps)])

    TASKS_DB[task_id]["log"] += f"[Server] 解析プロセスを開始します...\n[Server] コマンド: {' '.join(cmd)}\n\n"
    TASKS_DB[task_id]["log"] += "[Progress: 5%]\n"
    save_tasks_db(TASKS_DB)
    
    try:
        # サブプロセスでもUTF-8での標準出力を強制するため環境変数を設定
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        is_mock_used = False
        # ログのリアルタイム取得 + 進捗マーカーの自動挿入
        line_count = 0
        while True:
            line = process.stdout.readline()
            if not line:
                break
            TASKS_DB[task_id]["log"] += line
            
            # 各パイプラインステップに応じた進捗率を自動挿入
            progress_changed = False
            if "[Step 1]" in line or "カラーIDの自動アライメント" in line:
                TASKS_DB[task_id]["log"] += "[Progress: 40%]\n"
                progress_changed = True
            elif "[Step 2]" in line or "特徴量抽出" in line:
                TASKS_DB[task_id]["log"] += "[Progress: 70%]\n"
                progress_changed = True
            elif "[Step 3]" in line or "CSV書き出し" in line:
                TASKS_DB[task_id]["log"] += "[Progress: 90%]\n"
                progress_changed = True
            
            # モックフォールバックや重大警告の検出
            if "[Fallback] " in line or "[Mock]" in line:
                is_mock_used = True
                
            line_count += 1
            if progress_changed or line_count % 20 == 0:
                save_tasks_db(TASKS_DB)
                
        process.wait()
        
        if process.returncode == 0:
            TASKS_DB[task_id]["status"] = "completed"
            TASKS_DB[task_id]["log"] += "\n[Progress: 100%]\n"
            TASKS_DB[task_id]["log"] += "\n[Server] 解析パイプラインが正常に完了しました。"
            if is_mock_used or req.engine == "mock":
                TASKS_DB[task_id]["warning"] = "⚠️ 警告: 実際の姿勢推定モデルの代わりに、ダミーデータ(MockPoseEngine)が使用されました。このデータは学術論文に使用しないでください。"
        else:
            TASKS_DB[task_id]["status"] = "failed"
            TASKS_DB[task_id]["log"] += f"\n[Server] [Error] プロセスがエラー終了しました (Exit Code: {process.returncode})"
        
        save_tasks_db(TASKS_DB)
            
    except Exception as e:
        TASKS_DB[task_id]["status"] = "failed"
        TASKS_DB[task_id]["log"] += f"\n[Server] [Error] 解析プロセスの実行中に例外が発生しました: {e}"
        save_tasks_db(TASKS_DB)

# --- 3. Analysis Orchestrator API Endpoints ---

@app.post("/api/analyze")
async def start_analysis(req: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    動画ファイルに対するひっかき行動解析処理を非同期で開始します。
    """
    # 動画ファイルの存在確認（絶対パス → 相対パス → ファイル名のみ の順で解決を試みる）
    resolved_path = None
    candidates = [
        req.video_path,
        os.path.basename(req.video_path),
        os.path.join(os.getcwd(), os.path.basename(req.video_path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            resolved_path = candidate
            break
    
    if resolved_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"動画ファイルが見つかりません: {req.video_path}\n"
                   f"サーバーの作業ディレクトリ: {os.getcwd()}\n"
                   f"試行したパス: {candidates}"
        )
    req.video_path = resolved_path
            
    task_id = str(uuid.uuid4())
    TASKS_DB[task_id] = {
        "status": "pending",
        "video": req.video_path,
        "log": "",
        "warning": ""
    }
    save_tasks_db(TASKS_DB)
    
    # バックグラウンドタスクとして登録して即座に応答
    background_tasks.add_task(run_pipeline_task, task_id, req)
    
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "解析タスクが非同期で開始されました。"
    }

@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    """
    指定された解析タスクの現在の進捗状況およびログを取得します。
    """
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="指定されたタスクIDが見つかりません。")
        
    task_info = TASKS_DB[task_id]
    
    # 完了時の出力CSVのファイルパスを自動特定
    result_files = {}
    if task_info["status"] == "completed":
        video_basename = os.path.splitext(os.path.basename(task_info["video"]))[0]
        frame_csv = f"/outputs/{video_basename}_frame_labels.csv"
        bouts_csv = f"/outputs/{video_basename}_bouts.csv"
        
        # 実際にサーバー上に存在するか確認
        if os.path.exists(os.path.join("outputs", f"{video_basename}_frame_labels.csv")):
            result_files = {
                "frame_labels_url": frame_csv,
                "bouts_url": bouts_csv
            }
            
    return {
        "task_id": task_id,
        "status": task_info["status"],
        "warning": task_info["warning"],
        "log_tail": task_info["log"][-2000:], # 最新2000文字のログ
        "log_full": task_info["log"],
        "results": result_files
    }

if __name__ == "__main__":
    import uvicorn
    # 直接起動時のデフォルトポート 8000
    # コラボおよびlocaltunnel接続に対応するため、ホストを 0.0.0.0、リロードを False に変更
    uvicorn.run("app_server:app", host="0.0.0.0", port=8000, reload=False)

