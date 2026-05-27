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
    engine: str = "yolov8"         # sleap | yolov8 | dlc | mock
    model_path: Optional[str] = None
    fps: Optional[int] = None
    px_to_mm: Optional[float] = 0.25
    # マウス頭数設定 (0 = 自動検出 — 動画途中で個体数が増減する場合も 0 を推奨)
    mice_count: int = 0
    # ROI 設定 (x1, y1, x2, y2 — None の場合は全フレームを対象)
    roi_x1: Optional[int] = None
    roi_y1: Optional[int] = None
    roi_x2: Optional[int] = None
    roi_y2: Optional[int] = None
    # 推論デバイス指定 (None = GPU互換性チェック後に自動選択, "cuda" = GPU強制, "cpu" = CPU強制)
    device: Optional[str] = None
    # DLC 専用パラメータ (engine == "dlc" 時に有効)
    dlc_superanimal: Optional[str] = "superanimal_topviewmouse"  # or "superanimal_quadruped"
    dlc_detector: Optional[str] = "fasterrcnn_resnet50_fpn_v2"
    dlc_pose_model: Optional[str] = "hrnet_w32"

    # ── フェーズ1 初期化確認モード ─────────────────────────────────────────
    # preview_only=True の場合、動画冒頭のみ解析して _preview.json を出力する。
    # confirmed_mapping: フロントエンドで確認済みのマッピング JSON 文字列。
    preview_only: bool = False
    preview_duration_s: float = 300.0  # 5分 = 300秒
    confirmed_mapping: Optional[str] = None  # JSON文字列: {"track_0": 1, ...}

    model_config = {
        "protected_namespaces": ()
    }


class ConfigUpdateRequest(BaseModel):
    """
    実行時にスクラッチ判定閾値を動的に変更するリクエストモデル。
    フロントエンドの閾値エディタパネルからPOSTされる。
    変更はプロセスのライフタイム中のみ有効（再起動でリセット）。
    """
    SCRATCH_MIN_VELOCITY_MM_S: Optional[float] = None
    SCRATCH_MAX_HEAD_DIST_MM: Optional[float] = None
    SCRATCH_MIN_OSCILLATIONS: Optional[int] = None
    SCRATCH_MIN_OSC_AMPLITUDE_MM: Optional[float] = None
    SCRATCH_MIN_OSC_AMPLITUDE_PX: Optional[float] = None  # 後方互換（旧クライアント向け）
    SCRATCH_VELOCITY_SMOOTH_FRAMES: Optional[int] = None
    SCRATCH_WINDOW_FRAMES: Optional[int] = None
    BOUT_MIN_FRAMES: Optional[int] = None
    BOUT_MERGE_GAP: Optional[int] = None
    PIXEL_TO_MM_RATIO: Optional[float] = None
    DEFAULT_FPS: Optional[float] = None
    CAGE_MICE_COUNT: Optional[int] = None
    BODY_LENGTH_MIN_MM: Optional[float] = None
    BODY_LENGTH_MAX_MM: Optional[float] = None
    ENABLE_ANATOMICAL_CHECKS: Optional[bool] = None


class CorrectionsRequest(BaseModel):
    """
    GUI 上で手動補正されたフレームラベルデータを受け取り、
    既存のCSVファイルを上書き保存するリクエストモデル。
    """
    csv_path: str       # outputs/ 配下の既存 CSV のパス（相対または絶対）
    csv_content: str    # クライアントが生成した補正済み CSV テキスト全体

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
        # 振動振幅閾値: 新規 (mm単位、推奨) + 旧 (px単位、後方互換)
        "SCRATCH_MIN_OSC_AMPLITUDE_MM":   config.SCRATCH_MIN_OSC_AMPLITUDE_MM,
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

@app.post("/api/config-update")
async def update_config(req: ConfigUpdateRequest):
    """
    実行時にスクラッチ判定閾値を動的に変更します。
    変更はメモリ上のみ有効で、サーバー再起動でリセットされます。
    フロントエンドの解析設定パネルからの変更リクエストを処理します。
    """
    import config
    changed = {}
    for field, value in req.model_dump(exclude_none=True).items():
        if hasattr(config, field):
            old_val = getattr(config, field)
            setattr(config, field, value)
            changed[field] = {"old": old_val, "new": value}
            print(f"[Server] [Config] {field}: {old_val} → {value}")

    if not changed:
        return JSONResponse({"status": "no_change", "changed": {}})

    return JSONResponse({"status": "updated", "changed": changed})


@app.post("/api/save-corrections")
async def save_corrections(req: CorrectionsRequest):
    """
    GUI で手動補正されたフレームラベル CSV をサーバーに保存します。
    セキュリティ: outputs/ ディレクトリ外へのパス トラバーサルを防止します。
    """
    # パスの安全解決: outputs/ 配下のみ許可
    safe_name = os.path.basename(req.csv_path)
    if not safe_name.endswith(".csv"):
        raise HTTPException(status_code=400, detail="保存対象は .csv ファイルのみです。")
    save_path = os.path.join("outputs", safe_name)

    # ディレクトリが存在しない場合は作成
    os.makedirs("outputs", exist_ok=True)

    try:
        # バックアップを作成（既存ファイルがある場合）
        if os.path.exists(save_path):
            backup_path = save_path.replace(".csv", "_backup.csv")
            with open(save_path, "r", encoding="utf-8") as f:
                with open(backup_path, "w", encoding="utf-8") as bf:
                    bf.write(f.read())

        # 補正済みCSVを書き込み
        with open(save_path, "w", encoding="utf-8", newline="") as f:
            f.write(req.csv_content)

        return JSONResponse({
            "status": "saved",
            "path": save_path,
            "bytes": len(req.csv_content.encode("utf-8"))
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV保存中にエラーが発生しました: {e}")


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
    if req.device:
        cmd.extend(["--device", req.device])
    if req.fps:
        cmd.extend(["--fps", str(req.fps)])
    if req.mice_count and req.mice_count > 0:
        cmd.extend(["--mice-count", str(req.mice_count)])
    # ROI 指定: 全 4 値が揃っている場合のみ pipeline.py へ伝達
    if all(v is not None for v in [req.roi_x1, req.roi_y1, req.roi_x2, req.roi_y2]):
        cmd.extend(["--roi-x1", str(req.roi_x1), "--roi-y1", str(req.roi_y1),
                    "--roi-x2", str(req.roi_x2), "--roi-y2", str(req.roi_y2)])
    # DLC エンジン専用パラメータ
    if req.engine == "dlc":
        if req.dlc_superanimal:
            cmd.extend(["--dlc-superanimal", req.dlc_superanimal])
        if req.dlc_detector:
            cmd.extend(["--dlc-detector", req.dlc_detector])
        if req.dlc_pose_model:
            cmd.extend(["--dlc-pose-model", req.dlc_pose_model])

    # ── フェーズ1 初期化確認モード引数 ──────────────────────────────────────
    if req.preview_only:
        cmd.extend(["--preview-only", "--preview-duration", str(req.preview_duration_s)])
    if req.confirmed_mapping:
        cmd.extend(["--confirmed-mapping", req.confirmed_mapping])

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
            if "[Preview Mode]" in line or "初期化確認用プレビュー" in line:
                TASKS_DB[task_id]["log"] += "[Progress: 20%]\n"
                progress_changed = True
            elif "[Preview] プレビューJSON" in line:
                TASKS_DB[task_id]["log"] += "[Progress: 95%]\n"
                progress_changed = True
            elif "[Step 1]" in line or "カラーIDの自動アライメント" in line:
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
        "warning": "",
        "task_type": "preview" if req.preview_only else "analysis",
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
    
    # 完了時の出力ファイルパスを自動特定（タスク種別で分岐）
    result_files = {}
    if task_info["status"] == "completed":
        video_basename = os.path.splitext(os.path.basename(task_info["video"]))[0]
        task_type = task_info.get("task_type", "analysis")

        if task_type == "preview":
            # プレビューモード: _preview.json を返す
            preview_json = os.path.join("outputs", f"{video_basename}_preview.json")
            if os.path.exists(preview_json):
                result_files = {
                    "preview_data_url": f"/api/preview_data/{video_basename}",
                    "video_name": video_basename,
                }
        else:
            # 本解析: frame_labels.csv と bouts.csv を返す
            frame_csv = f"/outputs/{video_basename}_frame_labels.csv"
            bouts_csv = f"/outputs/{video_basename}_bouts.csv"
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

# --- 4. フェーズ1 初期化確認モード API ---

@app.get("/api/preview_data/{video_name}")
async def get_preview_data(video_name: str):
    """
    指定動画のプレビューJSONを返す。
    pipeline.py --preview-only で生成された {video_name}_preview.json を配信する。
    """
    preview_path = os.path.join("outputs", f"{video_name}_preview.json")
    if not os.path.exists(preview_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"プレビューデータが見つかりません: {preview_path}\n"
                "先に /api/analyze (preview_only=true) でプレビュー解析を実行してください。"
            )
        )
    return FileResponse(preview_path, media_type="application/json")


class ConfirmIdsRequest(BaseModel):
    """
    フロントエンドでの初期化確認完了後に送信されるリクエスト。
    研究者が確認済みの track→mouse_id マッピングと本解析の設定を含む。
    """
    video_path: str
    engine: str = "yolov8"
    model_path: Optional[str] = None
    fps: Optional[int] = None
    px_to_mm: float = 0.25
    mice_count: int = 0
    roi_x1: Optional[int] = None
    roi_y1: Optional[int] = None
    roi_x2: Optional[int] = None
    roi_y2: Optional[int] = None
    device: Optional[str] = None
    dlc_superanimal: Optional[str] = "superanimal_topviewmouse"
    dlc_detector: Optional[str] = "fasterrcnn_resnet50_fpn_v2"
    dlc_pose_model: Optional[str] = "hrnet_w32"
    # 研究者確認済みのマッピング: {"track_0": 1, "track_1": 2, ...}
    confirmed_mapping: dict

    model_config = {"protected_namespaces": ()}


@app.post("/api/confirm_ids")
async def confirm_ids_and_analyze(req: ConfirmIdsRequest, background_tasks: BackgroundTasks):
    """
    フロントエンドでの初期化確認完了後に本解析を開始する。
    confirmed_mapping に研究者が割り当てたマッピングが含まれており、
    カラーID自動分類をスキップしてこの値を使用する。
    """
    # 動画ファイルの存在確認
    resolved_path = None
    for candidate in [req.video_path,
                       os.path.basename(req.video_path),
                       os.path.join(os.getcwd(), os.path.basename(req.video_path))]:
        if os.path.exists(candidate):
            resolved_path = candidate
            break

    if resolved_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"動画ファイルが見つかりません: {req.video_path}"
        )

    # AnalysisRequest に変換（confirmed_mapping を JSON 文字列として渡す）
    analysis_req = AnalysisRequest(
        video_path=resolved_path,
        engine=req.engine,
        model_path=req.model_path,
        fps=req.fps,
        px_to_mm=req.px_to_mm,
        mice_count=req.mice_count,
        roi_x1=req.roi_x1,
        roi_y1=req.roi_y1,
        roi_x2=req.roi_x2,
        roi_y2=req.roi_y2,
        dlc_superanimal=req.dlc_superanimal,
        dlc_detector=req.dlc_detector,
        dlc_pose_model=req.dlc_pose_model,
        confirmed_mapping=json.dumps(req.confirmed_mapping, ensure_ascii=False),
    )

    task_id = str(uuid.uuid4())
    TASKS_DB[task_id] = {
        "status": "pending",
        "video": resolved_path,
        "log": (
            "[Server] 確認済みIDマッピングで本解析を開始します。\n"
            f"[Server] マッピング: {req.confirmed_mapping}\n"
        ),
        "warning": "",
        "task_type": "analysis",
    }
    save_tasks_db(TASKS_DB)

    background_tasks.add_task(run_pipeline_task, task_id, analysis_req)

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "確認済みIDマッピングで本解析が開始されました。/api/status/{task_id} で進捗を確認できます。",
    }


if __name__ == "__main__":
    import uvicorn
    # 直接起動時のデフォルトポート 8000
    # コラボおよびlocaltunnel接続に対応するため、ホストを 0.0.0.0、リロードを False に変更
    uvicorn.run("app_server:app", host="0.0.0.0", port=8000, reload=False)

