/**
 * Mouse Scratch Detection System - Main Application Orchestrator (拡張版)
 * ======================================================================
 * Manages playback states, bidirectional HTML5 video synchronization,
 * drag-and-drop keypoint correction with real-time kinematic recalculation,
 * manual mouse ID swapping, and CSV data export download streams.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. State Management ---
    let appState = {
        data: null,
        currentFrame: 0,
        isPlaying: false,
        playbackSpeed: 1.0,
        selectedMouse: 1,
        lastFrameTime: 0,
        rollingWindowSize: 30,
        analysisProgressTimer: null,  // pseudo-progress timer for analysis bar
        pollingInterval: null,        // B-1: Fix polling interval leak
        isRealProgress: false,        // Q-4: Fix pseudo progress bar vs log-extracted progress

        // ── 動的マウス数管理 ─────────────────────────────────────────────────
        // CSV読み込み時に自動検出される。デモデータは ID 1–4
        miceIds: [1, 2, 3, 4],

        // ── Undo/Redo スタック ────────────────────────────────────────────────
        // 操作前のフレームスナップショットを最大 50 件保存する。
        // 各エントリ: { frameIdx: number, mouseId: number, snapshot: miceObj }
        undoStack: [],
        redoStack: [],
        MAX_UNDO_STACK: 50,

        // ── 編集済みフレーム追跡 ─────────────────────────────────────────────
        // 手動補正が行われたフレームのインデックスを記録し、タイムライン上にインジケータを描画する
        editedFrames: new Set(),

        // ── B-3 Bout レビュー状態 ────────────────────────────────────────────
        // index: 現在レビュー中の Bout（時系列ソート後の配列インデックス, -1=未開始）
        // accepted: 承認済み Bout のキー集合。キーは `${mouse_id}:${bout_start_frame}`
        //   （フレーム番号は絶対値で安定するため、他 Bout の編集後も承認状態を保持できる）
        boutReview: { index: -1, accepted: new Set() },

        // ── 動画エクスポート ────────────────────────────────────────────────
        mediaRecorder: null,
        recordedChunks: [],

        // ── 動画解像度キャッシュ（canvas ↔ 動画座標変換用）────────────────────
        // bgVideo.onloadeddata 時に設定される。0 の場合は動画未ロード。
        videoW: 0,
        videoH: 0,

        // CSVデータが正常にロードされているか（デモデータを除く）
        hasRealData: false,

        // バックエンド解析が進行中か（true の場合はデモ座標をcanvasに描画しない）
        isAnalysisRunning: false,

        // ── フェーズ1 初期化確認モード ────────────────────────────────────
        // active: 確認モード中フラグ
        // step: 'bbox' (バウンディングボックス確認) | 'pose' (ポーズ確認) | null
        // pipelineConfig: 本解析開始時に再利用するパイプライン設定
        // previewData: pipeline.py が生成した _preview.json の内容
        // detectedTracks: プレビューから抽出されたトラックID一覧
        // confirmedMapping: 研究者が割り当てた track→mouse_id 辞書（visualizer も参照）
        previewMode: {
            active: false,
            step: null,
            pipelineConfig: null,
            previewData: null,
            videoName: null,
            detectedTracks: [],
        },
        confirmedMapping: {},

        // ── Layer 1: ケージマスク状態 ───────────────────────────────────────
        // maskActive: 編集モード中フラグ
        // maskPolygons: 確定済み多角形のリスト（各要素は [[x,y], [x,y], ...] 動画ピクセル座標）
        // maskInProgress: 編集中（未確定）の多角形の頂点リスト
        // maskVideoName: 現在マスク編集対象の動画名（保存時に使用）
        cageMask: {
            active: false,
            polygons: [],         // 確定済み多角形 (動画ピクセル座標)
            inProgress: [],       // 編集中の頂点列
            videoName: null,
        },
    };
    // visualizer.js から参照させるためグローバル公開
    window.appState = appState;

    // --- 2. Initialize Components ---
    const visualizer = new CageVisualizer('cage-canvas');
    const charts = new DashboardCharts();
    
    // Load programmatic high-fidelity demo data on start
    appState.data = window.generateDemoData();
    
    // Initialize empty/demo signals and analytics charts
    charts.initSignalChart('chart-live-signal');
    charts.updateAnalytics(appState.data.bouts);

    // Fetch threshold parameters from backend server (Q-2)
    fetch('/api/config')
    .then(res => {
        if (!res.ok) throw new Error("Server config endpoint unavailable");
        return res.json();
    })
    .then(cfg => {
        // mm 基準（バックエンドの判定と同じ閾値・パラメータ）
        if (cfg.SCRATCH_MIN_VELOCITY_MM_S      != null) SCRATCH_THRESHOLDS.minVelocityMmS       = cfg.SCRATCH_MIN_VELOCITY_MM_S;
        if (cfg.SCRATCH_MAX_HEAD_DIST_MM       != null) SCRATCH_THRESHOLDS.maxHeadDistMm        = cfg.SCRATCH_MAX_HEAD_DIST_MM;
        if (cfg.SCRATCH_MIN_OSCILLATIONS       != null) SCRATCH_THRESHOLDS.minOscillations      = cfg.SCRATCH_MIN_OSCILLATIONS;
        // 新規 (mm単位、推奨): キャリブレーション独立の振幅閾値
        if (cfg.SCRATCH_MIN_OSC_AMPLITUDE_MM   != null) SCRATCH_THRESHOLDS.minOscAmplitudeMm    = cfg.SCRATCH_MIN_OSC_AMPLITUDE_MM;
        // 旧 (px単位、後方互換)
        if (cfg.SCRATCH_MIN_OSC_AMPLITUDE_PX   != null) SCRATCH_THRESHOLDS.minOscAmplitudePx    = cfg.SCRATCH_MIN_OSC_AMPLITUDE_PX;
        if (cfg.SCRATCH_VELOCITY_SMOOTH_FRAMES != null) SCRATCH_THRESHOLDS.velocitySmoothFrames = cfg.SCRATCH_VELOCITY_SMOOTH_FRAMES;
        if (cfg.SCRATCH_WINDOW_FRAMES          != null) SCRATCH_THRESHOLDS.windowFrames         = cfg.SCRATCH_WINDOW_FRAMES;
        if (cfg.BOUT_MIN_FRAMES                != null) SCRATCH_THRESHOLDS.boutMinFrames         = cfg.BOUT_MIN_FRAMES;
        if (cfg.BOUT_MERGE_GAP                 != null) SCRATCH_THRESHOLDS.boutMergeGap          = cfg.BOUT_MERGE_GAP;
        if (cfg.INTERPOLATION_MAX_GAP          != null) SCRATCH_THRESHOLDS.interpolationMaxGap   = cfg.INTERPOLATION_MAX_GAP;
        if (cfg.PIXEL_TO_MM_RATIO              != null) SCRATCH_THRESHOLDS.pixelToMmRatio       = cfg.PIXEL_TO_MM_RATIO;
        if (cfg.DEFAULT_FPS                    != null) SCRATCH_THRESHOLDS.fps                  = cfg.DEFAULT_FPS;
        // 後方互換: 表示フォールバック用の px 値
        if (cfg.SCRATCH_MIN_VELOCITY)  SCRATCH_THRESHOLDS.minVelocity  = cfg.SCRATCH_MIN_VELOCITY;
        if (cfg.SCRATCH_MAX_HEAD_DIST) SCRATCH_THRESHOLDS.maxHeadDist = cfg.SCRATCH_MAX_HEAD_DIST;
        console.log("[Config] Loaded thresholds from backend:", SCRATCH_THRESHOLDS);

        // 閾値ラベル（span:first-child）をサーバー値で更新する
        // （HTML側は起動時デフォルトを表示しているが、サーバー設定優先）
        const _updLabel = (id, text) => {
            const el = document.getElementById(id);
            if (el) { const s = el.querySelector('span:first-child'); if (s) s.textContent = text; }
        };
        _updLabel('stat-speed-l', `閾値: >${SCRATCH_THRESHOLDS.minVelocityMmS}mm/s`);
        _updLabel('stat-speed-r', `閾値: >${SCRATCH_THRESHOLDS.minVelocityMmS}mm/s`);
        _updLabel('stat-dist-l',  `閾値: <${SCRATCH_THRESHOLDS.maxHeadDistMm}mm`);
        _updLabel('stat-dist-r',  `閾値: <${SCRATCH_THRESHOLDS.maxHeadDistMm}mm`);
        // 閾値エディタ入力欄も最新値に同期
        syncThresholdInputs();
    })
    .catch(err => {
        console.warn("[Config] Failed to fetch server config, using local default thresholds:", err);
    });

    // --- 3. DOM Elements ---
    const btnPlay = document.getElementById('btn-play');
    const playIcon = btnPlay.querySelector('i');
    const btnPrev = document.getElementById('btn-prev');
    const btnNext = document.getElementById('btn-next');
    const timeCurrent = document.getElementById('time-current');
    const timeTotal = document.getElementById('time-total');
    const scrubSlider = document.getElementById('scrub-slider');
    // 動画直下のシークバー（再生コントロールバー内）。scrubSlider と双方向同期する。
    const playbackSeekSlider = document.getElementById('playback-seek-slider');
    const speedSelect = document.getElementById('speed-select');
    
    // File inputs
    const csvFileInput = document.getElementById('csv-file-input');
    const csvDropZone = document.getElementById('csv-drop-zone');
    const videoFileInput = document.getElementById('video-file-input');
    const videoDropZone = document.getElementById('video-drop-zone');
    const bgVideo = document.getElementById('bg-video');
    const rawVideo = document.getElementById('raw-video');  // NEW: left raw panel
    const videoStatusText = document.getElementById('video-status-text');

    // Upload progress bar elements
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const uploadProgressBar = document.getElementById('upload-progress-bar');
    const uploadProgressPct = document.getElementById('upload-progress-pct');
    const uploadProgressLabel = document.getElementById('upload-progress-label');

    // Analysis progress bar elements
    const analysisProgressContainer = document.getElementById('analysis-progress-container');
    const analysisProgressBar = document.getElementById('analysis-progress-bar');
    const analysisProgressPct = document.getElementById('analysis-progress-pct');
    const analysisProgressLabel = document.getElementById('analysis-progress-label');

    // ID Swapper widgets
    const swapMouseA = document.getElementById('swap-mouse-a');
    const swapMouseB = document.getElementById('swap-mouse-b');
    const swapStartFrame = document.getElementById('swap-start-frame');
    const btnSwapIds = document.getElementById('btn-swap-ids');
    const btnExportCsv = document.getElementById('btn-export-csv');
    const btnExportPublication = document.getElementById('btn-export-publication');

    // FastAPI Pipeline integration widgets (NEW)
    const btnRunPipeline = document.getElementById('btn-run-pipeline');
    const pipelineEngine = document.getElementById('pipeline-engine');
    const pipelineCalibration = document.getElementById('pipeline-calibration');
    const pipelineModelPath = document.getElementById('pipeline-model-path');
    const pipelineMiceCount = document.getElementById('pipeline-mice-count');
    const pipelineRoiX1 = document.getElementById('pipeline-roi-x1');
    const pipelineRoiY1 = document.getElementById('pipeline-roi-y1');
    const pipelineRoiX2 = document.getElementById('pipeline-roi-x2');
    const pipelineRoiY2 = document.getElementById('pipeline-roi-y2');
    const pipelineVideoPath = document.getElementById('pipeline-video-path');
    const pipelineStatusPanel = document.getElementById('pipeline-status-panel');
    const mockWarningBanner = document.getElementById('mock-warning-banner');

    // Undo/Redo widgets (NEW)
    const btnUndo = document.getElementById('btn-undo');
    const btnRedo = document.getElementById('btn-redo');
    const invalidMouseId = document.getElementById('invalid-mouse-id');
    const btnToggleInvalid = document.getElementById('btn-toggle-invalid');
    const btnExportVideo = document.getElementById('btn-export-video');

    // Threshold editor widgets (NEW)
    const thrMinVelMms = document.getElementById('thr-min-vel-mms');
    const thrMaxHeadMm = document.getElementById('thr-max-head-mm');
    const thrMinOsc = document.getElementById('thr-min-osc');
    const thrMinOscAmp = document.getElementById('thr-min-osc-amp');
    const btnApplyThresholds = document.getElementById('btn-apply-thresholds');
    
    // Feature Inspector cards
    const valSpeedL = document.getElementById('val-speed-l');
    const valSpeedR = document.getElementById('val-speed-r');
    const valDistL = document.getElementById('val-dist-l');
    const valDistR = document.getElementById('val-dist-r');
    const valOscL = document.getElementById('val-osc-l');
    const valOscR = document.getElementById('val-osc-r');
    
    // Threshold status items
    const statSpeedL = document.getElementById('stat-speed-l');
    const statSpeedR = document.getElementById('stat-speed-r');
    const statDistL = document.getElementById('stat-dist-l');
    const statDistR = document.getElementById('stat-dist-r');
    
    const panelBoutsList = document.getElementById('bouts-list-container');
    const alarmGlowPanel = document.getElementById('visualizer-panel');

    // --- 4. Main Playback Sync Loop ---
    // If background video is loaded, we use the video's natural animation rate and sync the canvas to it.
    // If no video is loaded, we use custom timer intervals.
    
    function playbackTick(timestamp) {
        if (!appState.isPlaying) return;

        const hasVideo = bgVideo && bgVideo.src && bgVideo.readyState >= 2;

        if (hasVideo) {
            // Synchronize frame index to actual video play time
            appState.currentFrame = Math.floor(bgVideo.currentTime * appState.data.fps);
            if (appState.currentFrame >= appState.data.numFrames) {
                appState.currentFrame = 0;
                bgVideo.currentTime = 0;
            }
            // Keep rawVideo in sync
            if (rawVideo && rawVideo.src && rawVideo.readyState >= 2) {
                if (Math.abs(rawVideo.currentTime - bgVideo.currentTime) > 0.1) {
                    rawVideo.currentTime = bgVideo.currentTime;
                }
            }
            updateUIForFrame(false); // Don't scrub video during play to prevent jitter
        } else {
            if (!appState.lastFrameTime) appState.lastFrameTime = timestamp;
            
            const targetFps = appState.data.fps;
            const frameIntervalMs = (1000 / targetFps) / appState.playbackSpeed;
            const elapsed = timestamp - appState.lastFrameTime;

            if (elapsed >= frameIntervalMs) {
                appState.currentFrame = (appState.currentFrame + 1) % appState.data.numFrames;
                updateUIForFrame();
                appState.lastFrameTime = timestamp - (elapsed % frameIntervalMs);
            }
        }

        requestAnimationFrame(playbackTick);
    }

    function togglePlay() {
        appState.isPlaying = !appState.isPlaying;
        const hasVideo = bgVideo && bgVideo.src && bgVideo.readyState >= 2;
        const hasRaw = rawVideo && rawVideo.src && rawVideo.readyState >= 2;

        if (appState.isPlaying) {
            playIcon.setAttribute('data-lucide', 'pause');
            appState.lastFrameTime = 0;
            
            if (hasVideo) {
                bgVideo.playbackRate = appState.playbackSpeed;
                bgVideo.play();
            }
            if (hasRaw) {
                rawVideo.playbackRate = appState.playbackSpeed;
                rawVideo.play();
            }
            requestAnimationFrame(playbackTick);
        } else {
            playIcon.setAttribute('data-lucide', 'play');
            if (hasVideo) bgVideo.pause();
            if (hasRaw) rawVideo.pause();
        }
        lucide.createIcons();
    }

    // --- 5. UI Rendering & Sync Operations ---
    function updateUIForFrame(syncVideo = true) {
        const frameData = appState.data.frames[appState.currentFrame];
        if (!frameData) return;

        // Sync timeline slider
        scrubSlider.value = appState.currentFrame;
        // 動画直下シークバーも同期（範囲・現在値とも scrubSlider に合わせる）
        if (playbackSeekSlider) {
            playbackSeekSlider.max = scrubSlider.max;
            playbackSeekSlider.value = appState.currentFrame;
        }

        // Sync numeric frame inputs
        swapStartFrame.value = appState.currentFrame;

        // Sync time displays
        const currSeconds = appState.currentFrame / appState.data.fps;
        const totalSeconds = appState.data.numFrames / appState.data.fps;
        timeCurrent.textContent = formatTime(currSeconds);
        timeTotal.textContent = `/ ${formatTime(totalSeconds)}`;

        // Bidirectional Video Sync (both panels)
        const hasVideo = bgVideo && bgVideo.src && bgVideo.readyState >= 2;
        if (hasVideo && syncVideo) {
            bgVideo.currentTime = currSeconds;
        }
        // Sync raw video panel
        const hasRaw = rawVideo && rawVideo.src && rawVideo.readyState >= 2;
        if (hasRaw && syncVideo) {
            rawVideo.currentTime = currSeconds;
        }

        // Draw skeletons on canvas
        // 解析実行中で実データ未ロードの場合はデモ座標を描画しない
        // （解析中プレースホルダーは btnRunPipeline の click ハンドラで描画済み）
        if (appState.isAnalysisRunning && !appState.hasRealData) {
            // 何もしない（プレースホルダーを上書きしない）
        } else {
            visualizer.draw(frameData, appState.selectedMouse);
        }

        // Update features cards & signal charts
        updateFeatureInspector(frameData);
        updateRollingSignal();
        checkGlobalScratchAlarms(frameData);
    }

    function updateFeatureInspector(frameData) {
        const mouseData = frameData.mice[appState.selectedMouse];
        if (!mouseData) return;

        const feats = mouseData.features;

        // 物理単位の取得: 平滑化速度を優先 → 非平滑 mm/s → px 換算（比率は SCRATCH_THRESHOLDS から取得）
        const _disp_ratio = SCRATCH_THRESHOLDS.pixelToMmRatio;
        const _disp_fps   = appState.data.fps || SCRATCH_THRESHOLDS.fps;
        function _toMmS(smooth, mms, px) {
            if (smooth != null && !isNaN(smooth)) return smooth.toFixed(1);
            if (mms    != null && !isNaN(mms))    return mms.toFixed(1);
            return (px * _disp_ratio * _disp_fps).toFixed(1);
        }
        function _toMm(mm, px) {
            if (mm != null && !isNaN(mm)) return mm.toFixed(1);
            return (px * _disp_ratio).toFixed(1);
        }
        const speedL_mm = _toMmS(feats.hind_paw_speed_L_mm_s_smooth, feats.hind_paw_speed_L_mm_s, feats.hind_paw_speed_L);
        const speedR_mm = _toMmS(feats.hind_paw_speed_R_mm_s_smooth, feats.hind_paw_speed_R_mm_s, feats.hind_paw_speed_R);
        const distL_mm  = _toMm(feats.hind_to_ear_dist_L_mm, feats.hind_to_ear_dist_L);
        const distR_mm  = _toMm(feats.hind_to_ear_dist_R_mm, feats.hind_to_ear_dist_R);

        valSpeedL.textContent = `${feats.hind_paw_speed_L} px/f (${speedL_mm} mm/s)`;
        valSpeedR.textContent = `${feats.hind_paw_speed_R} px/f (${speedR_mm} mm/s)`;
        valDistL.textContent = `${feats.hind_to_ear_dist_L} px (${distL_mm} mm)`;
        valDistR.textContent = `${feats.hind_to_ear_dist_R} px (${distR_mm} mm)`;
        valOscL.textContent = `${feats.hind_osc_x_L} rev`;
        valOscR.textContent = `${feats.hind_osc_x_R} rev`;

        // Check threshold boundaries (mm 基準 — バックエンドと同一閾値で判定)
        const _ratio = SCRATCH_THRESHOLDS.pixelToMmRatio;
        const _fps   = SCRATCH_THRESHOLDS.fps;
        // 平滑化速度を優先、なければ非平滑 mm/s、なければ px→mm/s 換算
        const spdL_smooth_mm = (feats.hind_paw_speed_L_mm_s_smooth != null && !isNaN(feats.hind_paw_speed_L_mm_s_smooth))
            ? feats.hind_paw_speed_L_mm_s_smooth
            : (feats.hind_paw_speed_L_mm_s != null && !isNaN(feats.hind_paw_speed_L_mm_s))
                ? feats.hind_paw_speed_L_mm_s
                : (feats.hind_paw_speed_L * _ratio * _fps);
        const spdR_smooth_mm = (feats.hind_paw_speed_R_mm_s_smooth != null && !isNaN(feats.hind_paw_speed_R_mm_s_smooth))
            ? feats.hind_paw_speed_R_mm_s_smooth
            : (feats.hind_paw_speed_R_mm_s != null && !isNaN(feats.hind_paw_speed_R_mm_s))
                ? feats.hind_paw_speed_R_mm_s
                : (feats.hind_paw_speed_R * _ratio * _fps);
        const dstL_mm_num = (feats.hind_to_ear_dist_L_mm != null && !isNaN(feats.hind_to_ear_dist_L_mm))
            ? feats.hind_to_ear_dist_L_mm : (feats.hind_to_ear_dist_L * _ratio);
        const dstR_mm_num = (feats.hind_to_ear_dist_R_mm != null && !isNaN(feats.hind_to_ear_dist_R_mm))
            ? feats.hind_to_ear_dist_R_mm : (feats.hind_to_ear_dist_R * _ratio);

        // Left
        if (spdL_smooth_mm > SCRATCH_THRESHOLDS.minVelocityMmS) {
            statSpeedL.className = "threshold-indicator pass";
            statSpeedL.querySelector('span:last-child').textContent = `✓ ${SCRATCH_THRESHOLDS.minVelocityMmS}mm/s 超過`;
        } else {
            statSpeedL.className = "threshold-indicator fail";
            statSpeedL.querySelector('span:last-child').textContent = "✗ 低速";
        }
        if (dstL_mm_num < SCRATCH_THRESHOLDS.maxHeadDistMm) {
            statDistL.className = "threshold-indicator pass";
            statDistL.querySelector('span:last-child').textContent = "✓ 接近中";
        } else {
            statDistL.className = "threshold-indicator fail";
            statDistL.querySelector('span:last-child').textContent = "✗ 遠方";
        }

        // Right
        if (spdR_smooth_mm > SCRATCH_THRESHOLDS.minVelocityMmS) {
            statSpeedR.className = "threshold-indicator pass";
            statSpeedR.querySelector('span:last-child').textContent = `✓ ${SCRATCH_THRESHOLDS.minVelocityMmS}mm/s 超過`;
        } else {
            statSpeedR.className = "threshold-indicator fail";
            statSpeedR.querySelector('span:last-child').textContent = "✗ 低速";
        }
        if (dstR_mm_num < SCRATCH_THRESHOLDS.maxHeadDistMm) {
            statDistR.className = "threshold-indicator pass";
            statDistR.querySelector('span:last-child').textContent = "✓ 接近中";
        } else {
            statDistR.className = "threshold-indicator fail";
            statDistR.querySelector('span:last-child').textContent = "✗ 遠方";
        }
    }

    function updateRollingSignal() {
        const start = Math.max(0, appState.currentFrame - appState.rollingWindowSize + 1);
        const end = appState.currentFrame + 1;
        const slice = appState.data.frames.slice(start, end);
        charts.updateSignalChart(slice, appState.selectedMouse);
    }

    function checkGlobalScratchAlarms(frameData) {
        let anyScratching = false;
        for (const [id, mData] of Object.entries(frameData.mice)) {
            const mId = parseInt(id);
            const selectorBtn = document.getElementById(`mouse-btn-${mId}`);
            // disabled な個体はひっかき判定対象外（ボタンクラスも更新しない）
            if (mData.disabled) {
                if (selectorBtn) selectorBtn.classList.remove('scratching');
                continue;
            }
            if (mData.scratch_flag === 1) {
                anyScratching = true;
                if (selectorBtn) selectorBtn.classList.add('scratching');
            } else {
                if (selectorBtn) selectorBtn.classList.remove('scratching');
            }
        }

        if (anyScratching) {
            alarmGlowPanel.classList.add('active-scratching-glow');
        } else {
            alarmGlowPanel.classList.remove('active-scratching-glow');
        }
    }

    function formatTime(sec) {
        const minutes = Math.floor(sec / 60);
        const seconds = Math.floor(sec % 60);
        const centiseconds = Math.floor((sec % 1) * 100);
        return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(centiseconds).padStart(2, '0')}`;
    }

    // --- 6. Visualizer Drag & Drop Callbacks ---
    // Handle interactive posture corrections by the user
    visualizer.onNodeDrag = (mouseId, kpName, newX, newY) => {
        // Automatically pause on drag start for stable adjustment
        if (appState.isPlaying) togglePlay();

        // Undo 履歴: ドラッグ開始時のみスナップショットを保存（連続イベント中は重複保存しない）
        if (!appState._dragSnapshotTaken) {
            pushUndoSnapshot(appState.currentFrame, mouseId);
            appState._dragSnapshotTaken = true;
        }

        const frame = appState.data.frames[appState.currentFrame];
        if (!frame || !frame.mice[mouseId]) return;

        // Update joint coordinate
        frame.mice[mouseId].keypoints[kpName].x = newX;
        frame.mice[mouseId].keypoints[kpName].y = newY;

        // Recalculate kinematics for this frame
        recalculateFrameKinematics(appState.currentFrame, mouseId);
        
        // Redraw current frame
        updateUIForFrame(false);
    };

    visualizer.onNodeDragEnd = () => {
        // ドラッグ終了 → 次の操作で再びスナップショットが取れるようフラグをリセット
        appState._dragSnapshotTaken = false;
        // Recalculate rolling velocities & check if rule-based classification shifts
        const mId = appState.selectedMouse;
        
        // Re-evaluate a small window +/- 15 frames around the edit point to update local velocities smoothly
        const start = Math.max(1, appState.currentFrame - 15);
        const end = Math.min(appState.data.numFrames - 1, appState.currentFrame + 15);
        
        for (let f = start; f <= end; f++) {
            recalculateFrameKinematics(f, mId);
        }

        // Re-detect scratch bouts dynamically based on new flags!
        autoGenerateBoutsFromFlags();
        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);
        updateUIForFrame(false);
    };

    /**
     * Biomechanic Recalculation Engine — バックエンド feature_extraction.py と同一ロジック (mm 基準)
     *
     * 変更点（旧 px 基準からの差分）:
     *   [1] 速度を mm/s に換算し minVelocityMmS と比較
     *   [4] velocitySmoothFrames フレームの移動平均平滑化を適用
     *   [5] 振動数カウントに minOscAmplitudePx の最小振幅フィルタを導入（x・y 両軸）
     *   [3] scratch_prob を 3 条件マージン比の連続スコア [0,1] で算出
     *   [1] 耳距離を mm に換算し maxHeadDistMm と比較
     */
    function recalculateFrameKinematics(fIdx, mouseId) {
        const frame = appState.data.frames[fIdx];
        const prevFrame = appState.data.frames[fIdx - 1];
        if (!frame || !frame.mice[mouseId]) return;

        const mData = frame.mice[mouseId];
        const kps = mData.keypoints;

        const ratio = SCRATCH_THRESHOLDS.pixelToMmRatio;  // mm/px
        // FPS: 実動画から取得された値を優先（バックエンドと同じ動作）
        const fps   = (appState.data && appState.data.fps) || SCRATCH_THRESHOLDS.fps;

        const headX = kps["head_center"].x;
        const headY = kps["head_center"].y;
        const earLX = kps["left_ear"].x;
        const earLY = kps["left_ear"].y;
        const earRX = kps["right_ear"].x;
        const earRY = kps["right_ear"].y;

        // ── A. 距離特徴量 (px) ──────────────────────────────────────────────
        function dist2d(x1, y1, x2, y2) { return Math.sqrt((x1-x2)**2 + (y1-y2)**2); }

        mData.features.body_length_px      = Math.round(dist2d(kps["neck"].x, kps["neck"].y, kps["tail_base"].x, kps["tail_base"].y));
        mData.features.head_to_nose_dist   = Math.round(dist2d(kps["nose"].x, kps["nose"].y, headX, headY));
        mData.features.hind_to_head_dist_L = Math.round(dist2d(kps["left_hind_paw"].x,  kps["left_hind_paw"].y,  headX, headY));
        mData.features.hind_to_head_dist_R = Math.round(dist2d(kps["right_hind_paw"].x, kps["right_hind_paw"].y, headX, headY));
        mData.features.hind_to_ear_dist_L  = Math.round(dist2d(kps["left_hind_paw"].x,  kps["left_hind_paw"].y,  earLX, earLY));
        mData.features.hind_to_ear_dist_R  = Math.round(dist2d(kps["right_hind_paw"].x, kps["right_hind_paw"].y, earRX, earRY));

        // ── B. mm 単位距離（[1] スクラッチ判定用）──────────────────────────
        mData.features.hind_to_ear_dist_L_mm  = mData.features.hind_to_ear_dist_L  * ratio;
        mData.features.hind_to_ear_dist_R_mm  = mData.features.hind_to_ear_dist_R  * ratio;
        // 顔面・頭頂掻きも検出するため head_center 距離も mm 化
        mData.features.hind_to_head_dist_L_mm = mData.features.hind_to_head_dist_L * ratio;
        mData.features.hind_to_head_dist_R_mm = mData.features.hind_to_head_dist_R * ratio;

        // ── C. フレーム間速度（px/frame）────────────────────────────────────
        if (prevFrame && prevFrame.mice[mouseId]) {
            const prevKps = prevFrame.mice[mouseId].keypoints;
            const dxL = kps["left_hind_paw"].x - prevKps["left_hind_paw"].x;
            const dyL = kps["left_hind_paw"].y - prevKps["left_hind_paw"].y;
            mData.features.hind_paw_speed_L       = Math.round(Math.sqrt(dxL*dxL + dyL*dyL));
            mData.features.hind_paw_speed_L_mm_s  = mData.features.hind_paw_speed_L * ratio * fps;

            const dxR = kps["right_hind_paw"].x - prevKps["right_hind_paw"].x;
            const dyR = kps["right_hind_paw"].y - prevKps["right_hind_paw"].y;
            mData.features.hind_paw_speed_R       = Math.round(Math.sqrt(dxR*dxR + dyR*dyR));
            mData.features.hind_paw_speed_R_mm_s  = mData.features.hind_paw_speed_R * ratio * fps;
        }

        // ── D. 速度スムージング（[4] 移動平均 velocitySmoothFrames フレーム）─
        const smoothN = SCRATCH_THRESHOLDS.velocitySmoothFrames;
        const velBufL = [], velBufR = [];
        for (let i = Math.max(1, fIdx - smoothN + 1); i <= fIdx; i++) {
            const cF = appState.data.frames[i];
            const pF = appState.data.frames[i - 1];
            if (cF && pF && cF.mice[mouseId] && pF.mice[mouseId]) {
                const ck = cF.mice[mouseId].keypoints;
                const pk = pF.mice[mouseId].keypoints;
                const dxL = ck["left_hind_paw"].x - pk["left_hind_paw"].x;
                const dyL = ck["left_hind_paw"].y - pk["left_hind_paw"].y;
                velBufL.push(Math.sqrt(dxL*dxL + dyL*dyL) * ratio * fps);
                const dxR = ck["right_hind_paw"].x - pk["right_hind_paw"].x;
                const dyR = ck["right_hind_paw"].y - pk["right_hind_paw"].y;
                velBufR.push(Math.sqrt(dxR*dxR + dyR*dyR) * ratio * fps);
            }
        }
        const smoothSpeedL = velBufL.length > 0 ? velBufL.reduce((a, b) => a + b, 0) / velBufL.length : 0;
        const smoothSpeedR = velBufR.length > 0 ? velBufR.reduce((a, b) => a + b, 0) / velBufR.length : 0;
        mData.features.hind_paw_speed_L_mm_s_smooth = smoothSpeedL;
        mData.features.hind_paw_speed_R_mm_s_smooth = smoothSpeedR;

        // ── E. 振動数（[5] 最小振幅フィルタ付き・x/y 両軸）────────────────
        // バックエンドと同期: 振幅閾値は mm 単位で保持し、px に換算してから比較する。
        // これによりキャリブレーション値が変わっても生物学的に等価な閾値が適用される。
        const windowFrames = SCRATCH_THRESHOLDS.windowFrames;
        const _ratioForAmp = ratio > 0 ? ratio : 0.25;
        const minAmp = (SCRATCH_THRESHOLDS.minOscAmplitudeMm != null)
            ? SCRATCH_THRESHOLDS.minOscAmplitudeMm / _ratioForAmp
            : SCRATCH_THRESHOLDS.minOscAmplitudePx;
        const startF = Math.max(1, fIdx - windowFrames + 1);

        function countOsc(pawKey, axis) {
            let count = 0, lastD = 0;
            for (let i = startF; i <= fIdx; i++) {
                const cF = appState.data.frames[i];
                const pF = appState.data.frames[i - 1];
                if (!cF || !pF || !cF.mice[mouseId] || !pF.mice[mouseId]) continue;
                const d = cF.mice[mouseId].keypoints[pawKey][axis] - pF.mice[mouseId].keypoints[pawKey][axis];
                if (Math.abs(d) < minAmp) continue;  // ジッター無視（[5]）
                if (lastD !== 0 && ((d > 0 && lastD < 0) || (d < 0 && lastD > 0))) count++;
                lastD = d;
            }
            return count;
        }

        const oscXL = countOsc("left_hind_paw",  "x");
        const oscYL = countOsc("left_hind_paw",  "y");
        const oscXR = countOsc("right_hind_paw", "x");
        const oscYR = countOsc("right_hind_paw", "y");
        const oscL  = Math.max(oscXL, oscYL);  // バックエンドと同様 max(x, y)
        const oscR  = Math.max(oscXR, oscYR);

        mData.features.hind_osc_x_L = oscXL;
        mData.features.hind_osc_y_L = oscYL;
        mData.features.hind_osc_x_R = oscXR;
        mData.features.hind_osc_y_R = oscYR;

        // ── F. スクラッチ判定（[1][3] mm 基準・連続スコア）─────────────────
        // バックエンドの _compute_scratch_prob() と同一ロジック
        function computeScratchProb(speedMmS, distMm, osc) {
            const vT = SCRATCH_THRESHOLDS.minVelocityMmS;
            const dT = SCRATCH_THRESHOLDS.maxHeadDistMm;
            const oT = SCRATCH_THRESHOLDS.minOscillations;
            if (speedMmS <= vT || distMm >= dT || osc < oT) return 0.0;
            const sSpeed = Math.min(1.0, (speedMmS - vT) / vT);
            const sDist  = Math.min(1.0, (dT - distMm) / dT);
            const sOsc   = Math.min(1.0, (osc - oT) / Math.max(oT, 1));
            return Math.min(1.0, (sSpeed + sDist + sOsc) / 3.0);
        }

        // バックエンドと同期: ear と head_center の両方からの最近接距離を採用。
        // マウスは耳・顔・首など頭部全体を掻くため、min(ear, head) が掻き行動の指標として
        // より生物学的に正確。
        function _safeMin(a, b) {
            if (a == null || isNaN(a)) return b;
            if (b == null || isNaN(b)) return a;
            return Math.min(a, b);
        }
        const headDistLmm = _safeMin(mData.features.hind_to_ear_dist_L_mm, mData.features.hind_to_head_dist_L_mm);
        const headDistRmm = _safeMin(mData.features.hind_to_ear_dist_R_mm, mData.features.hind_to_head_dist_R_mm);

        const probL = computeScratchProb(smoothSpeedL, headDistLmm, oscL);
        const probR = computeScratchProb(smoothSpeedR, headDistRmm, oscR);
        const prob  = Math.max(probL, probR);

        mData.scratch_flag = prob > 0 ? 1 : 0;
        mData.scratch_prob = prob;
    }

    // --- 7. Individual ID Swapper (混同修正機能) ---
    btnSwapIds.addEventListener('click', () => {
        const mouseA = parseInt(swapMouseA.value);
        const mouseB = parseInt(swapMouseB.value);
        const startF = parseInt(swapStartFrame.value);

        if (mouseA === mouseB) {
            alert("⚠️ 異なるマウスを選択してください。");
            return;
        }

        const confirmSwap = confirm(`Mouse ${mouseA} と Mouse ${mouseB} の個別IDを\nフレーム ${startF} 以降から動画末尾まで入れ替えます。よろしいですか？`);
        if (!confirmSwap) return;

        // Undo: 起点フレームの両マウスを保存（全フレームを保存するとメモリ過大）
        pushUndoSnapshot(startF, mouseA);
        pushUndoSnapshot(startF, mouseB);

        // Perform Swap starting at startF to EOF
        for (let f = startF; f < appState.data.numFrames; f++) {
            const frame = appState.data.frames[f];
            if (frame && frame.mice[mouseA] && frame.mice[mouseB]) {
                const temp = frame.mice[mouseA];
                frame.mice[mouseA] = frame.mice[mouseB];
                frame.mice[mouseB] = temp;
                // [A-5] スワップで変更された全フレーム×両個体を補正済みとして記録
                markFrameMouseCorrected(f, mouseA);
                markFrameMouseCorrected(f, mouseB);
            }
        }

        // Re-generate bouts summary based on the new identity maps
        autoGenerateBoutsFromFlags();
        
        // Redraw GUI elements
        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);
        updateUIForFrame();

        alert(`✓ IDの入れ替えが完了しました!\n（フレーム ${startF}〜${appState.data.numFrames - 1} に適用）`);
    });

    // --- 8. Export CSV Handler ---
    btnExportCsv.addEventListener('click', () => {
        try {
            const csv = exportCorrectedCSV();
            const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement("a");
            
            const timestamp = new Date().toISOString().slice(0, 10);
            link.href = URL.createObjectURL(blob);
            link.setAttribute("download", `corrected_tracks_${timestamp}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            appState.exported = true;  // [B-1] 出力ステップ完了
            updateStepGuide();
            alert("✓ 補正後データのCSV出力を完了しました！");
        } catch (e) {
            console.error(e);
            alert("⚠️ CSVエクスポートに失敗しました。");
        }
    });

    // --- 8b. [B-5] 出版用エクスポート Handler ---
    if (btnExportPublication) {
        btnExportPublication.addEventListener('click', () => {
            if (!appState.data) {
                alert("⚠️ 解析データが読み込まれていません。");
                return;
            }
            try {
                const timestamp = new Date().toISOString().slice(0, 10);

                // ① 個体別統計 CSV
                triggerBlobDownload(
                    new Blob([exportPerMouseStatsCSV()], { type: 'text/csv;charset=utf-8;' }),
                    `scratch_summary_stats_${timestamp}.csv`
                );

                // ② 来歴 manifest JSON
                triggerBlobDownload(
                    new Blob([JSON.stringify(buildPublicationManifest(), null, 2)], { type: 'application/json;charset=utf-8;' }),
                    `publication_manifest_${timestamp}.json`
                );

                // ③ グラフ図 PNG（Bout数 / 累積時間）
                let figCount = 0;
                if (exportChartFigurePng(charts.boutCountChart, `fig_bout_counts_${timestamp}.png`)) figCount++;
                if (exportChartFigurePng(charts.boutDurationChart, `fig_bout_durations_${timestamp}.png`)) figCount++;

                appState.exported = true;  // [B-1] 出力ステップ完了
                updateStepGuide();
                alert(`✓ 出版用エクスポートを完了しました！\n・個体別統計 CSV\n・来歴 manifest JSON\n・グラフ図 PNG ${figCount} 枚`);
            } catch (e) {
                console.error(e);
                alert("⚠️ 出版用エクスポートに失敗しました。");
            }
        });
    }

    function exportCorrectedCSV() {
        // バックエンドの *_frame_labels.csv 出力スキーマと一致させる
        // （evaluate.py に直接流せる形式 + 学術論文用の mm 物理単位特徴量を含む）
        // [A-5] is_manually_corrected: この行（フレーム×個体）が手動補正されたか (1/0)。
        // 自動パイプライン出力（プレビュー）と研究者確定値を区別する来歴フラグ。
        const headers = ["frame_idx", "track", "mouse_id", "is_scratch", "scratch_flag", "scratch_prob", "is_manually_corrected"];

        // Add keypoint columns (x, y) — conf は手動補正後は 1.0 とみなす
        KEYPOINTS_LIST.forEach(kp => {
            headers.push(`${kp}_x`, `${kp}_y`, `${kp}_conf`);
        });

        // px 系特徴量
        const featKeysPx = [
            "body_length_px", "head_to_nose_dist",
            "hind_to_head_dist_L", "hind_to_head_dist_R",
            "hind_to_ear_dist_L", "hind_to_ear_dist_R",
            "hind_paw_speed_L", "hind_paw_speed_R",
            "hind_osc_x_L", "hind_osc_y_L",
            "hind_osc_x_R", "hind_osc_y_R"
        ];
        // mm 系特徴量（バックエンドの再評価互換のため必須）
        const featKeysMm = [
            "body_length_mm", "head_to_nose_dist_mm",
            "hind_to_head_dist_L_mm", "hind_to_head_dist_R_mm",
            "hind_to_ear_dist_L_mm", "hind_to_ear_dist_R_mm",
            "hind_paw_speed_L_mm_s", "hind_paw_speed_R_mm_s",
            "hind_paw_speed_L_mm_s_smooth", "hind_paw_speed_R_mm_s_smooth"
        ];
        featKeysPx.forEach(fk => headers.push(fk));
        featKeysMm.forEach(fk => headers.push(fk));

        const rows = [headers.join(",")];

        // 数値フォーマット（mm 単位は小数 3 桁、px は整数）
        const fmt = (v, decimals) => {
            if (v === undefined || v === null || (typeof v === "number" && isNaN(v))) return "";
            return decimals != null ? Number(v).toFixed(decimals) : v;
        };

        for (let f = 0; f < appState.data.numFrames; f++) {
            const frame = appState.data.frames[f];

            // 動的マウス数対応: appState.miceIds を使用
            for (const mId of appState.miceIds) {
                const mData = frame.mice[mId];
                if (!mData) continue;

                const flag = mData.scratch_flag | 0;
                const cols = [
                    f,
                    `mouse_${mId}`,
                    mId,
                    flag,                          // is_scratch (evaluate.py 互換)
                    flag,                          // scratch_flag (パイプライン互換)
                    fmt(mData.scratch_prob, 3),
                    mData._corrected ? 1 : 0       // [A-5] is_manually_corrected
                ];

                // Coordinates
                KEYPOINTS_LIST.forEach(kp => {
                    const node = mData.keypoints[kp];
                    cols.push(node ? fmt(node.x) : "",
                              node ? fmt(node.y) : "",
                              node && node.conf != null ? fmt(node.conf, 3) : "1.000");
                });

                // px 特徴量
                featKeysPx.forEach(fk => cols.push(fmt(mData.features[fk])));
                // mm 特徴量
                featKeysMm.forEach(fk => cols.push(fmt(mData.features[fk], 3)));

                rows.push(cols.join(","));
            }
        }

        return rows.join("\n");
    }

    // ===================================================================
    // [B-5] 出版用エクスポート（個体別統計 + 来歴 manifest + グラフ図）
    // -------------------------------------------------------------------
    // 補正後 CSV（フレーム×個体の生データ）とは別に、論文の Results/Methods に
    // そのまま転記できる「集計済み」成果物を一括で書き出す。
    //   ① 個体別ひっかき統計 CSV（Bout数・総時間・平均/中央値/範囲・頻度・%時間）
    //   ② 来歴 manifest JSON（閾値・校正値・補正状況・承認 Bout 数・日時）
    //   ③ グラフ図 PNG（Bout数棒グラフ・累積時間ドーナツ）
    // ===================================================================

    /**
     * 数値配列の中央値を返す。空配列は null。
     * @param {number[]} arr
     * @returns {number|null}
     */
    /**
     * キーポイント辞書 {kp: {x,y,conf}} の高速ディープコピー。
     * 欠損フレーム補完ループで数十万回呼ばれるため、低速な
     * JSON.parse(JSON.stringify()) を避けて手動コピーする（ブラウザの応答停止を防止）。
     * x/y/conf のみを持つ既知構造なので、これで完全に独立したオブジェクトになる
     * （後段の手動ドラッグ補正がフレーム間で干渉しない）。
     */
    function _cloneKeypoints(kps) {
        const out = {};
        for (const k in kps) {
            const p = kps[k];
            out[k] = { x: p.x, y: p.y, conf: p.conf };
        }
        return out;
    }

    function _median(arr) {
        if (!arr || arr.length === 0) return null;
        const s = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(s.length / 2);
        return s.length % 2 !== 0 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    }

    /**
     * 個体別のひっかき行動統計を集計する。
     * Akiyama 2010 等の標準的な scratch 指標（bout 数・総時間・頻度）に準拠。
     * 記録時間は numFrames / fps（秒）で算出する。
     * @returns {Array<Object>} 個体ごとの統計オブジェクト配列
     */
    function computePerMouseStats() {
        const fps = (appState.data && appState.data.fps) || SCRATCH_THRESHOLDS.fps || 30;
        const numFrames = (appState.data && appState.data.numFrames) || 0;
        const recordingSec = fps > 0 ? numFrames / fps : 0;
        const recordingMin = recordingSec / 60;
        const bouts = (appState.data && appState.data.bouts) || [];
        const accepted = appState.boutReview ? appState.boutReview.accepted : new Set();

        return appState.miceIds.map(mId => {
            const mBouts = bouts.filter(b => b.mouse_id === mId);
            const durations = mBouts.map(b => b.bout_duration_s);
            const totalDur = durations.reduce((a, b) => a + b, 0);
            const totalFrames = mBouts.reduce((a, b) => a + (b.bout_n_frames || 0), 0);
            const nAccepted = mBouts.filter(b => accepted && accepted.has(boutKey(b))).length;

            return {
                mouse_id: mId,
                n_bouts: mBouts.length,
                n_accepted_bouts: nAccepted,
                total_scratch_duration_s: totalDur,
                total_scratch_frames: totalFrames,
                mean_bout_duration_s: mBouts.length ? totalDur / mBouts.length : 0,
                median_bout_duration_s: _median(durations) || 0,
                min_bout_duration_s: durations.length ? Math.min(...durations) : 0,
                max_bout_duration_s: durations.length ? Math.max(...durations) : 0,
                // 頻度: 1分あたりの bout 数（標準的な scratch 指標）
                bouts_per_min: recordingMin > 0 ? mBouts.length / recordingMin : 0,
                // ひっかきに費やした時間の割合（記録全体に対する %）
                pct_time_scratching: recordingSec > 0 ? (totalDur / recordingSec) * 100 : 0
            };
        });
    }

    /**
     * 個体別統計を CSV 文字列に整形する（mm 系特徴量同様に時間は小数 3 桁）。
     * @returns {string}
     */
    function exportPerMouseStatsCSV() {
        const headers = [
            "mouse_id", "n_bouts", "n_accepted_bouts",
            "total_scratch_duration_s", "total_scratch_frames",
            "mean_bout_duration_s", "median_bout_duration_s",
            "min_bout_duration_s", "max_bout_duration_s",
            "bouts_per_min", "pct_time_scratching"
        ];
        const rows = [headers.join(",")];
        const f3 = v => Number(v).toFixed(3);

        computePerMouseStats().forEach(s => {
            rows.push([
                s.mouse_id, s.n_bouts, s.n_accepted_bouts,
                f3(s.total_scratch_duration_s), s.total_scratch_frames,
                f3(s.mean_bout_duration_s), f3(s.median_bout_duration_s),
                f3(s.min_bout_duration_s), f3(s.max_bout_duration_s),
                f3(s.bouts_per_min), f3(s.pct_time_scratching)
            ].join(","));
        });
        return rows.join("\n");
    }

    /**
     * GUI 現在状態の来歴 manifest を構築する。
     * pipeline.py の write_provenance_manifest と同じ思想で、補正後の成果物が
     * 「どの閾値・校正値・補正状況で確定したか」を 1 ファイルに残す。
     * （バックエンド manifest はパイプライン実行時、こちらは手動補正確定時の来歴）
     * @returns {Object}
     */
    function buildPublicationManifest() {
        const fps = (appState.data && appState.data.fps) || SCRATCH_THRESHOLDS.fps || 30;
        const numFrames = (appState.data && appState.data.numFrames) || 0;
        const isDefaultCalib = Math.abs(SCRATCH_THRESHOLDS.pixelToMmRatio - 0.25) < 1e-9;

        return {
            schema: "mouse-scratch-publication/v1",
            generated_at: new Date().toISOString(),
            source: appState.hasRealData ? "csv_or_pipeline" : "demo_data",
            recording: {
                num_frames: numFrames,
                fps: fps,
                duration_s: fps > 0 ? numFrames / fps : 0,
                mice_ids: [...appState.miceIds],
                mice_count: appState.miceIds.length
            },
            calibration: {
                pixel_to_mm_ratio: SCRATCH_THRESHOLDS.pixelToMmRatio,
                pixel_to_mm_is_default: isDefaultCalib
            },
            scratch_thresholds: {
                SCRATCH_MIN_VELOCITY_MM_S:     SCRATCH_THRESHOLDS.minVelocityMmS,
                SCRATCH_MAX_HEAD_DIST_MM:      SCRATCH_THRESHOLDS.maxHeadDistMm,
                SCRATCH_MIN_OSCILLATIONS:      SCRATCH_THRESHOLDS.minOscillations,
                SCRATCH_MIN_OSC_AMPLITUDE_MM:  SCRATCH_THRESHOLDS.minOscAmplitudeMm,
                SCRATCH_VELOCITY_SMOOTH_FRAMES: SCRATCH_THRESHOLDS.velocitySmoothFrames,
                SCRATCH_WINDOW_FRAMES:         SCRATCH_THRESHOLDS.windowFrames,
                BOUT_MIN_FRAMES:               SCRATCH_THRESHOLDS.boutMinFrames,
                BOUT_MERGE_GAP:                SCRATCH_THRESHOLDS.boutMergeGap,
                INTERPOLATION_MAX_GAP:         SCRATCH_THRESHOLDS.interpolationMaxGap
            },
            // [A-5] 手動補正の来歴: 補正済みフレーム数と承認 Bout 数を記録
            manual_correction: {
                edited_frame_count: appState.editedFrames ? appState.editedFrames.size : 0,
                accepted_bout_count: appState.boutReview ? appState.boutReview.accepted.size : 0,
                total_bout_count: (appState.data && appState.data.bouts) ? appState.data.bouts.length : 0
            },
            per_mouse_statistics: computePerMouseStats()
        };
    }

    /**
     * Blob をダウンロードさせる共通ヘルパー。
     * @param {Blob} blob
     * @param {string} filename
     */
    function triggerBlobDownload(blob, filename) {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.setAttribute("download", filename);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(link.href);
    }

    /**
     * Chart.js インスタンスの図を PNG として書き出す。
     * @param {Object} chart - Chart.js インスタンス
     * @param {string} filename
     * @returns {boolean} 書き出せたか
     */
    function exportChartFigurePng(chart, filename) {
        if (!chart || typeof chart.toBase64Image !== "function") return false;
        const dataUrl = chart.toBase64Image("image/png", 1.0);
        if (!dataUrl || dataUrl.length < 32) return false;
        // dataURL → Blob 変換
        const byteStr = atob(dataUrl.split(",")[1]);
        const len = byteStr.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) bytes[i] = byteStr.charCodeAt(i);
        triggerBlobDownload(new Blob([bytes], { type: "image/png" }), filename);
        return true;
    }

    // --- 9. Video Drop & Load Binding ---
    videoDropZone.addEventListener('click', () => videoFileInput.click());
    
    videoDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        videoDropZone.classList.add('drag-over');
    });

    videoDropZone.addEventListener('dragleave', () => {
        videoDropZone.classList.remove('drag-over');
    });

    videoDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        videoDropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            loadBackgroundVideo(e.dataTransfer.files[0]);
        }
    });

    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            loadBackgroundVideo(e.target.files[0]);
        }
    });

    // --- Upload Progress Bar Helpers ---
    function showUploadProgress(fileName) {
        uploadProgressContainer.style.display = 'block';
        uploadProgressBar.style.width = '0%';
        uploadProgressPct.textContent = '0%';
        uploadProgressLabel.textContent = `読み込み中: ${fileName}`;
        uploadProgressBar.classList.add('running');
    }

    function updateUploadProgress(loaded, total, fileName) {
        const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
        uploadProgressBar.style.width = `${pct}%`;
        uploadProgressPct.textContent = `${pct}%`;
        const sizeMB = (total / 1024 / 1024).toFixed(1);
        uploadProgressLabel.textContent = `${fileName} (${sizeMB} MB)`;
    }

    function completeUploadProgress(fileName) {
        uploadProgressBar.style.width = '100%';
        uploadProgressPct.textContent = '100%';
        uploadProgressBar.classList.remove('running');
        uploadProgressLabel.textContent = `✓ ${fileName} 読み込み完了`;
        uploadProgressLabel.style.color = '#10B981';
    }

    function loadBackgroundVideo(file) {
        // MIMEタイプまたは拡張子でビデオファイルを検証する
        const videoMimeTypes = [
            'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/avi',
            'video/mpeg', 'video/ogg', 'video/webm', 'video/x-matroska', 'video/x-ms-wmv'
        ];
        const videoExtensions = ['.mp4', '.mov', '.avi', '.mpeg', '.mpg', '.ogg', '.webm', '.mkv', '.wmv', '.m4v'];
        const fileName = file.name.toLowerCase();
        const fileExt = '.' + fileName.split('.').pop();
        
        const isVideoByMime = file.type.startsWith('video/') || videoMimeTypes.includes(file.type);
        const isVideoByExt = videoExtensions.includes(fileExt);
        
        if (!isVideoByMime && !isVideoByExt) {
            alert("⚠️ 動画ファイル（.mp4, .mov, .avi など）を選択してください。");
            return;
        }
        
        if (fileExt === '.mov' || file.type === 'video/quicktime') {
            console.log('[Video] MOVファイルを読み込み中: ', file.name, '| MIME:', file.type);
        }

        // アップロード進捗バーを表示（読み込み中）
        // ローカルファイルは URL.createObjectURL() で即時利用可能なため
        // FileReader.readAsArrayBuffer()（2GB超で進捗イベントが発火しない・全RAM消費）は使用しない。
        // 代わりに video.onloadeddata 完了時に 100% へ切り替える。
        showUploadProgress(file.name);
        const sizeMB = (file.size / 1024 / 1024).toFixed(1);
        uploadProgressLabel.textContent = `読み込み中: ${file.name} (${sizeMB} MB)`;
        // 動画デコード中を示す暫定値として 50% へアニメーション
        // requestAnimationFrame で1フレーム遅らせて 0%→50% のCSS遷移を確実に発生させる
        requestAnimationFrame(() => {
            uploadProgressBar.style.transition = 'width 0.8s ease';
            uploadProgressBar.style.width = '50%';
            uploadProgressPct.textContent = '50%';
        });

        const url = URL.createObjectURL(file);
        
        // Load into both video panels
        bgVideo.src = url;
        bgVideo.load();
        // Also load into rawVideo (left panel) with same Blob URL
        rawVideo.src = url;
        rawVideo.load();

        // Hide placeholders
        const rawPlaceholder = document.getElementById('raw-video-placeholder');
        const overlayPlaceholder = document.getElementById('overlay-video-placeholder');
        if (rawPlaceholder) rawPlaceholder.classList.add('hidden-placeholder');
        if (overlayPlaceholder) overlayPlaceholder.classList.add('hidden-placeholder');
        
        bgVideo.onloadeddata = () => {
            // 動画デコード完了 → アップロード進捗バーを 100% に
            completeUploadProgress(file.name);

            appState.isPlaying = false;
            playIcon.setAttribute('data-lucide', 'play');
            lucide.createIcons();

            // Set times
            bgVideo.currentTime = 0;
            appState.currentFrame = 0;

            const durationSec = bgVideo.duration;
            timeTotal.textContent = `/ ${formatTime(durationSec)}`;

            videoStatusText.textContent = `${file.name} 同期完了`;
            videoStatusText.style.color = "#A855F7";
            
            // FastAPI自動解析ボタンの活性化と動画名のバインド
            btnRunPipeline.disabled = false;
            btnRunPipeline.textContent = "解析パイプラインを実行";
            btnRunPipeline.style.background = "rgba(168, 85, 247, 0.15)";
            btnRunPipeline.style.borderColor = "#A855F7";
            btnRunPipeline.dataset.videoName = file.name;

            // 新設された入力欄にファイル名を初期値として設定
            const inputVideoPath = document.getElementById('pipeline-video-path');
            if (inputVideoPath) {
                inputVideoPath.value = file.name;
            }
            
            // Adjust slider max to match video frames if FPS is known (assume 30)
            const videoFramesCount = Math.floor(durationSec * appState.data.fps);
            if (videoFramesCount > 0 && Math.abs(videoFramesCount - appState.data.numFrames) > 5) {
                // If loaded video duration differs significantly, scale our frame array
                adjustFrameDuration(videoFramesCount);
            }

            // 動画の実解像度を記録（キーポイント座標の canvas 座標変換に使用）
            appState.videoW = bgVideo.videoWidth;
            appState.videoH = bgVideo.videoHeight;
            if (typeof updateStepGuide === 'function') updateStepGuide();  // [B-1]

            // CSVデータがロード済みであれば canvas を動画の描画領域にフィット
            if (appState.hasRealData) {
                resizeCanvasToVideo();
            }

            updateUIForFrame();
            alert(`✓ 動画の同期に成功しました！\nファイル: ${file.name}\n形式: ${file.type || fileExt}\n長さ: ${durationSec.toFixed(2)} 秒`);
        };
        
        // MOVファイルがブラウザで再生できない場合のエラーハンドリング
        bgVideo.onerror = (e) => {
            const errCode = bgVideo.error ? bgVideo.error.code : '?';
            const errMessages = {
                1: 'ユーザーによる再生中断',
                2: 'ネットワークエラー',
                3: 'デコードエラー: ファイルが破損しているか、コーデック非対応',
                4: 'ファイルが見つからないか、ファイル形式非対応'
            };
            const errMsg = errMessages[errCode] || '不明なエラー';
            
            console.error('[Video] 再生エラー:', errCode, errMsg, file.name);
            
            if (fileExt === '.mov' || file.type === 'video/quicktime') {
                alert(`⚠️ MOVファイルの再生に失敗しました。\nエラー: ${errMsg}\n\nMOVファイルはブラウザによってコーデック互換性が異なります。\n\n解決策:\n1. ブラウザを Safari または Edge に変更する\n2. FFmpeg で .mp4 に変換: ffmpeg -i input.mov output.mp4`);
            } else {
                alert(`⚠️ 動画の読み込みに失敗しました。\nエラー: ${errMsg}\nファイル: ${file.name}`);
            }
        };
    }

    function adjustFrameDuration(newFramesCount) {
        const oldFrames = appState.data.frames;
        const scaledFrames = [];

        for (let i = 0; i < newFramesCount; i++) {
            // Map index: Linear nearest-neighbor mapping without cumulative truncation bias (B-2)
            const oldIdx = Math.min(
                oldFrames.length - 1,
                Math.round((i / (newFramesCount - 1)) * (oldFrames.length - 1))
            );
            const srcFrame = oldFrames[oldIdx];
            
            // C-4: Deep copy JSON.parse(JSON.stringify()) によるメモリ爆発を避けるため、必要な箇所のみをシャローコピー
            const newMice = {};
            for (const [mId, mData] of Object.entries(srcFrame.mice)) {
                const newKeypoints = {};
                for (const [kpName, kpData] of Object.entries(mData.keypoints)) {
                    newKeypoints[kpName] = { x: kpData.x, y: kpData.y, conf: kpData.conf };
                }
                newMice[mId] = {
                    keypoints: newKeypoints,
                    scratch_flag: mData.scratch_flag,
                    scratch_prob: mData.scratch_prob,
                    features: { ...mData.features }
                };
            }

            scaledFrames.push({
                frame_idx: i,
                mice: newMice
            });
        }

        appState.data.frames = scaledFrames;
        appState.data.numFrames = newFramesCount;
        scrubSlider.max = newFramesCount - 1;

        autoGenerateBoutsFromFlags();
        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);
    }

    // --- 10. standard Control Buttons ---
    btnPlay.addEventListener('click', togglePlay);
    
    btnPrev.addEventListener('click', () => {
        if (appState.isPlaying) togglePlay();
        appState.currentFrame = (appState.currentFrame - 1 + appState.data.numFrames) % appState.data.numFrames;
        updateUIForFrame();
    });

    btnNext.addEventListener('click', () => {
        if (appState.isPlaying) togglePlay();
        appState.currentFrame = (appState.currentFrame + 1) % appState.data.numFrames;
        updateUIForFrame();
    });

    scrubSlider.addEventListener('input', (e) => {
        if (appState.isPlaying) togglePlay();
        appState.currentFrame = parseInt(e.target.value);
        updateUIForFrame();
    });

    // 動画直下シークバー：scrubSlider と同一の挙動（再生位置の即時移動）
    if (playbackSeekSlider) {
        playbackSeekSlider.addEventListener('input', (e) => {
            if (appState.isPlaying) togglePlay();
            appState.currentFrame = parseInt(e.target.value);
            updateUIForFrame();
        });
    }

    // B-2: キーポイント信頼度カラー表示トグル。
    // visualizer.showConfidence を切り替え、停止中でも即座に再描画する。
    const toggleConfidence = document.getElementById('toggle-confidence');
    if (toggleConfidence) {
        toggleConfidence.addEventListener('change', (e) => {
            visualizer.showConfidence = e.target.checked;
            updateUIForFrame();
        });
    }

    speedSelect.addEventListener('change', (e) => {
        appState.playbackSpeed = parseFloat(e.target.value);
        if (bgVideo && bgVideo.src) {
            bgVideo.playbackRate = appState.playbackSpeed;
        }
        if (rawVideo && rawVideo.src) {
            rawVideo.playbackRate = appState.playbackSpeed;
        }
    });

    // --- 10.5 マウスセレクターボタンイベントの委譲バインド ---
    // 動的生成ボタンに対応するため、親要素への委譲リスナーを使用する
    const mouseSelectorList = document.querySelector('.mouse-selector-list');
    if (mouseSelectorList) {
        mouseSelectorList.addEventListener('click', (e) => {
            const targetBtn = e.target.closest('[data-mouse-id]');
            if (!targetBtn) return;
            const mouseId = parseInt(targetBtn.getAttribute('data-mouse-id'));
            document.querySelectorAll('.mouse-btn').forEach(b => b.classList.remove('active'));
            targetBtn.classList.add('active');
            appState.selectedMouse = mouseId;
            updateUIForFrame();
        });
    }

    // --- 11. Custom Timeline & Bouts lists ---
    function renderTimelineBouts() {
        document.querySelectorAll('.channel-track').forEach(track => track.innerHTML = '');

        appState.data.bouts.forEach(b => {
            const mId = b.mouse_id;
            const track = document.getElementById(`track-mouse-${mId}`);
            if (!track) return;

            const startPct = (b.bout_start_frame / appState.data.numFrames) * 100;
            const durationPct = (b.bout_n_frames / appState.data.numFrames) * 100;

            const block = document.createElement('div');
            block.className = 'scratch-block';
            block.style.left = `${startPct}%`;
            block.style.width = `${durationPct}%`;
            // 動的カラーパレット対応: 未定義 ID はグレーへフォールバック
            const colorEntry = visualizer.mouseColors[mId] || { hex: "#AAAAAA" };
            block.style.color = colorEntry.hex;
            block.style.backgroundColor = colorEntry.hex;
            
            block.addEventListener('click', (e) => {
                e.stopPropagation();
                appState.currentFrame = b.bout_start_frame;
                updateUIForFrame();
            });

            track.appendChild(block);
        });
    }

    // B-3: Bout を時系列（開始フレーム→個体ID）で安定ソートする。
    // 配列インデックスがレビュー順と一致するため、ナビゲーションが直感的になる。
    function sortBouts() {
        appState.data.bouts.sort((a, b) =>
            (a.bout_start_frame - b.bout_start_frame) || (a.mouse_id - b.mouse_id));
    }

    // B-3: 承認状態キー（個体ID:開始フレーム）。フレーム番号は絶対値で安定。
    function boutKey(b) { return `${b.mouse_id}:${b.bout_start_frame}`; }

    function renderBoutsList() {
        panelBoutsList.innerHTML = '';
        if (appState.data.bouts.length === 0) {
            panelBoutsList.innerHTML = '<div style="color:var(--text-muted); font-size:0.85rem; text-align:center; padding: 20px 0;">検出されたひっかき行動はありません。</div>';
            updateBoutReviewBar();
            return;
        }

        sortBouts();
        appState.data.bouts.forEach((b, idx) => {
            const item = document.createElement('div');
            item.className = 'bout-item';
            item.dataset.boutIndex = idx;
            // B-3: 現在レビュー中の Bout 行をハイライト
            if (idx === appState.boutReview.index) item.classList.add('bout-reviewing');

            // 動的カラーパレット対応: 未定義IDはフォールバック
            const colorEntry = visualizer.mouseColors[b.mouse_id] || { name: `M${b.mouse_id}`, hex: "#AAAAAA" };
            const mouseName = colorEntry.name;
            const mouseHex = colorEntry.hex;

            // B-3: 承認済みバッジ
            const accepted = appState.boutReview.accepted.has(boutKey(b));
            const badge = accepted ? '<span class="bout-status-badge accepted">✓ 承認済</span>' : '';

            item.innerHTML = `
                <div class="bout-meta">
                    <span class="mouse-color-dot" style="color: ${mouseHex}; background-color: currentColor;"></span>
                    <div>
                        <div class="bout-title">Mouse ${b.mouse_id} (${mouseName}) - Bout #${b.bout_id}${badge}</div>
                        <div class="bout-duration">${b.bout_n_frames}f (開始: ${b.bout_start_frame}f)</div>
                    </div>
                </div>
                <div class="bout-duration">${b.bout_duration_s.toFixed(2)}s</div>
            `;

            item.addEventListener('click', () => gotoBout(idx));

            panelBoutsList.appendChild(item);
        });
        updateBoutReviewBar();
    }

    // ── B-3 Bout レビュー: ナビゲーション & 採否 ─────────────────────────────
    // 指定インデックスの Bout へジャンプし、再生ヘッドを開始フレームに移動、
    // 該当個体を選択状態にして、レビュー対象としてハイライトする。
    function gotoBout(idx) {
        const bouts = appState.data.bouts;
        if (idx < 0 || idx >= bouts.length) return;
        appState.boutReview.index = idx;
        const b = bouts[idx];
        appState.currentFrame = b.bout_start_frame;
        setSelectedMouse(b.mouse_id);
        if (appState.isPlaying) togglePlay(); // レビュー中は停止して観察しやすく
        updateUIForFrame();
        // レビュー対象行のハイライト更新 & スクロール
        panelBoutsList.querySelectorAll('.bout-item').forEach(el => {
            el.classList.toggle('bout-reviewing', parseInt(el.dataset.boutIndex) === idx);
        });
        const activeRow = panelBoutsList.querySelector(`.bout-item[data-bout-index="${idx}"]`);
        if (activeRow) activeRow.scrollIntoView({ block: 'nearest' });
        updateBoutReviewBar();
    }

    function gotoNextBout() {
        const n = appState.data.bouts.length;
        if (n === 0) return;
        const cur = appState.boutReview.index;
        gotoBout(cur < 0 ? 0 : Math.min(cur + 1, n - 1));
    }

    function gotoPrevBout() {
        const n = appState.data.bouts.length;
        if (n === 0) return;
        const cur = appState.boutReview.index;
        gotoBout(cur < 0 ? 0 : Math.max(cur - 1, 0));
    }

    // 現在の Bout を承認（真陽性確定）。データは変更せず、承認集合に記録する。
    function acceptCurrentBout() {
        const idx = appState.boutReview.index;
        const b = appState.data.bouts[idx];
        if (!b) return;
        appState.boutReview.accepted.add(boutKey(b));
        renderBoutsList();
        // 次の未承認 Bout へ自動で進む（連続レビュー効率化）
        gotoNextBout();
    }

    // 現在の Bout を却下＝誤検出として除去。区間の scratch_flag を 0 にして再生成する。
    // これは手動補正として扱い、editedFrames とエクスポートの is_manually_corrected に反映する。
    function rejectCurrentBout() {
        const idx = appState.boutReview.index;
        const b = appState.data.bouts[idx];
        if (!b) return;
        const mId = b.mouse_id;
        // 区間の各フレームについて、変更前の状態を Undo スタックに退避してから
        // scratch_flag を 0 に倒す。pushUndoSnapshot 内で markFrameMouseCorrected も
        // 呼ばれるため、is_manually_corrected フラグも各フレームに付与される。
        for (let f = b.bout_start_frame; f <= b.bout_end_frame; f++) {
            const fr = appState.data.frames[f];
            if (fr && fr.mice[mId]) {
                pushUndoSnapshot(f, mId);
                fr.mice[mId].scratch_flag = 0;
            }
        }
        appState.boutReview.accepted.delete(boutKey(b));
        // Bout 再生成（却下した区間は消える）。インデックスは範囲内にクランプ。
        autoGenerateBoutsFromFlags();
        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);
        const n = appState.data.bouts.length;
        appState.boutReview.index = n === 0 ? -1 : Math.min(idx, n - 1);
        if (appState.boutReview.index >= 0) gotoBout(appState.boutReview.index);
        else updateBoutReviewBar();
    }

    // レビュー ツールバーの表示更新（現在 Bout ラベル・進捗・ボタン活性）
    function updateBoutReviewBar() {
        const bar = document.getElementById('bout-review-bar');
        if (!bar) return;
        const curLbl = document.getElementById('bout-review-current');
        const progLbl = document.getElementById('bout-review-progress');
        const btnPrev = document.getElementById('btn-bout-prev');
        const btnNext = document.getElementById('btn-bout-next');
        const btnAcc = document.getElementById('btn-bout-accept');
        const btnRej = document.getElementById('btn-bout-reject');
        const bouts = appState.data.bouts;
        const n = bouts.length;
        const idx = appState.boutReview.index;
        const hasCurrent = idx >= 0 && idx < n;

        if (n === 0) {
            curLbl.textContent = 'Bout なし';
        } else if (!hasCurrent) {
            curLbl.textContent = `レビュー未開始 (全 ${n} 件)`;
        } else {
            const b = bouts[idx];
            const nm = (visualizer.mouseColors[b.mouse_id] || { name: `M${b.mouse_id}` }).name;
            curLbl.textContent = `${idx + 1} / ${n}: Mouse ${b.mouse_id}(${nm}) #${b.bout_id}`;
        }
        if (progLbl) {
            progLbl.textContent = n > 0 ? `承認 ${appState.boutReview.accepted.size} / ${n}` : '';
        }
        if (btnPrev) btnPrev.disabled = n === 0 || idx <= 0;
        if (btnNext) btnNext.disabled = n === 0 || idx >= n - 1;
        if (btnAcc) btnAcc.disabled = !hasCurrent;
        if (btnRej) btnRej.disabled = !hasCurrent;
    }

    // 個体選択状態を設定し、セレクターボタンの活性表示も同期する。
    function setSelectedMouse(mId) {
        appState.selectedMouse = mId;
        document.querySelectorAll('.mouse-btn').forEach(b => b.classList.remove('active'));
        const activeBtn = document.getElementById(`mouse-btn-${mId}`);
        if (activeBtn) activeBtn.classList.add('active');
    }

    // --- 12. CSV Loaders ---
    csvDropZone.addEventListener('click', () => csvFileInput.click());
    
    csvDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        csvDropZone.classList.add('drag-over');
    });

    csvDropZone.addEventListener('dragleave', () => {
        csvDropZone.classList.remove('drag-over');
    });

    csvDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        csvDropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleUploadedFiles(e.dataTransfer.files);
        }
    });

    csvFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUploadedFiles(e.target.files);
        }
    });

    function handleUploadedFiles(files) {
        let frameFile = null;
        let boutsFile = null;

        if (files.length === 1) {
            // A single file must be the frame labels file to be visualized
            frameFile = files[0];
        } else {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const lowerName = file.name.toLowerCase();
                if (lowerName.includes('bout')) {
                    boutsFile = file;
                } else if (lowerName.includes('frame') || lowerName.includes('label') || lowerName.includes('track') || lowerName.endsWith('.csv')) {
                    if (!frameFile || lowerName.includes('frame') || lowerName.includes('label') || lowerName.includes('track')) {
                        frameFile = file;
                    }
                }
            }
        }

        if (!frameFile) {
            alert("⚠️ 少なくともフレームラベルを含むCSVファイル（例: cage_01_frame_labels.csv）をアップロードしてください。");
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            const text = e.target.result;
            try {
                const parsedFrames = parseFrameCsv(text);
                if (parsedFrames.length === 0) throw new Error("Parsed zero records");

                // 大規模データ対策: Math.max(...array) は巨大配列でスプレッド展開すると
                // 引数数が上限を超え "Maximum call stack size exceeded" で落ちるため、
                // ループで最大フレーム番号を求める（数十万行でも安全）。
                let maxFrameIdx = 0;
                for (const r of parsedFrames) { if (r.frame_idx > maxFrameIdx) maxFrameIdx = r.frame_idx; }
                const totalFrames = maxFrameIdx + 1;

                const framesArray = [];
                for (let f = 0; f < totalFrames; f++) {
                    framesArray.push({
                        frame_idx: f,
                        mice: {}
                    });
                }

                // 動的マウス数対応: CSV に含まれる全マウスIDを収集
                const csvMiceIds = new Set();
                parsedFrames.forEach(row => {
                    const f = row.frame_idx;
                    const mId = row.mouse_id;
                    // 有効範囲チェック: 1以上 MAX_MICE_LIMIT 以下
                    if (f < totalFrames && mId >= 1 && mId <= MAX_MICE_LIMIT) {
                        framesArray[f].mice[mId] = {
                            keypoints: row.keypoints,
                            scratch_flag: row.scratch_flag,
                            scratch_prob: row.scratch_prob,
                            features: row.features,
                            disabled: false,
                        };
                        csvMiceIds.add(mId);
                    }
                });

                // appState.miceIds を CSV から検出されたIDで更新
                const detectedMiceIds = Array.from(csvMiceIds).sort((a, b) => a - b);
                appState.miceIds = detectedMiceIds.length > 0 ? detectedMiceIds : [1, 2, 3, 4];

                // 欠損フレーム補完: 前フレームから補間（動的個体数変化に対応）
                // 途中で個体が現れた場合はそのフレームから補完を開始する（以前に存在しない個体は補完しない）
                for (let f = 0; f < totalFrames; f++) {
                    const frame = framesArray[f];
                    for (const mId of appState.miceIds) {
                        if (!frame.mice[mId]) {
                            // 前フレームに同IDが存在する場合のみ補完（途中出現前の補完はしない）
                            if (f > 0 && framesArray[f-1].mice[mId]) {
                                const prev = framesArray[f-1].mice[mId];
                                frame.mice[mId] = {
                                    keypoints: _cloneKeypoints(prev.keypoints),
                                    scratch_flag: 0,
                                    scratch_prob: 0,
                                    features: { ...prev.features },
                                    disabled: false,
                                };
                            }
                            // 前フレームに存在しない場合は何もしない（途中登場マウスは空のまま）
                        }
                    }
                }

                appState.data.frames = framesArray;
                appState.data.numFrames = totalFrames;
                appState.data.bouts = [];

                if (boutsFile) {
                    const boutsReader = new FileReader();
                    boutsReader.onload = function(eb) {
                        try {
                            appState.data.bouts = parseBoutsCsv(eb.target.result);
                            finalizeImport(frameFile.name);
                        } catch (err) {
                            console.error("Error parsing bouts CSV:", err);
                            autoGenerateBoutsFromFlags();
                            finalizeImport(frameFile.name + " (Auto-bouts)");
                        }
                    };
                    boutsReader.readAsText(boutsFile);
                } else {
                    autoGenerateBoutsFromFlags();
                    finalizeImport(frameFile.name + " (Auto-bouts)");
                }

            } catch (err) {
                console.error("CSV Parse error:", err);
                alert("⚠️ CSVの解析に失敗しました。");
            }
        };
        reader.readAsText(frameFile);
    }

    function createDefaultStaticMouse(cx, cy) {
        const kps = {};
        KEYPOINTS_LIST.forEach(kp => {
            kps[kp] = { x: cx, y: cy, conf: 0.5 };
        });
        return {
            keypoints: kps,
            scratch_flag: 0,
            scratch_prob: 0.0,
            features: {
                body_length_px: 40,
                head_to_nose_dist: 10,
                hind_to_head_dist_L: 80,
                hind_to_head_dist_R: 80,
                hind_to_ear_dist_L: 75,
                hind_to_ear_dist_R: 75,
                hind_paw_speed_L: 0,
                hind_paw_speed_R: 0,
                hind_osc_x_L: 0,
                hind_osc_y_L: 0,
                hind_osc_x_R: 0,
                hind_osc_y_R: 0
            }
        };
    }

    // --- Canvas ↔ 動画座標系 同期ユーティリティ ---
    // 動画は bg-video に object-fit: contain で表示されるため、コンテナより小さな
    // 描画領域にレターボックス付きで収まる。cage-canvas をその描画領域に
    // ピッタリ重ね、内部解像度を動画のネイティブ解像度に設定することで
    // パイプライン出力のキーポイント座標（動画ピクセル空間）が canvas 内部座標系と
    // 完全に一致し、マーカーがマウスの実位置に正確に描画される。

    function resizeCanvasToVideo() {
        if (!appState.videoW || !appState.videoH) return;
        const canvas = document.getElementById('cage-canvas');
        if (!canvas) return;
        const container = canvas.parentElement;
        if (!container) return;

        const cRect = container.getBoundingClientRect();
        const cW = cRect.width;
        const cH = cRect.height;
        if (!cW || !cH) return;

        const vW = appState.videoW;
        const vH = appState.videoH;

        // object-fit: contain のスケール係数：縦横比を保ちコンテナに収める
        const s = Math.min(cW / vW, cH / vH);
        const rW = vW * s;
        const rH = vH * s;
        // レターボックス（黒帯）のオフセット
        const left = (cW - rW) / 2;
        const top  = (cH - rH) / 2;

        // CSS 表示サイズを動画描画領域に合わせて配置
        canvas.style.left   = `${left}px`;
        canvas.style.top    = `${top}px`;
        canvas.style.width  = `${rW}px`;
        canvas.style.height = `${rH}px`;

        // 内部解像度を動画の実ピクセル解像度に合わせる
        // キーポイント座標（動画ピクセル空間）= canvas 内部座標系 が成立する
        if (canvas.width !== vW || canvas.height !== vH) {
            canvas.width  = vW;
            canvas.height = vH;
        }
    }

    function resetCanvasToDefault() {
        const canvas = document.getElementById('cage-canvas');
        if (!canvas) return;
        canvas.style.left   = '0';
        canvas.style.top    = '0';
        canvas.style.width  = '100%';
        canvas.style.height = '100%';
        canvas.width  = 800;
        canvas.height = 550;
    }

    // ウィンドウリサイズ時に canvas を動画描画領域に再フィット
    window.addEventListener('resize', () => {
        if (appState.hasRealData && appState.videoW && appState.videoH) {
            resizeCanvasToVideo();
        }
    });

    function finalizeImport(fileName) {
        appState.currentFrame = 0;
        appState.isPlaying = false;
        playIcon.setAttribute('data-lucide', 'play');
        lucide.createIcons();

        scrubSlider.max = appState.data.numFrames - 1;

        // 動的UIの再構築 (マウスセレクター・タイムラインチャネル)
        initDynamicMiceUI(appState.miceIds);
        // 編集履歴をリセット
        appState.undoStack = [];
        appState.redoStack = [];
        appState.editedFrames = new Set();
        // [B-3] Bout レビュー状態を新規ロードごとに初期化
        appState.boutReview = { index: -1, accepted: new Set() };
        updateUndoRedoButtons();
        updateCorrectionStatus();  // [A-5] 新規ロード時はプレビュー状態に戻す

        // 実CSVデータがロードされたことを記録し、canvas を動画描画領域に合わせてリサイズ
        appState.hasRealData = true;
        if (appState.videoW && appState.videoH) {
            resizeCanvasToVideo();
        }

        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);
        updateUIForFrame();

        const fileBadge = document.getElementById('loaded-file-name');
        if (fileBadge) {
            fileBadge.textContent = fileName;
            fileBadge.style.color = "var(--accent)";
            fileBadge.style.borderColor = "var(--accent)";
        }
        
        alert(`✓ 解析データを正常にインポートしました!\nフレーム数: ${appState.data.numFrames}`);
    }

    function autoGenerateBoutsFromFlags() {
        // バックエンド feature_extraction.py の Bout 検出と同じ閾値を使用
        const MIN_FRAMES = SCRATCH_THRESHOLDS.boutMinFrames;
        const MERGE_GAP  = SCRATCH_THRESHOLDS.boutMergeGap;
        const bouts = [];
        const frames = appState.data.frames;

        // 動的マウス数対応: appState.miceIds を使用（固定 1–4 ループを廃止）
        for (const mId of appState.miceIds) {
            let inBout = false;
            let startFrame = 0;
            let boutIdx = 1;

            for (let f = 0; f < frames.length; f++) {
                const isScratching = frames[f].mice[mId]?.scratch_flag === 1;

                if (isScratching && !inBout) {
                    inBout = true;
                    startFrame = f;
                } else if (!isScratching && inBout) {
                    // 直近 MERGE_GAP フレーム内に scratch_flag=1 が再出現すれば結合
                    let merge = false;
                    for (let peek = f; peek < Math.min(f + MERGE_GAP, frames.length); peek++) {
                        if (frames[peek].mice[mId]?.scratch_flag === 1) {
                            merge = true;
                            f = peek - 1;
                            break;
                        }
                    }

                    if (!merge) {
                        const durationFrames = f - startFrame;
                        if (durationFrames >= MIN_FRAMES) {
                            bouts.push({
                                mouse_id: mId,
                                bout_id: boutIdx++,
                                bout_start_frame: startFrame,
                                bout_end_frame: f - 1,
                                bout_start_time_s: startFrame / appState.data.fps,
                                bout_end_time_s: (f - 1) / appState.data.fps,
                                bout_duration_s: durationFrames / appState.data.fps,
                                bout_n_frames: durationFrames
                            });
                        }
                        inBout = false;
                    }
                }
            }

            if (inBout && (frames.length - startFrame >= MIN_FRAMES)) {
                bouts.push({
                    mouse_id: mId,
                    bout_id: boutIdx++,
                    bout_start_frame: startFrame,
                    bout_end_frame: frames.length - 1,
                    bout_start_time_s: startFrame / appState.data.fps,
                    bout_end_time_s: (frames.length - 1) / appState.data.fps,
                    bout_duration_s: (frames.length - startFrame) / appState.data.fps,
                    bout_n_frames: frames.length - startFrame
                });
            }
        }
        appState.data.bouts = bouts;
    }

    function parseFrameCsv(csvText) {
        const lines = csvText.split(/\r?\n/);
        if (lines.length < 2) return [];

        const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
        const records = [];

        const fIdx = headers.indexOf('frame_idx');
        const trackIdx = headers.indexOf('track');
        const mouseIdIdx = headers.indexOf('mouse_id');
        const flagIdx = headers.indexOf('scratch_flag');
        const probIdx = headers.indexOf('scratch_prob');
        const isMockIdx = headers.indexOf('is_mock');
        const idConfIdx = headers.indexOf('id_confidence');  // [A-3] カラーID信頼度

        if (fIdx === -1) {
            throw new Error("Missing 'frame_idx'");
        }

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const cols = parseCsvLine(line);
            if (cols.length < headers.length) continue;

            const frame = parseInt(cols[fIdx]);
            let rawMouseId = mouseIdIdx !== -1 ? cols[mouseIdIdx] : cols[trackIdx];
            let mouseId = parseInt(rawMouseId);
            if (isNaN(mouseId)) {
                if (rawMouseId.includes('1') || rawMouseId.toLowerCase().includes('red')) mouseId = 1;
                else if (rawMouseId.includes('2') || rawMouseId.toLowerCase().includes('blue')) mouseId = 2;
                else if (rawMouseId.includes('3') || rawMouseId.toLowerCase().includes('yellow')) mouseId = 3;
                else if (rawMouseId.includes('4') || rawMouseId.toLowerCase().includes('black')) mouseId = 4;
                else mouseId = 1;
            }

            const scratchFlag = flagIdx !== -1 ? parseInt(cols[flagIdx]) : 0;
            const scratchProb = probIdx !== -1 ? parseFloat(cols[probIdx]) : 0.0;
            // [A-3] カラーID信頼度（採用色得票率）。列が無い旧CSVでは null。
            let idConfidence = null;
            if (idConfIdx !== -1) {
                const c = parseFloat(cols[idConfIdx]);
                idConfidence = isNaN(c) ? null : c;
            }

            const keypoints = {};
            KEYPOINTS_LIST.forEach(kp => {
                const xColName = `${kp}_x`;
                const yColName = `${kp}_y`;
                const xColIdx = headers.indexOf(xColName);
                const yColIdx = headers.indexOf(yColName);

                if (xColIdx !== -1 && yColIdx !== -1) {
                    // [B-2] キーポイント信頼度をCSVから取得する。
                    // 姿勢推定エンジン（SLEAP/YOLO）が出力した {kp}_conf 列が存在すればその値を、
                    // 列が無い・取得不能（-1.0 等の負値）の場合は null とし、
                    // 「信頼度データなし」として骨格カラー表示では中立色にフォールバックする。
                    const confColIdx = headers.indexOf(`${kp}_conf`);
                    let kpConf = null;
                    if (confColIdx !== -1) {
                        const cv = parseFloat(cols[confColIdx]);
                        kpConf = (isNaN(cv) || cv < 0) ? null : cv;
                    }
                    keypoints[kp] = {
                        x: Math.round(parseFloat(cols[xColIdx])),
                        y: Math.round(parseFloat(cols[yColIdx])),
                        conf: kpConf
                    };
                }
            });

            const features = {
                body_length_px: getFloatVal(headers, cols, 'body_length_px', 45),
                head_to_nose_dist: getFloatVal(headers, cols, 'head_to_nose_dist', 12),
                hind_to_head_dist_L: getFloatVal(headers, cols, 'hind_to_head_dist_L', 80),
                hind_to_head_dist_R: getFloatVal(headers, cols, 'hind_to_head_dist_R', 80),
                hind_to_ear_dist_L: getFloatVal(headers, cols, 'hind_to_ear_dist_L', 75),
                hind_to_ear_dist_R: getFloatVal(headers, cols, 'hind_to_ear_dist_R', 75),
                hind_paw_speed_L: getFloatVal(headers, cols, 'hind_paw_speed_L', 0),
                hind_paw_speed_R: getFloatVal(headers, cols, 'hind_paw_speed_R', 0),
                
                // 物理単位特徴量のパース (NEW)
                body_length_mm: getFloatVal(headers, cols, 'body_length_mm', undefined),
                head_to_nose_dist_mm: getFloatVal(headers, cols, 'head_to_nose_dist_mm', undefined),
                hind_to_head_dist_L_mm: getFloatVal(headers, cols, 'hind_to_head_dist_L_mm', undefined),
                hind_to_head_dist_R_mm: getFloatVal(headers, cols, 'hind_to_head_dist_R_mm', undefined),
                hind_to_ear_dist_L_mm: getFloatVal(headers, cols, 'hind_to_ear_dist_L_mm', undefined),
                hind_to_ear_dist_R_mm: getFloatVal(headers, cols, 'hind_to_ear_dist_R_mm', undefined),
                hind_paw_speed_L_mm_s: getFloatVal(headers, cols, 'hind_paw_speed_L_mm_s', undefined),
                hind_paw_speed_R_mm_s: getFloatVal(headers, cols, 'hind_paw_speed_R_mm_s', undefined),
                
                hind_osc_x_L: getFloatVal(headers, cols, 'hind_osc_x_L', 0),
                hind_osc_y_L: getFloatVal(headers, cols, 'hind_osc_y_L', 0),
                hind_osc_x_R: getFloatVal(headers, cols, 'hind_osc_x_R', 0),
                hind_osc_y_R: getFloatVal(headers, cols, 'hind_osc_y_R', 0)
            };

            const isMockVal = isMockIdx !== -1 ? parseInt(cols[isMockIdx]) : 0;

            if (Object.keys(keypoints).length === 0) {
                const defaultCenters = { 1: [200, 150], 2: [600, 150], 3: [200, 400], 4: [600, 400] };
                const [cx, cy] = defaultCenters[mouseId];
                KEYPOINTS_LIST.forEach(kp => {
                    keypoints[kp] = { x: cx, y: cy, conf: 0.5 };
                });
            }

            records.push({
                frame_idx: frame,
                mouse_id: mouseId,
                scratch_flag: scratchFlag,
                scratch_prob: scratchProb,
                id_confidence: idConfidence,
                is_mock: isMockVal,
                keypoints,
                features
            });
        }

        return records;
    }

    function parseBoutsCsv(csvText) {
        if (!csvText) return [];
        const lines = csvText.split(/\r?\n/);
        if (lines.length < 2) return [];

        const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
        const bouts = [];

        const mIdx = headers.indexOf('mouse_id');
        const bIdx = headers.indexOf('bout_id');
        const sfIdx = headers.indexOf('bout_start_frame');
        const efIdx = headers.indexOf('bout_end_frame');
        const durIdx = headers.indexOf('bout_duration_s');

        const fps = (appState && appState.data && appState.data.fps) || 30;

        for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            const cols = parseCsvLine(line);
            if (cols.length < headers.length) continue;

            const startF = parseInt(cols[sfIdx]);
            const endF = parseInt(cols[efIdx]);
            const nFrames = endF - startF + 1;

            bouts.push({
                mouse_id: parseInt(cols[mIdx]),
                bout_id: parseInt(cols[bIdx] || i),
                bout_start_frame: startF,
                bout_end_frame: endF,
                bout_start_time_s: parseFloat(cols[headers.indexOf('bout_start_time_s')] || startF / fps),
                bout_end_time_s: parseFloat(cols[headers.indexOf('bout_end_time_s')] || endF / fps),
                bout_duration_s: parseFloat(cols[durIdx] || nFrames / fps),
                bout_n_frames: nFrames
            });
        }

        return bouts;
    }

    function parseCsvLine(line) {
        const result = [];
        let curVal = "";
        let insideQuote = false;

        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                insideQuote = !insideQuote;
            } else if (char === ',' && !insideQuote) {
                result.push(curVal.trim().replace(/^"|"$/g, ''));
                curVal = "";
            } else {
                curVal += char;
            }
        }
        result.push(curVal.trim().replace(/^"|"$/g, ''));
        return result;
    }

    function getFloatVal(headers, cols, key, fallback) {
        const idx = headers.indexOf(key);
        if (idx === -1) return fallback;
        const val = parseFloat(cols[idx]);
        return isNaN(val) ? fallback : val;
    }

    // --- 12.4 Bulk Marker Offset Correction ---
    const btnApplyOffset = document.getElementById('btn-apply-offset');
    if (btnApplyOffset) {
        btnApplyOffset.addEventListener('click', () => {
            const mouseId = parseInt(document.getElementById('offset-mouse-id').value);
            const dx = parseFloat(document.getElementById('offset-dx').value) || 0;
            const dy = parseFloat(document.getElementById('offset-dy').value) || 0;
            const range = document.querySelector('input[name="offset-range"]:checked')?.value || 'current';

            if (dx === 0 && dy === 0) {
                alert('⚠️ X方向またはY方向のオフセット値を入力してください。');
                return;
            }

            // Pause if playing
            if (appState.isPlaying) togglePlay();

            let startF = 0;
            let endF = appState.data.numFrames - 1;

            if (range === 'current') {
                startF = appState.currentFrame;
                endF = appState.currentFrame;
            } else if (range === 'forward') {
                startF = appState.currentFrame;
                endF = appState.data.numFrames - 1;
            }
            // 'all' uses the defaults (0 to numFrames-1)

            // Undo: 現フレームのみスナップショット（範囲オフセット全フレーム保存はメモリ爆発のため避ける）
            pushUndoSnapshot(appState.currentFrame, mouseId);

            let modifiedCount = 0;
            for (let f = startF; f <= endF; f++) {
                const frame = appState.data.frames[f];
                if (!frame || !frame.mice[mouseId]) continue;

                const kps = frame.mice[mouseId].keypoints;
                for (const kpName of Object.keys(kps)) {
                    kps[kpName].x = Math.round(kps[kpName].x + dx);
                    kps[kpName].y = Math.round(kps[kpName].y + dy);
                }
                // [A-5] バルクオフセットで変更された全フレームを補正済みとして記録
                markFrameMouseCorrected(f, mouseId);
                modifiedCount++;
            }

            // Recalculate kinematics in affected range
            for (let f = Math.max(1, startF); f <= endF; f++) {
                recalculateFrameKinematics(f, mouseId);
            }

            // Re-detect bouts
            autoGenerateBoutsFromFlags();
            renderTimelineBouts();
            renderBoutsList();
            charts.updateAnalytics(appState.data.bouts);
            updateUIForFrame(false);

            const rangeLabel = range === 'current' ? '現在フレーム' : range === 'forward' ? `フレーム ${startF} 以降` : '全フレーム';
            alert(`✓ Mouse ${mouseId} の全関節を (dx=${dx >= 0 ? '+' : ''}${dx}, dy=${dy >= 0 ? '+' : ''}${dy}) px 移動しました。\n適用範囲: ${rangeLabel} (${modifiedCount} フレーム)`);
        });
    }

    // --- 12.5 FastAPI pipeline integration Event handlers (NEW) ---

    // Mock エンジン選択時は「初期化確認モード」トグルを非表示にする
    // （Mock はダミー座標を返すため、確認モードを通す意味がない）
    function _syncPreviewToggleVisibility() {
        const label = document.getElementById('preview-mode-toggle-label');
        const chk   = document.getElementById('chk-use-preview-mode');
        if (!label || !pipelineEngine) return;
        if (pipelineEngine.value === 'mock') {
            label.style.display = 'none';
            if (chk) chk.checked = false; // 確認モードを強制 OFF
        } else {
            label.style.display = 'flex';
        }
    }
    if (pipelineEngine) {
        pipelineEngine.addEventListener('change', _syncPreviewToggleVisibility);
        _syncPreviewToggleVisibility(); // 起動時にも反映
    }

    // パス入力欄に直接テキストを入力した場合もボタンを活性化する
    // （動画をドロップせずにサーバーパスを手動入力するワークフロー向け）
    if (pipelineVideoPath) {
        pipelineVideoPath.addEventListener('input', () => {
            const hasPath = pipelineVideoPath.value.trim().length > 0;
            btnRunPipeline.disabled = !hasPath;
            if (hasPath) {
                btnRunPipeline.textContent = "解析パイプラインを実行";
                btnRunPipeline.style.background = "rgba(168, 85, 247, 0.15)";
                btnRunPipeline.style.borderColor = "#A855F7";
            } else {
                btnRunPipeline.textContent = "解析実行 (動画選択後に有効)";
                btnRunPipeline.style.background = "";
                btnRunPipeline.style.borderColor = "";
            }
        });
    }

    btnRunPipeline.addEventListener('click', () => {
        // 新設された入力欄からパスを取得し、なければ dataset からのフォールバックを使用
        const inputVideoPath = document.getElementById('pipeline-video-path');
        const videoName = (inputVideoPath && inputVideoPath.value.trim()) || btnRunPipeline.dataset.videoName;
        
        if (!videoName) {
            alert("⚠️ 動画ファイルが指定されていません。先に動画をドロップするか、パスを直接入力してください。");
            return;
        }

        // GUI 上の擬似エンジン値を実エンジン + DLC モデル名へ展開する
        // 例: "dlc_quadruped" → engine=dlc, dlc_superanimal=superanimal_quadruped
        const engineRaw = pipelineEngine.value;
        let engine, dlcSuperanimal = null;
        if (engineRaw === "dlc_quadruped") {
            engine = "dlc";
            dlcSuperanimal = "superanimal_quadruped";
        } else if (engineRaw === "dlc") {
            engine = "dlc";
            dlcSuperanimal = "superanimal_topviewmouse";
        } else {
            engine = engineRaw;
        }
        const calibration = parseFloat(pipelineCalibration.value) || 0.25;
        const modelPath = pipelineModelPath.value.trim() || null;

        // マウス頭数指定（0 = 自動検出）。バックエンド config.py の CAGE_MICE_COUNT に対応。
        const miceCount = pipelineMiceCount ? Math.max(0, parseInt(pipelineMiceCount.value) || 0) : 0;

        // ROI(関心領域)指定 — 全空欄なら無視。範囲外検出の抑制に使用。
        const roiX1 = pipelineRoiX1 && pipelineRoiX1.value !== '' ? parseInt(pipelineRoiX1.value) : null;
        const roiY1 = pipelineRoiY1 && pipelineRoiY1.value !== '' ? parseInt(pipelineRoiY1.value) : null;
        const roiX2 = pipelineRoiX2 && pipelineRoiX2.value !== '' ? parseInt(pipelineRoiX2.value) : null;
        const roiY2 = pipelineRoiY2 && pipelineRoiY2.value !== '' ? parseInt(pipelineRoiY2.value) : null;

        // ── フェーズ1 初期化確認モードのトグル確認 ──────────────────────────
        const usePreviewMode = document.getElementById('chk-use-preview-mode')?.checked === true;
        // 本解析開始時に再利用するため、パイプライン設定を保存
        const pipelineConfig = {
            videoName, engine, modelPath, calibration, miceCount,
            dlcSuperanimal, roiX1, roiY1, roiX2, roiY2,
        };

        // 状態を「進行中」へ変更
        btnRunPipeline.disabled = true;
        btnRunPipeline.textContent = usePreviewMode ? "プレビュー解析中..." : "解析実行中...";
        pipelineStatusPanel.style.display = "block";
        pipelineStatusPanel.textContent = usePreviewMode
            ? "[Server] プレビュー解析リクエストを送信中 (冒頭5分)...\n"
            : "[Server] 解析リクエストを送信中...\n";
        pipelineStatusPanel.style.color = "#38BDF8";

        // ── Fix: バッジを「解析中」に切り替え ───────────────────────────────
        // 解析完了（finalizeImport）まで「デモ表示中」のまま残らないよう更新する
        appState.isAnalysisRunning = true;
        const _badge = document.getElementById('loaded-file-name');
        if (_badge) {
            _badge.textContent = usePreviewMode ? `プレビュー解析中: ${videoName}` : `解析実行中: ${videoName}`;
            _badge.style.color = "#38BDF8";
        }

        // ── Fix: デモ用オーバーレイを非表示にしてキャンバスをクリア ─────────
        // 実際の解析データが来る前にデモ座標が表示されないようにする
        const _cvs = document.getElementById('cage-canvas');
        if (_cvs) {
            const _ctx = _cvs.getContext('2d');
            _ctx.clearRect(0, 0, _cvs.width, _cvs.height);
            // 「解析中」プレースホルダーを描画
            _ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
            _ctx.fillRect(0, 0, _cvs.width, _cvs.height);
            _ctx.fillStyle = "#38BDF8";
            _ctx.font = `bold ${Math.floor(_cvs.height * 0.045)}px sans-serif`;
            _ctx.textAlign = "center";
            _ctx.textBaseline = "middle";
            _ctx.fillText("🔬 DLC 解析中 — 完了後にオーバーレイが表示されます", _cvs.width / 2, _cvs.height / 2);
            _ctx.font = `${Math.floor(_cvs.height * 0.03)}px sans-serif`;
            _ctx.fillStyle = "rgba(255,255,255,0.5)";
            _ctx.fillText(`動画: ${videoName}`, _cvs.width / 2, _cvs.height / 2 + _cvs.height * 0.06);
        }

        // 確認モード状態を初期化
        if (usePreviewMode) {
            appState.previewMode = {
                active: true,
                step: null,
                pipelineConfig,
                previewData: null,
                videoName: null,
                detectedTracks: [],
            };
            appState.confirmedMapping = {};
        }

        // 解析進捗バーを開始
        startAnalysisProgressBar();

        // APIリクエストの送信
        fetch("/api/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                video_path: videoName,
                engine: engine,
                model_path: modelPath,
                px_to_mm: calibration,
                mice_count: miceCount,
                // ROI 指定: 全 4 値が揃っている場合のみ送信
                ...(roiX1 != null && roiY1 != null && roiX2 != null && roiY2 != null
                    ? { roi_x1: roiX1, roi_y1: roiY1, roi_x2: roiX2, roi_y2: roiY2 } : {}),
                // DLC エンジン時のみ Model Zoo モデル名を渡す (それ以外はサーバ側で無視される)
                ...(dlcSuperanimal ? { dlc_superanimal: dlcSuperanimal } : {}),
                // フェーズ1 初期化確認モードの場合は preview_only を有効化
                ...(usePreviewMode ? { preview_only: true, preview_duration_s: 300.0 } : {}),
            })
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.detail || "APIエラーが発生しました。") });
            }
            return res.json();
        })
        .then(data => {
            const taskId = data.task_id;
            pipelineStatusPanel.textContent += `[Server] タスク登録成功: ID=${taskId}\n`;
            
            // ポーリング開始
            pollAnalysisStatus(taskId);
        })
        .catch(err => {
            console.error(err);
            pipelineStatusPanel.textContent += `\n[Error] 解析の開始に失敗しました: ${err.message}`;
            pipelineStatusPanel.style.color = "#EF4444";
            btnRunPipeline.disabled = false;
            btnRunPipeline.textContent = "解析パイプラインを実行";
        });
    });

    // --- Analysis Progress Bar Helpers ---
    function startAnalysisProgressBar() {
        appState.isRealProgress = false; // Reset real progress flag (Q-4)
        analysisProgressContainer.style.display = 'block';
        analysisProgressBar.style.width = '0%';
        analysisProgressBar.classList.add('running');
        analysisProgressPct.textContent = '0%';
        analysisProgressLabel.textContent = '解析エンジン起動中...';
        analysisProgressLabel.style.color = 'var(--text-muted)';

        // Pseudo-animation: ramp up to ~90% over ~30 seconds
        let pseudoPct = 0;
        const stages = [
            { target: 10, label: 'エンジン起動中...', duration: 3000 },
            { target: 35, label: 'ポーズ推定実行中...', duration: 10000 },
            { target: 60, label: 'カラーID自動同定中...', duration: 8000 },
            { target: 80, label: '特徴量抽出中...', duration: 6000 },
            { target: 92, label: 'ひっかき判定中...', duration: 4000 },
        ];

        let stageIdx = 0;
        const advance = () => {
            if (appState.isRealProgress) return; // Q-4: Exclude pseudo-progress if real progress exists
            if (stageIdx >= stages.length) return;
            const stage = stages[stageIdx];
            const step = (stage.target - pseudoPct) / (stage.duration / 300);
            const timer = setInterval(() => {
                if (appState.isRealProgress) {
                    clearInterval(timer);
                    return;
                }
                pseudoPct = Math.min(pseudoPct + step, stage.target);
                const pct = Math.round(pseudoPct);
                analysisProgressBar.style.width = `${pct}%`;
                analysisProgressPct.textContent = `${pct}%`;
                analysisProgressLabel.textContent = stage.label;
                if (pseudoPct >= stage.target) {
                    clearInterval(timer);
                    stageIdx++;
                    advance();
                }
            }, 300);
            appState.analysisProgressTimer = timer;
        };
        advance();
    }

    function completeAnalysisProgressBar(success = true) {
        if (appState.analysisProgressTimer) {
            clearInterval(appState.analysisProgressTimer);
            appState.analysisProgressTimer = null;
        }
        analysisProgressBar.classList.remove('running');
        if (success) {
            analysisProgressBar.style.width = '100%';
            analysisProgressPct.textContent = '100%';
            analysisProgressLabel.textContent = '✓ 解析完了！';
            analysisProgressLabel.style.color = '#10B981';
        } else {
            analysisProgressBar.style.background = 'linear-gradient(90deg, #EF4444 0%, #B91C1C 100%)';
            analysisProgressLabel.textContent = '✗ エラーが発生しました';
            analysisProgressLabel.style.color = '#EF4444';
        }
    }

    function pollAnalysisStatus(taskId) {
        if (appState.pollingInterval) {
            clearInterval(appState.pollingInterval);
        }
        appState.pollingInterval = setInterval(() => {
            fetch(`/api/status/${taskId}`)
            .then(res => res.json())
            .then(data => {
                // ログの最新部分を表示
                pipelineStatusPanel.textContent = data.log_full;
                pipelineStatusPanel.scrollTop = pipelineStatusPanel.scrollHeight;

                // ── 進捗パーセント抽出（2パターンに対応） ──────────────────────
                // パターン1: pipeline.py の明示出力 → [Progress: XX%]
                // パターン2: DLC/tqdm の標準出力 → "8%|█| 5931/71961"（\rで上書きされるため最後の行から抽出）
                let extractedPct = null;
                const explicitMatch = data.log_full.match(/\[Progress:\s*(\d+)%\]/g);
                if (explicitMatch) {
                    const last = explicitMatch[explicitMatch.length - 1];
                    extractedPct = parseInt(last.match(/(\d+)/)[0]);
                }
                if (extractedPct === null) {
                    // tqdm 形式: 行頭の "XX%" を全行から探し最後の値を採用
                    const tqdmMatches = data.log_full.match(/(?:^|\n|\r)\s*(\d+)%\s*[|│▏▎▍▌▋▊▉█]/gm);
                    if (tqdmMatches) {
                        const last = tqdmMatches[tqdmMatches.length - 1];
                        const m = last.match(/(\d+)%/);
                        if (m) extractedPct = parseInt(m[1]);
                    }
                }
                if (extractedPct !== null && extractedPct > 0) {
                    appState.isRealProgress = true; // Q-4: 実進捗が取得できた場合は擬似アニメを停止
                    if (appState.analysisProgressTimer) {
                        clearInterval(appState.analysisProgressTimer);
                        appState.analysisProgressTimer = null;
                    }
                    analysisProgressBar.style.width = `${extractedPct}%`;
                    analysisProgressPct.textContent = `${extractedPct}%`;
                }

                if (data.status === "completed") {
                    clearInterval(appState.pollingInterval);
                    appState.pollingInterval = null;
                    completeAnalysisProgressBar(true);
                    // 解析完了フラグをリセット（canvas の描画制御に使用）
                    appState.isAnalysisRunning = false;

                    // ── プレビュー完了: 確認モードへ遷移 ──────────────────
                    if (data.results && data.results.preview_data_url) {
                        pipelineStatusPanel.textContent += "\n[Server] プレビューデータを受信中...\n";
                        loadPreviewDataAndEnterConfirmation(
                            data.results.preview_data_url,
                            data.results.video_name
                        );
                        return;
                    }

                    pipelineStatusPanel.textContent += "\n[Server] 解析データを受信中...\n";
                    try {
                        loadPipelineResults(data.results.frame_labels_url, data.results.bouts_url || null);
                    } catch (err) {
                        console.error("Error loading results:", err);
                        pipelineStatusPanel.textContent += `\n[Error] 結果のロードに失敗しました: ${err.message}`;
                        pipelineStatusPanel.style.color = "#EF4444";
                        btnRunPipeline.disabled = false;
                        btnRunPipeline.textContent = "解析パイプラインを実行";
                    }
                } else if (data.status === "failed") {
                    clearInterval(appState.pollingInterval);
                    appState.pollingInterval = null;
                    completeAnalysisProgressBar(false);
                    appState.isAnalysisRunning = false; // エラー終了時もフラグをリセット
                    pipelineStatusPanel.textContent += "\n[Server] [Error] 解析が異常終了しました。\n";
                    pipelineStatusPanel.style.color = "#EF4444";
                    btnRunPipeline.disabled = false;
                    btnRunPipeline.textContent = "解析パイプラインを実行";
                    alert("⚠️ パイプライン処理でエラーが発生しました。ログをご確認ください。");
                }
            })
            .catch(err => {
                clearInterval(appState.pollingInterval);
                appState.pollingInterval = null;
                completeAnalysisProgressBar(false);
                appState.isAnalysisRunning = false; // 通信エラー時もフラグをリセット
                console.error(err);
                pipelineStatusPanel.textContent += `\n[Error] 通信エラー: ${err.message}`;
                pipelineStatusPanel.style.color = "#EF4444";
                btnRunPipeline.disabled = false;
                btnRunPipeline.textContent = "解析パイプラインを実行";
            });
        }, 1000);
    }

    function loadPipelineResults(frameCsvUrl, boutsCsvUrl) {
        const framePromise = fetch(frameCsvUrl).then(res => {
            if (!res.ok) throw new Error(`Failed to load frame data: ${res.statusText}`);
            return res.text();
        });
        const boutsPromise = boutsCsvUrl
            ? fetch(boutsCsvUrl).then(res => {
                if (!res.ok) {
                    console.warn(`Bouts CSV could not be fetched: ${res.statusText}`);
                    return "";
                }
                return res.text();
              }).catch(err => {
                console.warn(`Bouts CSV fetch error: ${err.message}`);
                return "";
              })
            : Promise.resolve("");

        Promise.all([framePromise, boutsPromise])
        .then(([frameText, boutsText]) => {
            // パースとロードの実行
            const parsedFrames = parseFrameCsv(frameText);
            if (parsedFrames.length === 0) throw new Error("Parsed zero records");

            // 大規模データ対策: Math.max(...array) は巨大配列でスプレッド展開すると
            // 引数数が上限を超え "Maximum call stack size exceeded" で落ちるため、
            // ループで最大フレーム番号を求める（数十万行でも安全）。
            let maxFrameIdx = 0;
            for (const r of parsedFrames) { if (r.frame_idx > maxFrameIdx) maxFrameIdx = r.frame_idx; }
            const totalFrames = maxFrameIdx + 1;

            const framesArray = [];
            for (let f = 0; f < totalFrames; f++) {
                framesArray.push({ frame_idx: f, mice: {} });
            }

            // モックフラグのグローバル検出 + 動的マウスID収集
            let isMockData = false;
            const pipelineMiceIds = new Set();

            parsedFrames.forEach(row => {
                const f = row.frame_idx;
                const mId = row.mouse_id;
                if (f < totalFrames && mId >= 1 && mId <= MAX_MICE_LIMIT) {
                    framesArray[f].mice[mId] = {
                        keypoints: row.keypoints,
                        scratch_flag: row.scratch_flag,
                        scratch_prob: row.scratch_prob,
                        features: row.features,
                        disabled: false,
                    };
                    pipelineMiceIds.add(mId);
                }
                if (row.is_mock === 1) isMockData = true;
            });

            // appState.miceIds を更新
            appState.miceIds = Array.from(pipelineMiceIds).sort((a, b) => a - b);
            if (appState.miceIds.length === 0) appState.miceIds = [1, 2, 3, 4];

            // 欠損フレームの補完（途中登場マウスは最初の登場フレーム以降のみ補完）
            for (let f = 0; f < totalFrames; f++) {
                const frame = framesArray[f];
                for (const mId of appState.miceIds) {
                    if (!frame.mice[mId] && f > 0 && framesArray[f-1].mice[mId]) {
                        const prev = framesArray[f-1].mice[mId];
                        frame.mice[mId] = {
                            keypoints: _cloneKeypoints(prev.keypoints),
                            scratch_flag: 0,
                            scratch_prob: 0,
                            features: { ...prev.features },
                            disabled: false,
                        };
                    }
                }
            }

            appState.data.frames = framesArray;
            appState.data.numFrames = totalFrames;
            appState.data.bouts = parseBoutsCsv(boutsText);

            // Mock警告バナーの制御 (display: flexで正しく表示)
            if (isMockData) {
                mockWarningBanner.style.display = "flex";
                mockWarningBanner.style.alignItems = "center";
                mockWarningBanner.style.justifyContent = "center";
                mockWarningBanner.style.gap = "10px";
            } else {
                mockWarningBanner.style.display = "none";
            }

            finalizeImport("API自動解析結果");
            
            // 解析実行表示を元に戻す
            btnRunPipeline.disabled = false;
            btnRunPipeline.textContent = "解析パイプラインを実行";
            pipelineStatusPanel.style.display = "none";
        })
        .catch(err => {
            console.error(err);
            alert("⚠️ 解析結果データのロードに失敗しました: " + err.message);
            btnRunPipeline.disabled = false;
            btnRunPipeline.textContent = "解析パイプラインを実行";
        });
    }

    // =====================================================================
    // ── フェーズ1 初期化確認モード制御ロジック ──────────────────────────
    // =====================================================================
    //
    // ワークフロー：
    //   1. ユーザーが「初期化確認モード」チェックボックスを有効化して「解析実行」をクリック
    //   2. pipeline.py が --preview-only で起動し、冒頭5分のバウンディングボックス JSON を生成
    //   3. loadPreviewDataAndEnterConfirmation: JSONを取得・確認モードに遷移
    //   4. enterConfirmationMode: 検出トラックごとのID割当UIを構築し、visualizer を bbox モードに
    //   5. ユーザーがシークバーで任意時刻に移動しつつ、各トラックに mouse_id を割り当て
    //   6. 「次へ：ポーズ確認」→ visualizer を pose モードに切替
    //   7. 「本解析開始」→ /api/confirm_ids にマッピング送信、本解析タスクが起動

    function loadPreviewDataAndEnterConfirmation(previewDataUrl, videoName) {
        pipelineStatusPanel.textContent += `[Server] プレビューJSON取得中: ${previewDataUrl}\n`;
        fetch(previewDataUrl)
            .then(res => {
                if (!res.ok) throw new Error(`プレビューデータ取得失敗: ${res.statusText}`);
                return res.json();
            })
            .then(previewData => {
                appState.previewMode.previewData = previewData;
                appState.previewMode.videoName = videoName;
                enterConfirmationMode(previewData);
            })
            .catch(err => {
                console.error(err);
                pipelineStatusPanel.textContent += `\n[Error] プレビューロード失敗: ${err.message}`;
                pipelineStatusPanel.style.color = "#EF4444";
                btnRunPipeline.disabled = false;
                btnRunPipeline.textContent = "解析パイプラインを実行";
            });
    }

    function enterConfirmationMode(previewData) {
        // 検出されたトラック ID 一覧を抽出（出現順を保持）
        // 併せて各トラックの出現フレーム範囲（first/last）を記録し、
        // 確認リストに「動画の何分に出現するか」を表示できるようにする。
        const trackOrder = [];
        const trackSuggestedIds = {};
        const trackFrameRange = {};  // { track: { first: frameIdx, last: frameIdx } }
        for (const frameKey of Object.keys(previewData.frames || {})) {
            const f = parseInt(frameKey);
            const dets = previewData.frames[frameKey].detections || [];
            for (const det of dets) {
                if (!trackOrder.includes(det.track)) {
                    trackOrder.push(det.track);
                    if (det.mouse_id != null) trackSuggestedIds[det.track] = det.mouse_id;
                }
                // Object.keys の順序に依存せず Math.min/max で範囲を更新
                if (!isNaN(f)) {
                    const r = trackFrameRange[det.track] || { first: f, last: f };
                    r.first = Math.min(r.first, f);
                    r.last = Math.max(r.last, f);
                    trackFrameRange[det.track] = r;
                }
            }
        }
        appState.previewMode.detectedTracks = trackOrder;
        appState.previewMode.trackFrameRange = trackFrameRange;
        appState.previewMode.suggestedIds = trackSuggestedIds;  // 「自動候補を全採用」用に保持
        appState.previewMode.step = 'bbox';
        appState.previewMode.active = true;
        // 割り当てリストの絞り込み状態
        appState.previewMode.filter = { text: '', unassignedOnly: false };
        // サムネイル生成済みキャッシュ（track → dataURL）。重複生成を避ける。
        appState.previewMode.thumbCache = appState.previewMode.thumbCache || {};

        // 自動推定されたカラーIDをデフォルトとして適用
        appState.confirmedMapping = { ...trackSuggestedIds };

        // visualizer をプレビューモード（bbox）に設定
        const sampledFrames = (previewData.sampled_frames || []).slice().sort((a, b) => a - b);
        visualizer.setConfirmationMode('bbox', previewData.frames, sampledFrames);

        // 動画解像度を appState に反映（CSVが無くても canvas を動画にフィット可能）
        if (previewData.video_width && previewData.video_height) {
            appState.videoW = previewData.video_width;
            appState.videoH = previewData.video_height;
            appState.hasRealData = true;
            if (typeof resizeCanvasToVideo === 'function') resizeCanvasToVideo();
        }

        // フレーム数/FPS を appState に反映してシークバー範囲を冒頭5分に制限
        const previewMaxFrame = Math.floor((previewData.preview_duration_s || 300) * (previewData.fps || 30));
        appState.data = appState.data || {};
        appState.data.fps = previewData.fps || 30;
        appState.data.numFrames = previewMaxFrame + 1;
        appState.data.frames = [];
        for (let f = 0; f <= previewMaxFrame; f++) {
            appState.data.frames.push({ frame_idx: f, mice: {} });
        }
        scrubSlider.max = previewMaxFrame;
        scrubSlider.value = 0;
        appState.currentFrame = 0;

        // 確認パネル表示
        const panel = document.getElementById('confirmation-panel');
        panel.style.display = 'flex';
        updateConfirmationStepLabel();
        initConfirmationToolbar();
        rebuildConfirmationMappingList();
        updateConfirmationProgress();
        updateConfirmationNextButton();
        updateUIForFrame();

        // ステータス更新
        pipelineStatusPanel.textContent += `[Confirm] 検出されたトラック数: ${trackOrder.length}\n`;
        pipelineStatusPanel.textContent += `[Confirm] シークバーで任意時刻を確認しながら、各トラックに正しい mouse_id を割り当ててください。\n`;

        // メイン解析ボタンを無効化（確認完了まで）
        btnRunPipeline.disabled = true;
        btnRunPipeline.textContent = "確認モード中...";
    }

    /**
     * 確認モードのツールバー（検索・未割当フィルタ・一括割当・自動候補採用・全クリア）を
     * 初期化する。enterConfirmationMode から一度だけ呼ばれる。
     */
    function initConfirmationToolbar() {
        const palette = (typeof MOUSE_COLOR_PALETTE !== 'undefined') ? MOUSE_COLOR_PALETTE : {};
        const maxMice = (typeof MAX_MICE_LIMIT !== 'undefined') ? MAX_MICE_LIMIT : 8;

        // 一括割当の個体IDセレクトを生成
        const bulkSel = document.getElementById('confirmation-bulk-id');
        if (bulkSel) {
            bulkSel.innerHTML = '';
            for (let i = 1; i <= maxMice; i++) {
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = `Mouse ${i} (${palette[i]?.name || 'M' + i})`;
                bulkSel.appendChild(opt);
            }
            // 誤検出（除外）も一括適用できるようにする（ノイズトラックをまとめて除外）
            const exOpt = document.createElement('option');
            exOpt.value = '0';
            exOpt.textContent = '🚫 誤検出（除外）';
            bulkSel.appendChild(exOpt);
        }

        const searchInput = document.getElementById('confirmation-search');
        const filterChk = document.getElementById('confirmation-filter-unassigned');
        const bulkApply = document.getElementById('confirmation-bulk-apply');
        const applySuggested = document.getElementById('confirmation-apply-suggested');
        const clearAll = document.getElementById('confirmation-clear-all');

        // 二重バインドを防ぐため onX プロパティで設定（再入時も上書きで安全）
        if (searchInput) {
            searchInput.value = '';
            searchInput.oninput = (e) => {
                appState.previewMode.filter.text = e.target.value.trim().toLowerCase();
                rebuildConfirmationMappingList();
            };
        }
        if (filterChk) {
            filterChk.checked = false;
            filterChk.onchange = (e) => {
                appState.previewMode.filter.unassignedOnly = e.target.checked;
                rebuildConfirmationMappingList();
            };
        }
        // 一括割当：現在「表示中」のトラックすべてに選択IDを適用（フラグメント統合に有効）
        if (bulkApply) {
            bulkApply.onclick = () => {
                const id = parseInt(bulkSel.value);
                if (isNaN(id)) return;
                const visible = getVisibleConfirmationTracks();
                if (visible.length === 0) return;
                const label = (id === 0) ? '誤検出（除外）' : `Mouse ${id}`;
                if (!confirm(`表示中の ${visible.length} 件のトラックを「${label}」に割り当てます。よろしいですか？`)) return;
                for (const t of visible) appState.confirmedMapping[t] = id;
                rebuildConfirmationMappingList();
                updateConfirmationProgress();
                updateConfirmationNextButton();
                updateUIForFrame();
            };
        }
        // 自動候補（色推定）を全トラックへ採用
        if (applySuggested) {
            applySuggested.onclick = () => {
                const sug = appState.previewMode.suggestedIds || {};
                appState.confirmedMapping = { ...sug };
                rebuildConfirmationMappingList();
                updateConfirmationProgress();
                updateConfirmationNextButton();
                updateUIForFrame();
            };
        }
        // 全クリア
        if (clearAll) {
            clearAll.onclick = () => {
                if (!confirm('全トラックの割り当てを解除します。よろしいですか？')) return;
                appState.confirmedMapping = {};
                rebuildConfirmationMappingList();
                updateConfirmationProgress();
                updateConfirmationNextButton();
                updateUIForFrame();
            };
        }
    }

    /** 現在のフィルタ条件（検索文字列・未割当のみ）に合致するトラック配列を返す。 */
    function getVisibleConfirmationTracks() {
        const f = appState.previewMode.filter || { text: '', unassignedOnly: false };
        return appState.previewMode.detectedTracks.filter(track => {
            // 「割当済み」はキー存在で判定（誤検出=0 も割当済み扱い）
            const assigned = Object.prototype.hasOwnProperty.call(appState.confirmedMapping, track);
            if (f.unassignedOnly && assigned) return false;
            if (f.text && !String(track).toLowerCase().includes(f.text)) return false;
            return true;
        });
    }

    function rebuildConfirmationMappingList() {
        const listDiv = document.getElementById('confirmation-mapping-list');
        if (!listDiv) return;
        listDiv.innerHTML = '';

        const palette = (typeof MOUSE_COLOR_PALETTE !== 'undefined') ? MOUSE_COLOR_PALETTE : {};
        const maxMice = (typeof MAX_MICE_LIMIT !== 'undefined') ? MAX_MICE_LIMIT : 8;
        const fps = (appState.data && appState.data.fps) ? appState.data.fps : 30;

        const visibleTracks = getVisibleConfirmationTracks();

        // 絞り込み結果が空の場合の案内
        if (visibleTracks.length === 0) {
            const empty = document.createElement('div');
            empty.style.cssText = 'font-size:0.7rem; color:var(--text-muted); text-align:center; padding:10px;';
            empty.textContent = '該当するトラックがありません（フィルタ条件を確認してください）。';
            listDiv.appendChild(empty);
            return;
        }

        // サムネイル生成キュー（このリビルドで表示する分のみ）
        const thumbQueue = [];

        for (const track of visibleTracks) {
            // 割り当て状態の判定：0（誤検出）も「割当済み」として扱うため、
            // truthy 判定ではなくキー存在で判定する（0 || '' は '' になってしまう罠を回避）。
            const hasAssign = Object.prototype.hasOwnProperty.call(appState.confirmedMapping, track);
            const currentId = hasAssign ? appState.confirmedMapping[track] : '';
            const isExcluded = (currentId === 0);
            const row = document.createElement('div');
            row.style.cssText = `display:flex; align-items:center; gap:6px; padding:4px; border-radius:3px; background:${isExcluded ? 'rgba(239,68,68,0.06)' : 'rgba(255,255,255,0.03)'};`;
            if (isExcluded) row.style.opacity = '0.65';  // 誤検出は控えめに表示

            // サムネイル（その枠の切り抜き）。動画から非同期で生成して埋める。
            const thumb = document.createElement('canvas');
            thumb.width = 40; thumb.height = 40;
            thumb.style.cssText = 'width:40px; height:40px; border-radius:3px; background:rgba(0,0,0,0.5); border:1px solid rgba(255,255,255,0.15); flex-shrink:0; object-fit:cover;';
            thumb.title = 'この枠の切り抜き（クリックで出現時刻へ移動）';
            row.appendChild(thumb);
            const range = appState.previewMode.trackFrameRange
                ? appState.previewMode.trackFrameRange[track] : null;
            thumbQueue.push({ track, canvas: thumb, range });

            // カラー色サンプル（誤検出=0 は赤系、未割当は灰）
            const sample = document.createElement('div');
            const colorHex = isExcluded ? '#7F1D1D' : (palette[currentId]?.hex || '#666');
            sample.style.cssText = `width:14px; height:14px; border-radius:3px; background:${colorHex}; border:1px solid rgba(255,255,255,0.3); flex-shrink:0;`;
            row.appendChild(sample);

            // トラック名
            const trackLabel = document.createElement('span');
            trackLabel.style.cssText = 'font-size:0.7rem; color:var(--text-secondary); flex: 1; font-family: monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;';
            trackLabel.textContent = track;
            row.appendChild(trackLabel);

            // 出現時刻ラベル：このトラックが動画の何分に現れるかを表示。
            // クリックするとその時刻（出現開始フレーム）へシークし、枠を直接確認できる。
            const timeBtn = document.createElement('button');
            timeBtn.type = 'button';
            timeBtn.style.cssText = 'font-size:0.66rem; font-family:monospace; color:#38BDF8; background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.35); border-radius:3px; padding:2px 5px; cursor:pointer; flex-shrink:0; white-space:nowrap;';
            const jumpToTrack = () => {
                if (!range) return;
                if (appState.isPlaying) togglePlay();
                appState.currentFrame = range.first;
                updateUIForFrame();
            };
            if (range) {
                const t0 = formatTime(range.first / fps);
                const t1 = formatTime(range.last / fps);
                timeBtn.textContent = `▶ ${t0}`;
                timeBtn.title = `このトラックの出現区間 ${t0}〜${t1}（クリックで開始時刻へ移動）`;
                timeBtn.addEventListener('click', jumpToTrack);
                thumb.style.cursor = 'pointer';
                thumb.addEventListener('click', jumpToTrack);
            } else {
                timeBtn.textContent = '--:--';
                timeBtn.disabled = true;
                timeBtn.style.opacity = '0.4';
                timeBtn.style.cursor = 'default';
            }
            row.appendChild(timeBtn);

            // ID セレクト
            const sel = document.createElement('select');
            sel.style.cssText = 'font-size:0.7rem; background:rgba(0,0,0,0.4); color:var(--text-primary); border:1px solid var(--border-color); border-radius:3px; padding:2px 4px;';
            const emptyOpt = document.createElement('option');
            emptyOpt.value = ''; emptyOpt.textContent = '— 未割当 —';
            sel.appendChild(emptyOpt);
            for (let i = 1; i <= maxMice; i++) {
                const opt = document.createElement('option');
                opt.value = i;
                const name = palette[i]?.name || `M${i}`;
                opt.textContent = `Mouse ${i} (${name})`;
                if (parseInt(currentId) === i) opt.selected = true;
                sel.appendChild(opt);
            }
            // 誤検出（除外）オプション：mouse_id=0 を割り当てると解析・集計から完全に除外される
            const exOpt = document.createElement('option');
            exOpt.value = '0';
            exOpt.textContent = '🚫 誤検出（除外）';
            if (isExcluded) exOpt.selected = true;
            sel.appendChild(exOpt);
            sel.addEventListener('change', (e) => {
                const v = e.target.value;
                if (v === '') {
                    delete appState.confirmedMapping[track];
                } else {
                    // '0'（誤検出）も含めて数値として保存する
                    appState.confirmedMapping[track] = parseInt(v);
                }
                // 行の見た目を in-place 更新（137 行でも全再構築せず軽快に保つ）
                const assignedNow = Object.prototype.hasOwnProperty.call(appState.confirmedMapping, track);
                const nowExcluded = (appState.confirmedMapping[track] === 0);
                sample.style.background = nowExcluded ? '#7F1D1D' : (palette[appState.confirmedMapping[track]]?.hex || '#666');
                row.style.background = nowExcluded ? 'rgba(239,68,68,0.06)' : 'rgba(255,255,255,0.03)';
                row.style.opacity = nowExcluded ? '0.65' : '1';
                updateConfirmationProgress();
                updateConfirmationNextButton();
                // 即座にビジュアルを更新
                updateUIForFrame();
                // 「未割当のみ」表示中に割当済みになった行はリストから外すため、その場合のみ再構築
                if (appState.previewMode.filter.unassignedOnly && assignedNow) rebuildConfirmationMappingList();
            });
            row.appendChild(sel);

            listDiv.appendChild(row);
        }

        // サムネイルを順次生成（動画シークを伴うため非同期・直列実行）
        generateConfirmationThumbnails(thumbQueue);
    }

    // ── 確認リスト用サムネイル生成（専用の隠し video を用い、本体の再生を妨げない） ──
    let _thumbVideo = null;       // サムネ抽出専用の <video>
    let _thumbBusy = false;       // 直列実行ロック
    function _getThumbVideo() {
        const bg = document.getElementById('bg-video');
        const src = bg && (bg.currentSrc || bg.src);
        if (!src) return null;
        if (!_thumbVideo) {
            _thumbVideo = document.createElement('video');
            _thumbVideo.muted = true;
            _thumbVideo.playsInline = true;
            _thumbVideo.preload = 'auto';
            // crossOrigin は設定しない：動画は同一オリジン配信 or blob のため不要で、
            // 設定すると CORS ヘッダ未送出時に canvas 描画が逆に阻害されうる。
        }
        if (_thumbVideo.src !== src) _thumbVideo.src = src;
        return _thumbVideo;
    }

    let _latestThumbQueue = null;  // 最新のサムネ生成要求（リビルドのたびに上書き）
    /**
     * thumbQueue: [{ track, canvas, range }] のサムネイルを直列生成する。
     * ワーカーは1つだけ走り、処理中に新しい要求が来たら現行ジョブ完了後に最新分を続けて処理する
     * （フィルタを素早く切り替えても取りこぼさない）。
     * 生成結果は appState.previewMode.thumbCache に dataURL でキャッシュし再利用する。
     */
    async function generateConfirmationThumbnails(thumbQueue) {
        _latestThumbQueue = thumbQueue;
        if (_thumbBusy) return;   // 既存ワーカーが finally で最新キューを拾う
        _thumbBusy = true;
        try {
            while (_latestThumbQueue) {
                const q = _latestThumbQueue;
                _latestThumbQueue = null;
                await _processThumbQueue(q);
            }
        } catch (err) {
            console.warn('[thumbnail] 生成中にエラー:', err);
        } finally {
            _thumbBusy = false;
        }
    }

    async function _processThumbQueue(thumbQueue) {
        const cache = appState.previewMode.thumbCache || (appState.previewMode.thumbCache = {});
        const previewData = appState.previewMode.previewData;
        if (!previewData) return;

        // まずキャッシュ済みは即描画
        const pending = [];
        for (const item of thumbQueue) {
            if (!item.canvas || !document.body.contains(item.canvas)) continue;
            if (cache[item.track]) {
                _paintThumbFromDataURL(item.canvas, cache[item.track]);
            } else if (item.range) {
                pending.push(item);
            }
        }
        if (pending.length === 0) return;

        const vid = _getThumbVideo();
        if (!vid) return;  // 動画未ロード時はサムネ無し（プレースホルダーのまま）

        // メタデータ読込待ち
        if (vid.readyState < 1) {
            await new Promise((res) => {
                const onMeta = () => { vid.removeEventListener('loadedmetadata', onMeta); res(); };
                vid.addEventListener('loadedmetadata', onMeta);
                setTimeout(res, 3000);  // フォールバック
            });
        }
        const fps = (appState.data && appState.data.fps) ? appState.data.fps : 30;
        for (const item of pending) {
            // 行がリストから消えていたらスキップ（フィルタ変更等）
            if (!document.body.contains(item.canvas)) continue;
            const bbox = _lookupTrackBbox(previewData, item.track, item.range.first);
            if (!bbox) continue;
            const t = item.range.first / fps;
            const dataUrl = await _captureThumb(vid, t, bbox);
            if (dataUrl) {
                cache[item.track] = dataUrl;
                if (document.body.contains(item.canvas)) _paintThumbFromDataURL(item.canvas, dataUrl);
            }
        }
    }

    /** previewData から指定トラックの指定フレームにおける bbox を取得。 */
    function _lookupTrackBbox(previewData, track, frameIdx) {
        const entry = (previewData.frames || {})[String(frameIdx)];
        if (!entry) return null;
        for (const det of (entry.detections || [])) {
            if (det.track === track && det.bbox) return det.bbox;
        }
        return null;
    }

    /** video を時刻 t にシークし、bbox 領域を切り抜いて 40x40 の dataURL を返す。 */
    function _captureThumb(vid, t, bbox) {
        return new Promise((resolve) => {
            let done = false;
            const finish = (val) => { if (!done) { done = true; vid.removeEventListener('seeked', onSeeked); resolve(val); } };
            const onSeeked = () => {
                try {
                    const [x1, y1, x2, y2] = bbox;
                    const sw = Math.max(1, x2 - x1);
                    const sh = Math.max(1, y2 - y1);
                    // 正方形サムネに収めるため少し余白を付けて中央寄せ
                    const off = document.createElement('canvas');
                    off.width = 40; off.height = 40;
                    const octx = off.getContext('2d');
                    octx.fillStyle = '#000';
                    octx.fillRect(0, 0, 40, 40);
                    // アスペクト比維持でフィット
                    const scale = Math.min(40 / sw, 40 / sh);
                    const dw = sw * scale, dh = sh * scale;
                    const dx = (40 - dw) / 2, dy = (40 - dh) / 2;
                    octx.drawImage(vid, x1, y1, sw, sh, dx, dy, dw, dh);
                    finish(off.toDataURL('image/jpeg', 0.7));
                } catch (e) {
                    // CORS 等で toDataURL が失敗する場合は諦める（プレースホルダーのまま）
                    finish(null);
                }
            };
            vid.addEventListener('seeked', onSeeked);
            try { vid.currentTime = t; } catch (e) { finish(null); }
            setTimeout(() => finish(null), 2500);  // シークが返らない場合のフォールバック
        });
    }

    /** dataURL をサムネ canvas に描画する。 */
    function _paintThumbFromDataURL(canvas, dataUrl) {
        const ctx = canvas.getContext('2d');
        const img = new Image();
        img.onload = () => { ctx.clearRect(0, 0, 40, 40); ctx.drawImage(img, 0, 0, 40, 40); };
        img.src = dataUrl;
    }

    function updateConfirmationStepLabel() {
        const lbl = document.getElementById('confirmation-step-label');
        const nextBtn = document.getElementById('btn-confirmation-next');
        if (!lbl || !nextBtn) return;
        if (appState.previewMode.step === 'bbox') {
            lbl.textContent = 'Step 1/2: バウンディングボックスの確認';
            nextBtn.textContent = '次へ：ポーズ確認 →';
        } else if (appState.previewMode.step === 'pose') {
            lbl.textContent = 'Step 2/2: ポーズ推定の確認';
            nextBtn.textContent = '本解析を開始 ▶';
        }
    }

    function updateConfirmationProgress() {
        const total = appState.previewMode.detectedTracks.length;
        const assigned = Object.keys(appState.confirmedMapping).length;
        const lbl = document.getElementById('confirmation-progress-label');
        if (lbl) lbl.textContent = `${assigned} / ${total} トラックを割り当て済み`;
    }

    function updateConfirmationNextButton() {
        const total = appState.previewMode.detectedTracks.length;
        const assigned = Object.keys(appState.confirmedMapping).length;
        const nextBtn = document.getElementById('btn-confirmation-next');
        if (!nextBtn) return;
        // 重複IDは許容する。トラッキングが断片化すると 1 個体が複数トラックに分裂するため、
        // 「複数トラック → 同一 mouse_id」は正当な統合操作であり、バックエンド
        // （color_id.py: track.map(initial_mapping.get)）も問題なく処理できる。
        // よって全トラックに割り当てが済んでいれば次へ進める。
        nextBtn.disabled = (assigned !== total);
        nextBtn.title = (assigned !== total) ? '全トラックにIDを割り当ててください' : '';
    }

    function handleConfirmationNext() {
        if (appState.previewMode.step === 'bbox') {
            // Step 1 → Step 2: bbox 確認完了、pose 表示モードへ
            appState.previewMode.step = 'pose';
            visualizer.setConfirmationMode('pose', appState.previewMode.previewData.frames,
                                            (appState.previewMode.previewData.sampled_frames || []).slice().sort((a, b) => a - b));
            // pose モードでは appState.data.frames を previewData から再構築（キーポイント表示用）
            buildFramesFromPreviewData(appState.previewMode.previewData);
            updateConfirmationStepLabel();
            updateUIForFrame();
        } else if (appState.previewMode.step === 'pose') {
            // Step 2 完了 → 本解析開始
            submitConfirmedMappingAndRunFullAnalysis();
        }
    }

    /**
     * プレビュー JSON から appState.data.frames を再構築する。
     * ポーズ確認 (Step 2) で、サンプリング済みフレームに対しては実キーポイントを描画する。
     * サンプル間のフレームは空のまま（visualizer 側で bbox を描く）。
     */
    function buildFramesFromPreviewData(previewData) {
        const numFrames = appState.data.numFrames;
        const framesArray = [];
        for (let f = 0; f < numFrames; f++) {
            framesArray.push({ frame_idx: f, mice: {} });
        }
        for (const [frameKey, frameEntry] of Object.entries(previewData.frames || {})) {
            const f = parseInt(frameKey);
            if (f >= numFrames) continue;
            for (const det of (frameEntry.detections || [])) {
                const assignedId = appState.confirmedMapping[det.track];
                // 未割当(undefined)・誤検出(0以下)は描画も解析もしない
                if (!assignedId || assignedId <= 0) continue;
                framesArray[f].mice[assignedId] = {
                    keypoints: det.keypoints,
                    scratch_flag: 0,
                    scratch_prob: 0.0,
                    features: {},
                    disabled: false,
                };
            }
        }
        appState.data.frames = framesArray;
        // appState.miceIds を更新（誤検出 ID=0 は個体に含めない）
        const ids = [...new Set(Object.values(appState.confirmedMapping))]
            .filter(v => v > 0).sort((a, b) => a - b);
        if (ids.length > 0) appState.miceIds = ids;
    }

    function submitConfirmedMappingAndRunFullAnalysis() {
        const cfg = appState.previewMode.pipelineConfig;
        if (!cfg) {
            alert("⚠️ パイプライン設定が見つかりません。確認モードをキャンセルしてやり直してください。");
            return;
        }

        const nextBtn = document.getElementById('btn-confirmation-next');
        if (nextBtn) { nextBtn.disabled = true; nextBtn.textContent = '本解析投入中...'; }
        pipelineStatusPanel.textContent += `\n[Confirm] 確認済みマッピング: ${JSON.stringify(appState.confirmedMapping)}\n`;
        pipelineStatusPanel.textContent += `[Confirm] /api/confirm_ids へ本解析リクエストを送信中...\n`;

        const body = {
            video_path: cfg.videoName,
            engine: cfg.engine,
            model_path: cfg.modelPath,
            px_to_mm: cfg.calibration,
            mice_count: cfg.miceCount,
            confirmed_mapping: appState.confirmedMapping,
            ...(cfg.roiX1 != null && cfg.roiY1 != null && cfg.roiX2 != null && cfg.roiY2 != null
                ? { roi_x1: cfg.roiX1, roi_y1: cfg.roiY1, roi_x2: cfg.roiX2, roi_y2: cfg.roiY2 } : {}),
            ...(cfg.dlcSuperanimal ? { dlc_superanimal: cfg.dlcSuperanimal } : {}),
        };

        fetch("/api/confirm_ids", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        })
        .then(res => {
            if (!res.ok) return res.json().then(e => { throw new Error(e.detail || "確認APIエラー"); });
            return res.json();
        })
        .then(data => {
            pipelineStatusPanel.textContent += `[Server] 本解析タスク登録: ID=${data.task_id}\n`;
            // 確認パネルを閉じて、visualizer を通常モードに戻す
            exitConfirmationMode();
            // 通常の解析進捗ポーリングへ移行
            startAnalysisProgressBar();
            btnRunPipeline.disabled = true;
            btnRunPipeline.textContent = "本解析実行中...";
            pollAnalysisStatus(data.task_id);
        })
        .catch(err => {
            console.error(err);
            pipelineStatusPanel.textContent += `\n[Error] 本解析開始失敗: ${err.message}`;
            pipelineStatusPanel.style.color = "#EF4444";
            if (nextBtn) { nextBtn.disabled = false; nextBtn.textContent = '本解析を開始 ▶'; }
        });
    }

    function exitConfirmationMode() {
        appState.previewMode.active = false;
        appState.previewMode.step = null;
        visualizer.setConfirmationMode(null);
        const panel = document.getElementById('confirmation-panel');
        if (panel) panel.style.display = 'none';
    }

    // 確認パネルのアクションボタンを配線
    const btnConfirmationCancel = document.getElementById('btn-confirmation-cancel');
    const btnConfirmationNext = document.getElementById('btn-confirmation-next');
    if (btnConfirmationCancel) {
        btnConfirmationCancel.addEventListener('click', () => {
            if (confirm('確認モードをキャンセルしますか？プレビュー結果は破棄されます。')) {
                exitConfirmationMode();
                btnRunPipeline.disabled = false;
                btnRunPipeline.textContent = "解析パイプラインを実行";
                pipelineStatusPanel.textContent += "\n[Confirm] 確認モードがキャンセルされました。\n";
            }
        });
    }
    if (btnConfirmationNext) {
        btnConfirmationNext.addEventListener('click', handleConfirmationNext);
    }

    // --- 12.6 Dynamic Server Result Loader bindings (NEW) ---
    const serverResultsSelect = document.getElementById('server-results-select');
    const btnRefreshServerResults = document.getElementById('btn-refresh-server-results');

    function refreshServerResultsList() {
        if (!serverResultsSelect) return;

        // Reset to default
        serverResultsSelect.innerHTML = '<option value="" selected>-- ロードする比較用CSVを選択 --</option>';

        fetch("/api/results")
        .then(res => res.json())
        .then(data => {
            if (data.results && data.results.length > 0) {
                data.results.forEach(res => {
                    const opt = document.createElement('option');
                    opt.value = res.frame_csv_url;
                    opt.dataset.boutsUrl = res.bouts_csv_url || '';
                    opt.dataset.name = res.name;
                    opt.textContent = res.name;
                    serverResultsSelect.appendChild(opt);
                });
                console.log(`[Results-Loader] サーバー内の結果ファイルを ${data.results.length} 件ロードしました。`);
            } else {
                console.log("[Results-Loader] サーバー内にロード可能な結果CSVが見つかりません。");
            }
        })
        .catch(err => {
            console.error("[Results-Loader] 結果リストの取得に失敗しました:", err);
        });
    }

    if (serverResultsSelect) {
        serverResultsSelect.addEventListener('change', (e) => {
            const selectedOpt = serverResultsSelect.options[serverResultsSelect.selectedIndex];
            const frameUrl = e.target.value;
            const boutsUrl = selectedOpt ? selectedOpt.dataset.boutsUrl : '';
            const resName = selectedOpt ? selectedOpt.dataset.name : '';

            if (!frameUrl) return;

            // Pause playback if running
            if (appState.isPlaying) togglePlay();

            console.log(`[Results-Loader] 選択された結果のロードを開始します: ${resName}`);
            
            // Progress display
            const fileBadge = document.getElementById('loaded-file-name');
            if (fileBadge) {
                fileBadge.textContent = "自動ロード中...";
                fileBadge.style.color = "#38BDF8";
            }

            // [A-3] カラーID品質レポートJSONのURLを frame_labels.csv のURLから導出する
            const qualityUrl = frameUrl.replace(/_frame_labels\.csv$/, '_id_quality.json');

            Promise.all([
                fetch(frameUrl).then(res => {
                    if (!res.ok) throw new Error("CSVファイルを取得できませんでした。");
                    return res.text();
                }),
                boutsUrl ? fetch(boutsUrl).then(res => res.ok ? res.text() : '') : Promise.resolve(''),
                // 品質JSONは存在しない場合もある（旧結果）のでエラーを握り潰す
                (qualityUrl !== frameUrl)
                    ? fetch(qualityUrl).then(res => res.ok ? res.json() : null).catch(() => null)
                    : Promise.resolve(null)
            ])
            .then(([frameText, boutsText, qualityJson]) => {
                appState.idQuality = qualityJson || null;  // [A-3]
                const parsedFrames = parseFrameCsv(frameText);
                if (parsedFrames.length === 0) throw new Error("Parsed zero records");

                // 大規模データ対策: Math.max(...array) は巨大配列でスプレッド展開すると
                // 引数数が上限を超え "Maximum call stack size exceeded" で落ちるため、
                // ループで最大フレーム番号を求める（数十万行でも安全）。
                let maxFrameIdx = 0;
                for (const r of parsedFrames) { if (r.frame_idx > maxFrameIdx) maxFrameIdx = r.frame_idx; }
                const totalFrames = maxFrameIdx + 1;

                const framesArray = [];
                for (let f = 0; f < totalFrames; f++) {
                    framesArray.push({ frame_idx: f, mice: {} });
                }

                let isMockData = false;
                const serverMiceIds = new Set();

                parsedFrames.forEach(row => {
                    const f = row.frame_idx;
                    const mId = row.mouse_id;
                    if (f < totalFrames && mId >= 1 && mId <= MAX_MICE_LIMIT) {
                        framesArray[f].mice[mId] = {
                            keypoints: row.keypoints,
                            scratch_flag: row.scratch_flag,
                            scratch_prob: row.scratch_prob,
                            id_confidence: row.id_confidence,  // [A-3]
                            features: row.features,
                            disabled: false,
                        };
                        serverMiceIds.add(mId);
                    }
                    if (row.is_mock === 1 || resName.toLowerCase().includes('mock')) {
                        isMockData = true;
                    }
                });

                appState.miceIds = Array.from(serverMiceIds).sort((a, b) => a - b);
                if (appState.miceIds.length === 0) appState.miceIds = [1, 2, 3, 4];

                // Missing frames interpolation (動的マウス数対応)
                for (let f = 0; f < totalFrames; f++) {
                    const frame = framesArray[f];
                    for (const mId of appState.miceIds) {
                        if (!frame.mice[mId] && f > 0 && framesArray[f-1].mice[mId]) {
                            const prev = framesArray[f-1].mice[mId];
                            frame.mice[mId] = {
                                keypoints: _cloneKeypoints(prev.keypoints),
                                scratch_flag: 0, scratch_prob: 0,
                                features: { ...prev.features }, disabled: false,
                            };
                        }
                    }
                }

                appState.data.frames = framesArray;
                appState.data.numFrames = totalFrames;
                
                if (boutsText) {
                    try {
                        appState.data.bouts = parseBoutsCsv(boutsText);
                    } catch (err) {
                        autoGenerateBoutsFromFlags();
                    }
                } else {
                    autoGenerateBoutsFromFlags();
                }

                // Warnings
                if (isMockData) {
                    mockWarningBanner.style.display = "flex";
                    mockWarningBanner.style.alignItems = "center";
                    mockWarningBanner.style.justifyContent = "center";
                    mockWarningBanner.style.gap = "10px";
                } else {
                    mockWarningBanner.style.display = "none";
                }

                finalizeImport(resName);
                renderColorIdQuality();  // [A-3] カラーID品質パネルを更新
            })
            .catch(err => {
                console.error("[Results-Loader] ロードエラー:", err);
                alert(`⚠️ 解析結果の動的ロードに失敗しました:\n${err.message}`);
                if (fileBadge) {
                    fileBadge.textContent = "エラー終了";
                    fileBadge.style.color = "#EF4444";
                }
            });
        });
    }

    if (btnRefreshServerResults) {
        btnRefreshServerResults.addEventListener('click', () => {
            refreshServerResultsList();
            alert("✓ サーバー内の比較結果CSVリストを再スキャンして更新しました！");
        });
    }

    // ==========================================================================
    // --- 12.8 動的マウスUI / Undo・Redo / 無効化マスク / 動画録画 (NEW) ---
    // ==========================================================================

    /**
     * マウス選択 UI とタイムラインチャネルを appState.miceIds に基づいて再構築する。
     *
     * 動的マウス数に対応するため、以下の DOM 要素を都度作り直す:
     *   1) .mouse-selector-list 内の `.mouse-btn`
     *   2) .timeline-channels 内の `.channel-row` / `.channel-track`
     *   3) #swap-mouse-a, #swap-mouse-b, #offset-mouse-id, #invalid-mouse-id の <select>
     *
     * @param {number[]} miceIds - 描画対象マウスID配列（例: [1,2,3,4,5]）
     */
    function initDynamicMiceUI(miceIds) {
        if (!Array.isArray(miceIds) || miceIds.length === 0) {
            miceIds = [1, 2, 3, 4];
        }

        // ── (1) マウスセレクター ─────────────────────────────────────────────
        const selectorList = document.querySelector('.mouse-selector-list');
        if (selectorList) {
            selectorList.innerHTML = '';
            miceIds.forEach((mId, idx) => {
                const palette = (typeof MOUSE_COLOR_PALETTE !== 'undefined' && MOUSE_COLOR_PALETTE[mId])
                    ? MOUSE_COLOR_PALETTE[mId]
                    : { hex: '#AAAAAA', name: `M${mId}`, cssVar: '--text-muted', glowRgb: '170, 170, 170' };
                const btn = document.createElement('button');
                btn.className = 'mouse-btn' + (idx === 0 ? ' active' : '');
                btn.id = `mouse-btn-${mId}`;
                btn.dataset.mouseId = mId;
                btn.style.setProperty('--active-border-color', `var(${palette.cssVar})`);
                btn.style.setProperty('--active-glow-color', `rgba(${palette.glowRgb}, 0.3)`);
                btn.innerHTML = `
                    <div class="mouse-info">
                        <span class="mouse-color-dot" style="color: ${palette.hex}; background-color: currentColor;"></span>
                        <span class="mouse-label">Mouse ${mId} (${palette.name})</span>
                    </div>
                    <span class="mouse-scratch-badge">ひっかき中</span>
                `;
                selectorList.appendChild(btn);
            });
        }

        // ── (2) タイムラインチャネル ─────────────────────────────────────────
        const channelContainer = document.querySelector('.timeline-channels');
        if (channelContainer) {
            channelContainer.innerHTML = '';
            miceIds.forEach((mId) => {
                const palette = (typeof MOUSE_COLOR_PALETTE !== 'undefined' && MOUSE_COLOR_PALETTE[mId])
                    ? MOUSE_COLOR_PALETTE[mId]
                    : { hex: '#AAAAAA', name: `M${mId}` };
                const row = document.createElement('div');
                row.className = 'channel-row';
                row.innerHTML = `
                    <span class="channel-label" style="color: ${palette.hex};">Mouse ${mId}</span>
                    <div class="channel-track" id="track-mouse-${mId}"></div>
                `;
                channelContainer.appendChild(row);
            });
        }

        // ── (3) ID選択用 <select> 群を全て再生成 ─────────────────────────────
        const _rebuildSelect = (sel, defaultIndex = 0) => {
            if (!sel) return;
            sel.innerHTML = '';
            miceIds.forEach((mId) => {
                const palette = (typeof MOUSE_COLOR_PALETTE !== 'undefined' && MOUSE_COLOR_PALETTE[mId])
                    ? MOUSE_COLOR_PALETTE[mId] : { name: `M${mId}` };
                const opt = document.createElement('option');
                opt.value = String(mId);
                opt.textContent = `Mouse ${mId} (${palette.name})`;
                sel.appendChild(opt);
            });
            if (sel.options.length > defaultIndex) sel.selectedIndex = defaultIndex;
        };
        _rebuildSelect(document.getElementById('swap-mouse-a'), 0);
        _rebuildSelect(document.getElementById('swap-mouse-b'), Math.min(1, miceIds.length - 1));
        _rebuildSelect(document.getElementById('offset-mouse-id'), 0);
        _rebuildSelect(document.getElementById('invalid-mouse-id'), 0);

        // ── (4) 選択中マウス整合性チェック ───────────────────────────────────
        if (!miceIds.includes(appState.selectedMouse)) {
            appState.selectedMouse = miceIds[0];
        }
        // active クラスを選択中マウスに合わせる
        const activeBtn = document.getElementById(`mouse-btn-${appState.selectedMouse}`);
        if (activeBtn) {
            document.querySelectorAll('.mouse-btn').forEach(b => b.classList.remove('active'));
            activeBtn.classList.add('active');
        }

        // Lucide アイコン再生成
        if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons();
    }

    /**
     * Undo スタックに現在フレームの mouse スナップショットを保存。
     * 50 件を超えた場合は最古を捨てる。Redo スタックは編集発生でクリアされる。
     *
     * @param {number} frameIdx - 対象フレームインデックス
     * @param {number} mouseId  - 対象マウスID
     */
    function pushUndoSnapshot(frameIdx, mouseId) {
        const frame = appState.data.frames[frameIdx];
        if (!frame || !frame.mice[mouseId]) return;
        const snapshot = JSON.parse(JSON.stringify(frame.mice[mouseId]));
        appState.undoStack.push({ frameIdx, mouseId, snapshot });
        if (appState.undoStack.length > appState.MAX_UNDO_STACK) {
            appState.undoStack.shift();
        }
        appState.redoStack = [];
        markFrameMouseCorrected(frameIdx, mouseId);
        updateUndoRedoButtons();
    }

    /**
     * [A-5] 指定フレーム×個体を「手動補正済み」として記録する。
     * - frame.mice[mouseId]._corrected を立て、エクスポート CSV の is_manually_corrected 列に反映する。
     * - editedFrames（フレーム単位インジケータ用）にも frameIdx を加える。
     * これにより、自動パイプライン出力（プレビュー）と研究者が補正した確定値を
     * 出力データ上で区別でき、学術的なデータ来歴（provenance）を担保する。
     */
    function markFrameMouseCorrected(frameIdx, mouseId) {
        const frame = appState.data.frames[frameIdx];
        if (frame && frame.mice[mouseId]) {
            frame.mice[mouseId]._corrected = true;
        }
        appState.editedFrames.add(frameIdx);
        updateCorrectionStatus();
    }

    /**
     * [A-5] 表示中データの位置づけ（自動プレビュー or 手動補正済み）をUIに明示する。
     * editedFrames が空 → 「自動解析プレビュー（未補正）」を青で表示。
     * 補正があれば → 「手動補正済み（N フレーム）」を緑で表示し、出力CSVが確定値である旨を伝える。
     */
    function updateCorrectionStatus() {
        const box = document.getElementById('correction-status');
        const txt = document.getElementById('correction-status-text');
        if (!box || !txt) return;
        const n = appState.editedFrames ? appState.editedFrames.size : 0;
        if (n > 0) {
            txt.textContent = `手動補正済み（${n} フレーム）— 出力CSVは研究者確定値`;
            box.style.background = 'rgba(16,185,129,0.10)';
            box.style.borderColor = 'rgba(16,185,129,0.5)';
            box.style.color = '#10B981';
            const icon = box.querySelector('[data-lucide]');
            if (icon) { icon.setAttribute('data-lucide', 'check-circle'); if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons(); }
        } else {
            txt.textContent = '自動解析プレビュー（未補正）';
            box.style.background = 'rgba(56,189,248,0.08)';
            box.style.borderColor = 'rgba(56,189,248,0.4)';
            box.style.color = '#38BDF8';
            const icon = box.querySelector('[data-lucide]');
            if (icon) { icon.setAttribute('data-lucide', 'eye'); if (typeof lucide !== 'undefined' && lucide.createIcons) lucide.createIcons(); }
        }
        updateStepGuide();
    }

    /**
     * [B-1] ステップ型ワークフローガイドを現在の状態に同期する。
     * 観測可能な appState から各ステップの完了状況を判定し、
     * 「完了（緑）」「現在（青・最初の未完了）」「未着手（灰）」を表示する。
     *   1 データ準備   : 実データロード or 動画ロード済み
     *   2 校正         : 実測キャリブレーションを適用済み
     *   3 ID確認       : 研究者が track→個体マッピングを確定済み
     *   4 解析         : 自動解析結果（実データ）をロード済み
     *   5 補正         : 手動補正フレームが1件以上
     *   6 出力         : CSV/動画をエクスポート済み
     */
    function updateStepGuide() {
        const guide = document.getElementById('step-guide');
        if (!guide) return;
        const done = {
            data:       !!(appState.hasRealData || appState.videoW > 0),
            calib:      !!appState.calibrationApplied,
            idconfirm:  !!(appState.confirmedMapping && Object.keys(appState.confirmedMapping).length > 0),
            analysis:   !!appState.hasRealData,
            correction: !!(appState.editedFrames && appState.editedFrames.size > 0),
            export:     !!appState.exported,
        };
        const order = ['data', 'calib', 'idconfirm', 'analysis', 'correction', 'export'];
        // 現在ステップ = 「最後に完了したステップの次」。
        // サーバ保存済み結果のロード（解析=完了だが校正/ID確認はダッシュボード外で実施）でも、
        // 直感に合うよう「次に行うべきステップ」を強調する。
        let lastDoneIdx = -1;
        order.forEach((s, i) => { if (done[s]) lastDoneIdx = i; });
        const currentStep = (lastDoneIdx >= 0 && lastDoneIdx < order.length - 1)
            ? order[lastDoneIdx + 1]
            : (lastDoneIdx < 0 ? order[0] : null);
        order.forEach(s => {
            const el = guide.querySelector(`.step-item[data-step="${s}"]`);
            if (!el) return;
            el.classList.toggle('step-done', done[s]);
            el.classList.toggle('step-current', s === currentStep);
        });
    }

    /**
     * Undo: 直近の操作を取り消し、現在状態を Redo スタックへ移す。
     */
    function undo() {
        if (appState.undoStack.length === 0) return;
        const entry = appState.undoStack.pop();
        const frame = appState.data.frames[entry.frameIdx];
        if (!frame) { updateUndoRedoButtons(); return; }

        // 現在状態を Redo へ退避
        const currentSnap = frame.mice[entry.mouseId]
            ? JSON.parse(JSON.stringify(frame.mice[entry.mouseId])) : null;
        appState.redoStack.push({ frameIdx: entry.frameIdx, mouseId: entry.mouseId, snapshot: currentSnap });

        // 復元
        frame.mice[entry.mouseId] = entry.snapshot;

        // 速度等は周辺フレーム再計算が必要
        const start = Math.max(1, entry.frameIdx - 15);
        const end   = Math.min(appState.data.numFrames - 1, entry.frameIdx + 15);
        for (let f = start; f <= end; f++) {
            recalculateFrameKinematics(f, entry.mouseId);
        }
        autoGenerateBoutsFromFlags();
        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);

        appState.currentFrame = entry.frameIdx;
        updateUIForFrame(false);
        updateUndoRedoButtons();
    }

    /**
     * Redo: 取り消した操作をやり直し、現在状態を Undo スタックへ戻す。
     */
    function redo() {
        if (appState.redoStack.length === 0) return;
        const entry = appState.redoStack.pop();
        const frame = appState.data.frames[entry.frameIdx];
        if (!frame || !entry.snapshot) { updateUndoRedoButtons(); return; }

        const currentSnap = frame.mice[entry.mouseId]
            ? JSON.parse(JSON.stringify(frame.mice[entry.mouseId])) : null;
        appState.undoStack.push({ frameIdx: entry.frameIdx, mouseId: entry.mouseId, snapshot: currentSnap });

        frame.mice[entry.mouseId] = entry.snapshot;

        const start = Math.max(1, entry.frameIdx - 15);
        const end   = Math.min(appState.data.numFrames - 1, entry.frameIdx + 15);
        for (let f = start; f <= end; f++) {
            recalculateFrameKinematics(f, entry.mouseId);
        }
        autoGenerateBoutsFromFlags();
        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);

        appState.currentFrame = entry.frameIdx;
        updateUIForFrame(false);
        updateUndoRedoButtons();
    }

    /**
     * Undo/Redo ボタンの有効/無効状態をスタック深度に従って更新。
     */
    function updateUndoRedoButtons() {
        if (btnUndo) {
            btnUndo.disabled = appState.undoStack.length === 0;
            btnUndo.title = `Undo (Ctrl+Z) — 履歴: ${appState.undoStack.length}/${appState.MAX_UNDO_STACK}`;
        }
        if (btnRedo) {
            btnRedo.disabled = appState.redoStack.length === 0;
            btnRedo.title = `Redo (Ctrl+Y) — 再実行可能: ${appState.redoStack.length}`;
        }
    }

    /**
     * 現フレームにおける指定マウスの「無効（誤検出）」フラグを切り替える。
     * 無効化されたマウスは scratch 判定とアラームから除外され、可視化上は灰色＋[除外]表示。
     */
    function toggleMouseInvalid(mouseId) {
        const frame = appState.data.frames[appState.currentFrame];
        if (!frame || !frame.mice[mouseId]) {
            alert(`⚠️ 現フレームに Mouse ${mouseId} のデータが存在しません。`);
            return;
        }
        pushUndoSnapshot(appState.currentFrame, mouseId);
        frame.mice[mouseId].disabled = !frame.mice[mouseId].disabled;
        if (frame.mice[mouseId].disabled) {
            // 無効化時はスクラッチ判定もクリア
            frame.mice[mouseId].scratch_flag = 0;
            frame.mice[mouseId].scratch_prob = 0.0;
        }
        autoGenerateBoutsFromFlags();
        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);
        updateUIForFrame(false);
    }

    /**
     * 注釈付き動画の録画 (MediaRecorder API)。
     * オーバーレイ canvas を MediaStream で取得し、再生中の bgVideo と同期して
     * 全フレームを WebM (VP9) で書き出す。
     */
    function exportAnnotatedVideo() {
        const canvas = document.getElementById('cage-canvas');
        if (!canvas) {
            alert("⚠️ オーバーレイキャンバスが見つかりません。");
            return;
        }
        if (typeof MediaRecorder === 'undefined') {
            alert("⚠️ お使いのブラウザは MediaRecorder API に対応していません。Chrome/Edge の最新版をご使用ください。");
            return;
        }
        if (appState.mediaRecorder && appState.mediaRecorder.state === 'recording') {
            // 録画中: 停止
            appState.mediaRecorder.stop();
            return;
        }

        // FPS は動画 fps か既定値を使用
        const fps = (appState.data && appState.data.fps) || SCRATCH_THRESHOLDS.fps || 30;
        let stream;
        try {
            stream = canvas.captureStream(fps);
        } catch (e) {
            alert(`⚠️ canvas.captureStream に失敗しました: ${e.message}`);
            return;
        }

        // ブラウザサポートの mime を順に試行
        const mimeCandidates = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm'
        ];
        let mimeType = '';
        for (const m of mimeCandidates) {
            if (MediaRecorder.isTypeSupported(m)) { mimeType = m; break; }
        }
        if (!mimeType) {
            alert("⚠️ サポート可能な WebM コーデックが見つかりません。");
            return;
        }

        appState.recordedChunks = [];
        try {
            appState.mediaRecorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 4_000_000 });
        } catch (e) {
            alert(`⚠️ MediaRecorder 生成失敗: ${e.message}`);
            return;
        }

        appState.mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) appState.recordedChunks.push(e.data);
        };
        appState.mediaRecorder.onstop = () => {
            const blob = new Blob(appState.recordedChunks, { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const ts = new Date().toISOString().replace(/[:.]/g, '-');
            a.download = `mouse_scratch_annotated_${ts}.webm`;
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
            URL.revokeObjectURL(url);
            btnExportVideo.innerHTML = '<i data-lucide="video" style="width:16px; height:16px; margin-right:5px;"></i>注釈付き動画を録画 (WebM)';
            btnExportVideo.style.background = 'linear-gradient(135deg, #A855F7 0%, #7C3AED 100%)';
            if (typeof lucide !== 'undefined') lucide.createIcons();
        };

        appState.mediaRecorder.start(250);  // 250ms ごとに dataavailable
        btnExportVideo.innerHTML = '<i data-lucide="square" style="width:16px; height:16px; margin-right:5px;"></i>録画停止';
        btnExportVideo.style.background = 'linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)';
        if (typeof lucide !== 'undefined') lucide.createIcons();

        // 動画が再生中でなければ自動再生（描画ループ駆動のため）
        if (!appState.isPlaying) togglePlay();

        // 動画終端到達で自動停止
        const stopOnEnd = () => {
            if (appState.mediaRecorder && appState.mediaRecorder.state === 'recording') {
                appState.mediaRecorder.stop();
            }
            if (bgVideo) bgVideo.removeEventListener('ended', stopOnEnd);
        };
        if (bgVideo) bgVideo.addEventListener('ended', stopOnEnd);
    }

    // ── イベントバインド: Undo/Redo/Invalid/Video Export ──────────────────────
    if (btnUndo) btnUndo.addEventListener('click', undo);
    if (btnRedo) btnRedo.addEventListener('click', redo);
    if (btnToggleInvalid) {
        btnToggleInvalid.addEventListener('click', () => {
            const mId = parseInt(invalidMouseId.value);
            if (!isNaN(mId)) toggleMouseInvalid(mId);
        });
    }
    if (btnExportVideo) btnExportVideo.addEventListener('click', exportAnnotatedVideo);

    // ── キーボードショートカット (Ctrl+Z / Ctrl+Y) ───────────────────────────
    document.addEventListener('keydown', (e) => {
        // テキスト入力中は無効化
        const tag = (document.activeElement && document.activeElement.tagName) || '';
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'z') {
            e.preventDefault(); undo();
        } else if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' ||
                   (e.shiftKey && e.key.toLowerCase() === 'z'))) {
            e.preventDefault(); redo();
        } else if (!e.ctrlKey && !e.metaKey && !e.altKey) {
            // ── B-3 Bout レビュー キーボードショートカット ──────────────────
            // N=次, P=前, A=承認, R=却下。修飾キー無しのときのみ反応。
            const k = e.key.toLowerCase();
            if (k === 'n') { e.preventDefault(); gotoNextBout(); }
            else if (k === 'p') { e.preventDefault(); gotoPrevBout(); }
            else if (k === 'a') { e.preventDefault(); acceptCurrentBout(); }
            else if (k === 'r') { e.preventDefault(); rejectCurrentBout(); }
        }
    });

    // ── B-3 Bout レビュー ツールバー ボタン配線 ──────────────────────────────
    (function wireBoutReviewBar() {
        const btnPrev = document.getElementById('btn-bout-prev');
        const btnNext = document.getElementById('btn-bout-next');
        const btnAcc = document.getElementById('btn-bout-accept');
        const btnRej = document.getElementById('btn-bout-reject');
        if (btnPrev) btnPrev.addEventListener('click', gotoPrevBout);
        if (btnNext) btnNext.addEventListener('click', gotoNextBout);
        if (btnAcc) btnAcc.addEventListener('click', acceptCurrentBout);
        if (btnRej) btnRej.addEventListener('click', rejectCurrentBout);
    })();

    // ── 閾値エディタ: ライブプレビュー対応スライダー & 適用ボタン (B-4) ──────────
    const thrMinVelRange  = document.getElementById('thr-min-vel-mms-range');
    const thrMaxHeadRange = document.getElementById('thr-max-head-mm-range');
    const thrMinOscRange  = document.getElementById('thr-min-osc-range');
    const thrMinAmpRange  = document.getElementById('thr-min-osc-amp-range');
    const thrLiveText     = document.getElementById('thr-live-text');
    const thrLiveReadout  = document.getElementById('thr-live-readout');

    // スライダー・数値入力・値ラベルを SCRATCH_THRESHOLDS の現在値に同期する。
    function syncThresholdInputs() {
        const set = (numEl, rangeEl, valId, v) => {
            if (numEl) numEl.value = v;
            if (rangeEl) rangeEl.value = v;
            const valEl = document.getElementById(valId);
            if (valEl) valEl.textContent = v;
        };
        set(thrMinVelMms, thrMinVelRange,  'thr-min-vel-mms-val', SCRATCH_THRESHOLDS.minVelocityMmS);
        set(thrMaxHeadMm, thrMaxHeadRange, 'thr-max-head-mm-val', SCRATCH_THRESHOLDS.maxHeadDistMm);
        set(thrMinOsc,    thrMinOscRange,  'thr-min-osc-val',     SCRATCH_THRESHOLDS.minOscillations);
        set(thrMinOscAmp, thrMinAmpRange,  'thr-min-osc-amp-val', SCRATCH_THRESHOLDS.minOscAmplitudeMm);
    }
    syncThresholdInputs();

    // 現在の入力値を検証して取得する。不正な場合は null を返す。
    function readThresholdInputs() {
        const newMinVel  = parseFloat(thrMinVelMms.value);
        const newMaxHead = parseFloat(thrMaxHeadMm.value);
        const newMinOsc  = parseInt(thrMinOsc.value);
        const newMinAmp  = parseFloat(thrMinOscAmp.value);
        if ([newMinVel, newMaxHead, newMinOsc, newMinAmp].some(v => isNaN(v) || v < 0)) return null;
        return { newMinVel, newMaxHead, newMinOsc, newMinAmp };
    }

    // 閾値を SCRATCH_THRESHOLDS に反映し、全フレームを再計算して Bout・チャート・UI を更新する。
    // ※ feature_extraction.py と同一の recalculateFrameKinematics を用いるため、
    //   GUI プレビューとバックエンド本解析の判定結果は乖離しない（CLAUDE.md 同期ルール）。
    function applyThresholdsLocal({ newMinVel, newMaxHead, newMinOsc, newMinAmp }) {
        SCRATCH_THRESHOLDS.minVelocityMmS    = newMinVel;
        SCRATCH_THRESHOLDS.maxHeadDistMm     = newMaxHead;
        SCRATCH_THRESHOLDS.minOscillations   = newMinOsc;
        SCRATCH_THRESHOLDS.minOscAmplitudeMm = newMinAmp;
        // frame 0 は前フレームが無く速度算出不能のため f=1 から再計算する。
        for (let f = 1; f < appState.data.numFrames; f++) {
            for (const mId of appState.miceIds) {
                recalculateFrameKinematics(f, mId);
            }
        }
        autoGenerateBoutsFromFlags();
        renderTimelineBouts();
        renderBoutsList();
        charts.updateAnalytics(appState.data.bouts);
        updateUIForFrame(false);
    }

    // B-4: ライブプレビュー（デバウンス）。スライダー/数値入力の変更に追従して
    // フロント側のみ即時再計算し、検出 Bout 数を読み取り表示に反映する。
    // サーバ config への送信は行わない（確定は適用ボタンで実施）。
    let thrLiveTimer = null;
    function scheduleLivePreview() {
        if (thrLiveTimer) clearTimeout(thrLiveTimer);
        thrLiveTimer = setTimeout(() => {
            const vals = readThresholdInputs();
            if (!vals) { if (thrLiveText) thrLiveText.textContent = '⚠ 値が不正です'; return; }
            applyThresholdsLocal(vals);
            if (thrLiveText) thrLiveText.textContent = `検出 Bout: ${appState.data.bouts.length} 件（プレビュー）`;
            if (thrLiveReadout) {
                thrLiveReadout.classList.add('flash');
                setTimeout(() => thrLiveReadout.classList.remove('flash'), 150);
            }
        }, 200);
    }

    // スライダー ↔ 数値入力 ↔ 値ラベル の双方向同期 + ライブプレビュー起動。
    function wireThresholdPair(numEl, rangeEl, valId) {
        if (!numEl || !rangeEl) return;
        const valEl = document.getElementById(valId);
        rangeEl.addEventListener('input', () => {
            numEl.value = rangeEl.value;
            if (valEl) valEl.textContent = rangeEl.value;
            scheduleLivePreview();
        });
        numEl.addEventListener('input', () => {
            rangeEl.value = numEl.value;
            if (valEl) valEl.textContent = numEl.value;
            scheduleLivePreview();
        });
    }
    wireThresholdPair(thrMinVelMms, thrMinVelRange,  'thr-min-vel-mms-val');
    wireThresholdPair(thrMaxHeadMm, thrMaxHeadRange, 'thr-max-head-mm-val');
    wireThresholdPair(thrMinOsc,    thrMinOscRange,  'thr-min-osc-val');
    wireThresholdPair(thrMinOscAmp, thrMinAmpRange,  'thr-min-osc-amp-val');

    if (btnApplyThresholds) {
        btnApplyThresholds.addEventListener('click', () => {
            const vals = readThresholdInputs();
            if (!vals) { alert("⚠️ 閾値は0以上の数値を入力してください。"); return; }
            applyThresholdsLocal(vals);

            // バックエンド config へ確定値を同期 (失敗してもフロント側計算は続行)
            fetch('/api/config-update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    SCRATCH_MIN_VELOCITY_MM_S:    vals.newMinVel,
                    SCRATCH_MAX_HEAD_DIST_MM:     vals.newMaxHead,
                    SCRATCH_MIN_OSCILLATIONS:     vals.newMinOsc,
                    SCRATCH_MIN_OSC_AMPLITUDE_MM: vals.newMinAmp,
                })
            }).catch(err => console.warn('[Config] サーバ閾値同期失敗:', err));

            if (thrLiveText) thrLiveText.textContent = `検出 Bout: ${appState.data.bouts.length} 件（確定・サーバ同期済）`;
            alert(`✓ 閾値を適用してサーバへ同期しました。\n  最小速度: ${vals.newMinVel} mm/s\n  後肢↔耳 間距離上限: ${vals.newMaxHead} mm\n  最小振動: ${vals.newMinOsc}\n  最小振幅: ${vals.newMinAmp} mm\n  検出 Bout: ${appState.data.bouts.length} 件`);
        });
    }

    // ── A-2 実測校正ウィザード: 動画上の既知長を2点クリックして mm/px を実測 ──────
    // 計測は canvas 内部座標系（= 動画ピクセル空間）で行う。resizeCanvasToVideo() により
    // canvas.width/height を動画ネイティブ解像度に合わせることで、getMousePos が返す座標が
    // 動画上の真のピクセル座標に一致し、pipeline.py の px-to-mm と整合する。
    (function initCalibrationWizard() {
        const btnMeasure   = document.getElementById('btn-calibrate-measure');
        const statusBox     = document.getElementById('calibrate-status');
        const statusText    = document.getElementById('calibrate-status-text');
        const lengthRow     = document.getElementById('calibrate-length-row');
        const lengthInput   = document.getElementById('calibrate-length-mm');
        const btnApply      = document.getElementById('btn-calibrate-apply');
        const btnReset      = document.getElementById('btn-calibrate-reset');
        const resultBox     = document.getElementById('calibrate-result');
        if (!btnMeasure || !statusBox) return;

        // 計測のために canvas を動画解像度へ強制リサイズした場合に元へ戻すためのフラグ
        let resizedForCalib = false;

        function endMeasureMode() {
            visualizer.setMeasureMode(false);
            statusBox.style.display = 'none';
            lengthRow.style.display = 'none';
            resultBox.style.display = 'none';
            btnMeasure.innerHTML = '<i data-lucide="crosshair" style="width:12px; height:12px;"></i>2点計測を開始';
            if (typeof lucide !== 'undefined') lucide.createIcons();
            // 解析データが無い状態で canvas を動画解像度に変更していた場合はデフォルトへ戻す
            if (resizedForCalib && !appState.hasRealData) {
                resetCanvasToDefault();
                updateUIForFrame(false);
            }
            resizedForCalib = false;
        }

        btnMeasure.addEventListener('click', () => {
            if (visualizer.measureMode) { endMeasureMode(); return; }

            if (!appState.videoW || !appState.videoH) {
                alert('⚠️ 先に動画を読み込んでください。\n実測校正は動画上の既知長を計測するため、動画の表示が必要です。');
                return;
            }
            // canvas 内部解像度を動画ピクセル空間に合わせる（解析前でも計測可能にする）
            if (visualizer.canvas.width !== appState.videoW || visualizer.canvas.height !== appState.videoH) {
                resizeCanvasToVideo();
                resizedForCalib = true;
            }

            visualizer.setMeasureMode(true);
            statusBox.style.display = 'block';
            resultBox.style.display = 'none';
            lengthRow.style.display = 'none';
            statusText.textContent = '動画上で1点目（既知長の一端）をクリックしてください。';
            btnMeasure.innerHTML = '<i data-lucide="x" style="width:12px; height:12px;"></i>計測を終了';
            if (typeof lucide !== 'undefined') lucide.createIcons();
        });

        // 計測点が変化するたびに呼ばれる
        visualizer.onMeasureUpdate = (points) => {
            if (points.length === 0) {
                statusText.textContent = '動画上で1点目（既知長の一端）をクリックしてください。';
                lengthRow.style.display = 'none';
                resultBox.style.display = 'none';
            } else if (points.length === 1) {
                statusText.textContent = '2点目（既知長のもう一端）をクリックしてください。';
                lengthRow.style.display = 'none';
            } else {
                const distPx = visualizer.getMeasuredPixelDistance();
                statusText.textContent = `2点間のピクセル距離: ${distPx.toFixed(1)} px。実距離を入力して「適用」を押してください。`;
                lengthRow.style.display = 'flex';
                lengthInput.focus();
            }
        };

        if (btnReset) {
            btnReset.addEventListener('click', () => visualizer.clearMeasurePoints());
        }

        if (btnApply) {
            btnApply.addEventListener('click', () => {
                const distPx = visualizer.getMeasuredPixelDistance();
                if (distPx == null || distPx < 1) {
                    alert('⚠️ 2点が近すぎるか未計測です。動画上で2点をクリックし直してください。');
                    return;
                }
                const realMm = parseFloat(lengthInput.value);
                if (isNaN(realMm) || realMm <= 0) {
                    alert('⚠️ 実距離（mm）に正の数値を入力してください。');
                    return;
                }
                // mm/px = 実距離(mm) ÷ ピクセル距離(px)
                const ratio = realMm / distPx;

                // フロント側の校正値を更新
                SCRATCH_THRESHOLDS.pixelToMmRatio = ratio;
                if (pipelineCalibration) pipelineCalibration.value = ratio.toFixed(4);
                appState.calibrationApplied = true;  // [B-1] 校正ステップ完了
                if (typeof updateStepGuide === 'function') updateStepGuide();

                // バックエンドのランタイム config にも反映（失敗してもフロント計算は続行）
                fetch('/api/config-update', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ PIXEL_TO_MM_RATIO: ratio })
                }).catch(err => console.warn('[Calib] サーバ校正値同期失敗:', err));

                // 校正値変更で mm/s・mm 特徴量が変わるため全フレーム再計算
                for (let f = 1; f < appState.data.numFrames; f++) {
                    for (const mId of appState.miceIds) {
                        recalculateFrameKinematics(f, mId);
                    }
                }
                autoGenerateBoutsFromFlags();
                renderTimelineBouts();
                renderBoutsList();
                charts.updateAnalytics(appState.data.bouts);

                resultBox.style.display = 'block';
                resultBox.textContent = `✓ 校正完了: ${ratio.toFixed(4)} mm/px （${realMm} mm ÷ ${distPx.toFixed(1)} px）`;

                // 計測モードを終了（canvas を元の状態へ戻す）
                endMeasureMode();
                // endMeasureMode が statusBox を隠すので、結果だけ再表示する
                statusBox.style.display = 'block';
                statusText.textContent = '校正値を適用しました。再計測する場合は「2点計測を開始」を押してください。';
                resultBox.style.display = 'block';
                updateUIForFrame(false);
            });
        }
    })();

    // ── A-1 精度評価 (Ground Truth 照合) パネル ──────────────────────────────
    // /api/evaluate を呼び出し、フレーム/Bout レベルの混同行列・各種指標を表示する。
    // サーバーは CSV をパスで読み込むため、自動検出 CSV は /api/results 由来の
    // outputs/ 相対パス、GT CSV はユーザー入力のパス（絶対 or outputs/相対）を渡す。
    (function initEvaluationPanel() {
        const autoSelect  = document.getElementById('eval-auto-csv');
        const gtPathInput = document.getElementById('eval-gt-path');
        const gtFormatSel  = document.getElementById('eval-gt-format');
        const fpsInput     = document.getElementById('eval-fps');
        const iouInput     = document.getElementById('eval-iou');
        const btnRun       = document.getElementById('btn-run-evaluate');
        const btnRefresh   = document.getElementById('btn-eval-refresh');
        const statusBox    = document.getElementById('eval-status');
        const frameWrap    = document.getElementById('eval-frame-results');
        const boutWrap     = document.getElementById('eval-bout-results');
        const frameTable   = document.getElementById('eval-frame-table');
        const boutTable    = document.getElementById('eval-bout-table');
        if (!btnRun || !autoSelect) return;

        // URL "/outputs/foo/bar.csv" → evaluate が解決できる "foo/bar.csv" に変換
        function urlToOutputsPath(url) {
            if (!url) return null;
            return url.replace(/^\/?outputs\//, '');
        }

        function populateAutoResults() {
            fetch('/api/results')
                .then(r => r.json())
                .then(data => {
                    autoSelect.innerHTML = '<option value="" selected>-- サーバー結果から選択 --</option>';
                    (data.results || []).forEach(res => {
                        const opt = document.createElement('option');
                        opt.value = urlToOutputsPath(res.frame_csv_url);
                        opt.dataset.boutsPath = urlToOutputsPath(res.bouts_csv_url) || '';
                        opt.textContent = res.name;
                        autoSelect.appendChild(opt);
                    });
                })
                .catch(err => console.warn('[Eval] 結果リスト取得失敗:', err));
        }
        populateAutoResults();
        if (btnRefresh) btnRefresh.addEventListener('click', populateAutoResults);

        function setStatus(msg, kind) {
            statusBox.style.display = 'block';
            statusBox.textContent = msg;
            const styles = {
                info:    ['rgba(56,189,248,0.1)',  '#38BDF8'],
                error:   ['rgba(239,68,68,0.1)',   '#EF4444'],
                success: ['rgba(16,185,129,0.1)',  '#10B981'],
            };
            const [bg, fg] = styles[kind] || styles.info;
            statusBox.style.background = bg;
            statusBox.style.color = fg;
        }

        // 指標値を 0–1 の良し悪しで色分け（precision/recall/f1/kappa 等の高いほど良い指標用）
        function metricClass(v) {
            if (v == null || isNaN(v)) return '';
            if (v >= 0.8) return 'eval-metric-good';
            if (v >= 0.5) return 'eval-metric-mid';
            return 'eval-metric-bad';
        }
        function fmt(v, digits = 3) {
            return (v == null || isNaN(v)) ? '—' : Number(v).toFixed(digits);
        }
        function mouseLabel(mid) {
            const c = visualizer.mouseColors[mid];
            return c ? `Mouse ${mid} (${c.name})` : `Mouse ${mid}`;
        }
        function mouseColor(mid) {
            const c = visualizer.mouseColors[mid];
            return c ? c.hex : '#AAAAAA';
        }

        function renderFrameTable(rows) {
            if (!rows || rows.length === 0) { frameWrap.style.display = 'none'; return; }
            const head = `<thead><tr>
                <th>個体</th><th>フレーム数</th>
                <th>Precision</th><th>Recall</th><th>F1</th><th>Cohen's κ</th>
                <th>TP</th><th>FP</th><th>FN</th><th>TN</th>
            </tr></thead>`;
            const body = rows.map(r => `<tr>
                <td class="eval-mouse-cell" style="color:${mouseColor(r.mouse_id)}">${mouseLabel(r.mouse_id)}</td>
                <td>${r.n_frames ?? '—'}</td>
                <td class="${metricClass(r.precision)}">${fmt(r.precision)}</td>
                <td class="${metricClass(r.recall)}">${fmt(r.recall)}</td>
                <td class="${metricClass(r.f1)}">${fmt(r.f1)}</td>
                <td class="${metricClass(r.kappa)}">${fmt(r.kappa)}</td>
                <td>${r.tp ?? '—'}</td><td>${r.fp ?? '—'}</td><td>${r.fn ?? '—'}</td><td>${r.tn ?? '—'}</td>
            </tr>`).join('');
            frameTable.innerHTML = head + `<tbody>${body}</tbody>`;
            frameWrap.style.display = 'block';
        }

        function renderBoutTable(rows) {
            if (!rows || rows.length === 0) { boutWrap.style.display = 'none'; return; }
            const head = `<thead><tr>
                <th>個体</th><th>GT Bouts</th><th>自動 Bouts</th>
                <th>TP</th><th>FP</th><th>FN</th>
                <th>感度 (Sensitivity)</th><th>FDR</th><th>Precision</th><th>F1</th>
            </tr></thead>`;
            const body = rows.map(r => `<tr>
                <td class="eval-mouse-cell" style="color:${mouseColor(r.mouse_id)}">${mouseLabel(r.mouse_id)}</td>
                <td>${r.n_gt_bouts ?? '—'}</td><td>${r.n_auto_bouts ?? '—'}</td>
                <td>${r.tp_bouts ?? '—'}</td><td>${r.fp_bouts ?? '—'}</td><td>${r.fn_bouts ?? '—'}</td>
                <td class="${metricClass(r.sensitivity)}">${fmt(r.sensitivity)}</td>
                <td>${fmt(r.fdr)}</td>
                <td class="${metricClass(r.precision_bout)}">${fmt(r.precision_bout)}</td>
                <td class="${metricClass(r.f1_bout)}">${fmt(r.f1_bout)}</td>
            </tr>`).join('');
            boutTable.innerHTML = head + `<tbody>${body}</tbody>`;
            boutWrap.style.display = 'block';
        }

        btnRun.addEventListener('click', () => {
            const autoPath = autoSelect.value;
            const gtPath   = gtPathInput.value.trim();
            if (!autoPath) { setStatus('⚠️ 自動検出結果 CSV を選択してください。', 'error'); return; }
            if (!gtPath)   { setStatus('⚠️ Ground Truth CSV のパスを入力してください。', 'error'); return; }

            const fmtVal = gtFormatSel.value;
            const fps    = parseFloat(fpsInput.value) || 30.0;
            const iou    = parseFloat(iouInput.value) || 0.5;
            // 選択された結果に対応する bouts.csv（あれば bout 指標に使用）
            const opt = autoSelect.options[autoSelect.selectedIndex];
            const boutsPath = opt && opt.dataset.boutsPath ? opt.dataset.boutsPath : null;

            setStatus('評価を実行中...', 'info');
            frameWrap.style.display = 'none';
            boutWrap.style.display = 'none';
            btnRun.disabled = true;

            fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    auto_frame_csv: autoPath,
                    gt_csv: gtPath,
                    gt_format: fmtVal,
                    fps: fps,
                    iou_threshold: iou,
                    auto_bouts_csv: boutsPath,
                })
            })
            .then(res => {
                if (!res.ok) return res.json().then(e => { throw new Error(e.detail || '評価エラー'); });
                return res.json();
            })
            .then(data => {
                renderFrameTable(data.frame_metrics);
                renderBoutTable(data.bout_metrics);
                const nf = (data.frame_metrics || []).length;
                const nb = (data.bout_metrics || []).length;
                const saved = (data.output_files || []).length;
                setStatus(`✓ 評価完了: フレーム指標 ${nf} 個体, Bout 指標 ${nb} 個体。`
                    + (saved ? ` 結果CSVを outputs/ に保存しました（${saved} ファイル）。` : ''), 'success');
                if (nf === 0 && nb === 0) {
                    setStatus('⚠️ 指標が算出されませんでした。GTと自動結果の mouse_id / 列名が一致しているか確認してください。', 'error');
                }
            })
            .catch(err => {
                console.error('[Eval] 評価失敗:', err);
                setStatus(`⚠️ 評価に失敗しました: ${err.message}`, 'error');
            })
            .finally(() => { btnRun.disabled = false; });
        });
    })();

    // ── A-3 カラーID品質管理パネルのレンダリング ────────────────────────────
    // 個体ごとのID信頼度（採用色の得票率）・低信頼フレーム数・Swap（混同）検知件数を
    // 可視化する。品質JSON（{basename}_id_quality.json）が無い場合は、フレームCSVの
    // id_confidence 列から個体別サマリーをクライアント側で再集計してフォールバック表示する。
    function renderColorIdQuality() {
        const panel = document.getElementById('colorid-qc-panel');
        if (!panel) return;
        const summaryWrap = document.getElementById('colorid-qc-summary');
        const swapWrap    = document.getElementById('colorid-qc-swaps');
        const statusBox   = document.getElementById('colorid-qc-status');

        const LOW_THRESH = (appState.idQuality && appState.idQuality.low_conf_threshold) || 0.70;

        // 個体別サマリーを決定: 品質JSON優先、無ければ frame id_confidence から再集計
        let perMouse = {};
        let swapEvents = [];
        let source = 'frames';

        if (appState.idQuality && appState.idQuality.per_mouse) {
            perMouse = appState.idQuality.per_mouse;
            swapEvents = appState.idQuality.swap_events || [];
            source = appState.idQuality.source || 'pipeline';
        } else {
            // フォールバック: ロード済みフレームから id_confidence を集計
            const acc = {};  // mId -> {sum, n, nLow, hasConf}
            const frames = (appState.data && appState.data.frames) || [];
            frames.forEach(fr => {
                for (const mId in fr.mice) {
                    const conf = fr.mice[mId].id_confidence;
                    if (conf == null || isNaN(conf)) continue;
                    const a = acc[mId] || (acc[mId] = { sum: 0, n: 0, nLow: 0 });
                    a.sum += conf; a.n += 1;
                    if (conf < LOW_THRESH) a.nLow += 1;
                }
            });
            for (const mId in acc) {
                const a = acc[mId];
                perMouse[mId] = {
                    mean_confidence: a.n ? a.sum / a.n : 0,
                    n_frames: a.n,
                    n_low_conf_frames: a.nLow,
                };
            }
        }

        const mouseIds = Object.keys(perMouse).map(Number).sort((x, y) => x - y);

        // id_confidence データが全く無い（旧CSV・手動データ等）場合はパネルを隠す
        if (mouseIds.length === 0) {
            panel.style.display = 'none';
            return;
        }
        panel.style.display = 'block';

        function confClass(v) {
            if (v == null || isNaN(v)) return '';
            if (v >= 0.85) return 'eval-metric-good';
            if (v >= LOW_THRESH) return 'eval-metric-mid';
            return 'eval-metric-bad';
        }
        function colorFor(mid) {
            const c = visualizer.mouseColors[mid];
            return c ? c.hex : '#AAAAAA';
        }
        function labelFor(mid) {
            const c = visualizer.mouseColors[mid];
            return c ? `Mouse ${mid} (${c.name})` : `Mouse ${mid}`;
        }

        // 個体別サマリー表
        const rows = mouseIds.map(mid => {
            const q = perMouse[mid];
            const conf = q.mean_confidence;
            const pct = (conf * 100).toFixed(1);
            const low = q.n_low_conf_frames || 0;
            const total = q.n_frames || 0;
            const lowPct = total ? ((low / total) * 100).toFixed(1) : '0.0';
            const warn = (conf < LOW_THRESH || (total && low / total > 0.2));
            return `<tr>
                <td class="eval-mouse-cell" style="color:${colorFor(mid)}">${labelFor(mid)}</td>
                <td class="${confClass(conf)}">${pct}%</td>
                <td>${low} / ${total} (${lowPct}%)</td>
                <td>${warn ? '<span style="color:#EF4444;">⚠ 要確認</span>' : '<span style="color:#10B981;">✓ 良好</span>'}</td>
            </tr>`;
        }).join('');

        summaryWrap.innerHTML = `<table class="eval-table">
            <thead><tr>
                <th>個体</th><th>平均ID信頼度</th>
                <th>低信頼フレーム (&lt;${(LOW_THRESH * 100).toFixed(0)}%)</th><th>判定</th>
            </tr></thead><tbody>${rows}</tbody></table>`;

        // Swap（混同）イベント一覧
        if (swapEvents.length > 0) {
            const fps = (appState.data && appState.data.fps) || 30;
            const items = swapEvents.map(ev => {
                const t = (ev.frame_idx / fps).toFixed(1);
                return `<li>フレーム <strong>${ev.frame_idx}</strong> (${t}s): トラック <code>${ev.orig_track}</code> で色変化を検知し自動分割</li>`;
            }).join('');
            swapWrap.innerHTML = `<div class="qc-swap-title">⚠ Identity Swap（個体取り違え）検知: ${swapEvents.length} 件</div>
                <ul class="qc-swap-list">${items}</ul>
                <div class="qc-swap-note">これらの時点付近では自動トラッキングが個体を取り違えた可能性があります。手動でID割り当てを確認してください。</div>`;
            swapWrap.style.display = 'block';
        } else {
            swapWrap.innerHTML = '';
            swapWrap.style.display = 'none';
        }

        if (statusBox) {
            const srcLabel = source === 'confirmed_mapping'
                ? '（研究者が確認したIDマッピングに基づく — 信頼度は最高値）'
                : (source === 'frames' ? '（フレームCSVから再集計）' : '（パイプライン品質レポートに基づく）');
            statusBox.textContent = `ID品質サマリー ${srcLabel}`;
        }
    }

    // --- 13. Initial Kickoff ---
    // デモデータの初期マウスIDを appState に同期
    if (appState.data && appState.data.frames && appState.data.frames[0]) {
        const initialMice = Object.keys(appState.data.frames[0].mice).map(n => parseInt(n)).sort((a, b) => a - b);
        if (initialMice.length > 0) appState.miceIds = initialMice;
    }
    // スクラブスライダー & 総時間表示をデモデータに合わせて初期化
    // （HTML 側のハードコード値を削除したため、JS 側で動的にセットする）
    if (appState.data && appState.data.numFrames > 0) {
        scrubSlider.max = appState.data.numFrames - 1;
        const _demoDurSec = appState.data.numFrames / (appState.data.fps || 30);
        timeTotal.textContent = `/ ${formatTime(_demoDurSec)}`;
    }
    initDynamicMiceUI(appState.miceIds);
    updateUndoRedoButtons();
    updateCorrectionStatus();  // [A-5] 初期表示はプレビュー状態
    updateUIForFrame();
    renderTimelineBouts();
    renderBoutsList();
    refreshServerResultsList(); // Auto-load server results list on kickoff

    // ── Layer 1: ケージマスク編集 ロジック ────────────────────────────────
    // 装置領域を多角形で塗り、検出後フィルタで装置の誤検出を除外する。
    // 動画ピクセル座標で多角形を保持し、サーバへ JSON で送信する。
    // CSS 表示サイズと動画解像度の差異を考慮した座標変換を行う点に注意。
    initCageMaskUI();

    function initCageMaskUI() {
        const btnEdit       = document.getElementById('btn-edit-cage-mask');
        const modal         = document.getElementById('cage-mask-modal');
        const maskCanvas    = document.getElementById('cage-mask-canvas');
        const statusLabel   = document.getElementById('cage-mask-status');
        const btnFinishPoly = document.getElementById('btn-mask-finish-poly');
        const btnUndo       = document.getElementById('btn-mask-undo');
        const btnClear      = document.getElementById('btn-mask-clear');
        const btnSave       = document.getElementById('btn-mask-save');
        const btnCancel     = document.getElementById('btn-mask-cancel');
        if (!btnEdit || !modal || !maskCanvas) return;
        const ctx = maskCanvas.getContext('2d');

        // 編集開始: 既存マスクがあればロードし、モーダルを表示
        btnEdit.addEventListener('click', async () => {
            if (!appState.videoW || !appState.videoH) {
                alert('動画をロードしてからケージマスクを編集してください。');
                return;
            }
            const videoName = (btnRunPipeline.dataset.videoName || 'unnamed.mp4');
            appState.cageMask.videoName = videoName;
            appState.cageMask.inProgress = [];
            // サーバから既存マスクを取得
            try {
                const stem = videoName.replace(/\.[^/.]+$/, '');
                const res = await fetch(`/api/cage_mask/${encodeURIComponent(stem)}`);
                if (res.ok) {
                    const data = await res.json();
                    appState.cageMask.polygons = (data.exclude_polygons || []).map(poly =>
                        poly.map(pt => [pt[0], pt[1]])
                    );
                } else {
                    appState.cageMask.polygons = [];
                }
            } catch (err) {
                console.warn('[CageMask] 既存マスク取得失敗 (新規作成として続行):', err);
                appState.cageMask.polygons = [];
            }
            openMaskEditor();
        });

        function openMaskEditor() {
            appState.cageMask.active = true;
            modal.style.display = 'block';
            resizeMaskCanvas();
            drawMaskCanvas();
            // 動画は一時停止（フレームを固定して編集しやすくする）
            if (appState.isPlaying && typeof togglePlay === 'function') togglePlay();
        }

        function closeMaskEditor() {
            appState.cageMask.active = false;
            appState.cageMask.inProgress = [];
            modal.style.display = 'none';
        }

        function resizeMaskCanvas() {
            // 内部解像度を動画解像度に合わせ、CSS は親要素いっぱいに広げる
            // これにより polygon 座標が動画ピクセル空間で扱える
            const vW = appState.videoW || 800;
            const vH = appState.videoH || 550;
            if (maskCanvas.width !== vW || maskCanvas.height !== vH) {
                maskCanvas.width = vW;
                maskCanvas.height = vH;
            }
            // CSS サイズは bg-video の object-fit:contain と合わせる必要があるため、
            // cage-canvas と同じレターボックス位置に揃える
            const cage = document.getElementById('cage-canvas');
            if (cage) {
                maskCanvas.style.left   = cage.style.left   || '0';
                maskCanvas.style.top    = cage.style.top    || '0';
                maskCanvas.style.width  = cage.style.width  || '100%';
                maskCanvas.style.height = cage.style.height || '100%';
            }
        }

        // マウス座標 → canvas 内部座標 (= 動画ピクセル座標)
        function getCanvasPoint(evt) {
            const rect = maskCanvas.getBoundingClientRect();
            const sx = maskCanvas.width  / rect.width;
            const sy = maskCanvas.height / rect.height;
            return [
                Math.max(0, Math.min(maskCanvas.width,  (evt.clientX - rect.left) * sx)),
                Math.max(0, Math.min(maskCanvas.height, (evt.clientY - rect.top ) * sy)),
            ];
        }

        function drawMaskCanvas() {
            ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

            // 確定済み多角形（赤の半透明塗り + 実線縁取り）
            ctx.save();
            for (const poly of appState.cageMask.polygons) {
                if (poly.length < 3) continue;
                ctx.beginPath();
                ctx.moveTo(poly[0][0], poly[0][1]);
                for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
                ctx.closePath();
                ctx.fillStyle = 'rgba(239, 68, 68, 0.28)';
                ctx.fill();
                ctx.strokeStyle = '#EF4444';
                ctx.lineWidth = 2;
                ctx.stroke();
            }
            ctx.restore();

            // 編集中の多角形（緑の点線 + 頂点マーカー）
            const ip = appState.cageMask.inProgress;
            if (ip.length > 0) {
                ctx.save();
                ctx.strokeStyle = '#22C55E';
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.moveTo(ip[0][0], ip[0][1]);
                for (let i = 1; i < ip.length; i++) ctx.lineTo(ip[i][0], ip[i][1]);
                ctx.stroke();
                ctx.setLineDash([]);
                // 各頂点を丸で表示
                for (const [px, py] of ip) {
                    ctx.beginPath();
                    ctx.arc(px, py, 5, 0, Math.PI * 2);
                    ctx.fillStyle = '#22C55E';
                    ctx.fill();
                    ctx.strokeStyle = '#FFFFFF';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
                ctx.restore();
            }

            // ステータス更新
            const nPoly = appState.cageMask.polygons.length;
            const nPoints = ip.length;
            if (nPoints > 0) {
                statusLabel.textContent = `編集中: ${nPoints} 点配置 (3点以上で確定可能) / 確定済み多角形: ${nPoly}`;
            } else if (nPoly > 0) {
                statusLabel.textContent = `確定済み多角形: ${nPoly} / 装置の角をクリックして新しい多角形を追加`;
            } else {
                statusLabel.textContent = '装置の角をクリックして多角形を作成 → ダブルクリック or Enter で確定';
            }
        }

        function finishCurrentPolygon() {
            if (appState.cageMask.inProgress.length >= 3) {
                appState.cageMask.polygons.push(appState.cageMask.inProgress.slice());
                appState.cageMask.inProgress = [];
                drawMaskCanvas();
            } else if (appState.cageMask.inProgress.length > 0) {
                statusLabel.textContent = `多角形には最低 3 点必要です (現在 ${appState.cageMask.inProgress.length} 点)`;
                statusLabel.style.color = '#FF8080';
                setTimeout(() => { statusLabel.style.color = ''; drawMaskCanvas(); }, 1500);
            }
        }

        // クリック: 点追加 / ダブルクリック: 多角形確定
        maskCanvas.addEventListener('click', (e) => {
            const pt = getCanvasPoint(e);
            appState.cageMask.inProgress.push(pt);
            drawMaskCanvas();
        });
        maskCanvas.addEventListener('dblclick', (e) => {
            e.preventDefault();
            finishCurrentPolygon();
        });

        // Enter: 確定 / Escape: キャンセル
        document.addEventListener('keydown', (e) => {
            if (!appState.cageMask.active) return;
            if (e.key === 'Enter') {
                e.preventDefault();
                finishCurrentPolygon();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                closeMaskEditor();
            } else if ((e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                btnUndo.click();
            }
        });

        btnFinishPoly.addEventListener('click', finishCurrentPolygon);

        btnUndo.addEventListener('click', () => {
            if (appState.cageMask.inProgress.length > 0) {
                appState.cageMask.inProgress.pop();
            } else if (appState.cageMask.polygons.length > 0) {
                // 編集中の点がない場合は最後の確定多角形を再編集対象に戻す
                appState.cageMask.inProgress = appState.cageMask.polygons.pop();
            }
            drawMaskCanvas();
        });

        btnClear.addEventListener('click', () => {
            if (!confirm('すべての多角形を消去しますか？')) return;
            appState.cageMask.polygons = [];
            appState.cageMask.inProgress = [];
            drawMaskCanvas();
        });

        btnSave.addEventListener('click', async () => {
            // 編集中の多角形があれば自動確定
            if (appState.cageMask.inProgress.length >= 3) {
                appState.cageMask.polygons.push(appState.cageMask.inProgress.slice());
                appState.cageMask.inProgress = [];
            }
            const videoName = appState.cageMask.videoName || 'unnamed.mp4';
            const payload = {
                video: videoName,
                video_width:  appState.videoW || 0,
                video_height: appState.videoH || 0,
                // 動画ピクセル座標で送信。サーバ側は小数 1 桁程度で十分。
                exclude_polygons: appState.cageMask.polygons.map(poly =>
                    poly.map(pt => [Math.round(pt[0] * 10) / 10, Math.round(pt[1] * 10) / 10])
                ),
            };
            try {
                btnSave.disabled = true;
                btnSave.textContent = '保存中...';
                const res = await fetch('/api/cage_mask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (!res.ok) {
                    const err = await res.text();
                    throw new Error(`HTTP ${res.status}: ${err}`);
                }
                const result = await res.json();
                console.log('[CageMask] 保存完了:', result);
                closeMaskEditor();
            } catch (err) {
                alert(`マスク保存に失敗しました: ${err.message || err}`);
            } finally {
                btnSave.disabled = false;
                btnSave.textContent = '保存して閉じる';
            }
        });

        btnCancel.addEventListener('click', closeMaskEditor);

        // ウィンドウリサイズ時に canvas を再フィット
        window.addEventListener('resize', () => {
            if (appState.cageMask.active) {
                resizeMaskCanvas();
                drawMaskCanvas();
            }
        });
    }

    lucide.createIcons();
});
