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
        isRealProgress: false         // Q-4: Fix pseudo progress bar vs log-extracted progress
    };

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
        if (cfg.SCRATCH_MIN_VELOCITY) SCRATCH_THRESHOLDS.minVelocity = cfg.SCRATCH_MIN_VELOCITY;
        if (cfg.SCRATCH_MAX_HEAD_DIST) SCRATCH_THRESHOLDS.maxHeadDist = cfg.SCRATCH_MAX_HEAD_DIST;
        if (cfg.SCRATCH_MIN_OSCILLATIONS) SCRATCH_THRESHOLDS.minOscillations = cfg.SCRATCH_MIN_OSCILLATIONS;
        console.log("[Config] Loaded thresholds from backend:", SCRATCH_THRESHOLDS);
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
    
    // FastAPI Pipeline integration widgets (NEW)
    const btnRunPipeline = document.getElementById('btn-run-pipeline');
    const pipelineEngine = document.getElementById('pipeline-engine');
    const pipelineCalibration = document.getElementById('pipeline-calibration');
    const pipelineModelPath = document.getElementById('pipeline-model-path');
    const pipelineStatusPanel = document.getElementById('pipeline-status-panel');
    const mockWarningBanner = document.getElementById('mock-warning-banner');
    
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
        visualizer.draw(frameData, appState.selectedMouse);

        // Update features cards & signal charts
        updateFeatureInspector(frameData);
        updateRollingSignal();
        checkGlobalScratchAlarms(frameData);
    }

    function updateFeatureInspector(frameData) {
        const mouseData = frameData.mice[appState.selectedMouse];
        if (!mouseData) return;

        const feats = mouseData.features;

        // 物理単位の取得 (CSVに物理カラムがない場合はピクセル値から概算)
        const speedL_mm = feats.hind_paw_speed_L_mm_s !== undefined && feats.hind_paw_speed_L_mm_s !== null && !isNaN(feats.hind_paw_speed_L_mm_s) ? feats.hind_paw_speed_L_mm_s.toFixed(1) : (feats.hind_paw_speed_L * 0.25 * appState.data.fps).toFixed(1);
        const speedR_mm = feats.hind_paw_speed_R_mm_s !== undefined && feats.hind_paw_speed_R_mm_s !== null && !isNaN(feats.hind_paw_speed_R_mm_s) ? feats.hind_paw_speed_R_mm_s.toFixed(1) : (feats.hind_paw_speed_R * 0.25 * appState.data.fps).toFixed(1);
        const distL_mm = feats.hind_to_ear_dist_L_mm !== undefined && feats.hind_to_ear_dist_L_mm !== null && !isNaN(feats.hind_to_ear_dist_L_mm) ? feats.hind_to_ear_dist_L_mm.toFixed(1) : (feats.hind_to_ear_dist_L * 0.25).toFixed(1);
        const distR_mm = feats.hind_to_ear_dist_R_mm !== undefined && feats.hind_to_ear_dist_R_mm !== null && !isNaN(feats.hind_to_ear_dist_R_mm) ? feats.hind_to_ear_dist_R_mm.toFixed(1) : (feats.hind_to_ear_dist_R * 0.25).toFixed(1);

        valSpeedL.textContent = `${feats.hind_paw_speed_L} px/f (${speedL_mm} mm/s)`;
        valSpeedR.textContent = `${feats.hind_paw_speed_R} px/f (${speedR_mm} mm/s)`;
        valDistL.textContent = `${feats.hind_to_ear_dist_L} px (${distL_mm} mm)`;
        valDistR.textContent = `${feats.hind_to_ear_dist_R} px (${distR_mm} mm)`;
        valOscL.textContent = `${feats.hind_osc_x_L} rev`;
        valOscR.textContent = `${feats.hind_osc_x_R} rev`;

        // Check threshold boundaries
        // Left
        if (feats.hind_paw_speed_L > SCRATCH_THRESHOLDS.minVelocity) {
            statSpeedL.className = "threshold-indicator pass";
            statSpeedL.querySelector('span:last-child').textContent = `✓ ${SCRATCH_THRESHOLDS.minVelocity}px 超過`;
        } else {
            statSpeedL.className = "threshold-indicator fail";
            statSpeedL.querySelector('span:last-child').textContent = "✗ 低速";
        }
        if (feats.hind_to_ear_dist_L < SCRATCH_THRESHOLDS.maxHeadDist) {
            statDistL.className = "threshold-indicator pass";
            statDistL.querySelector('span:last-child').textContent = "✓ 接近中";
        } else {
            statDistL.className = "threshold-indicator fail";
            statDistL.querySelector('span:last-child').textContent = "✗ 遠方";
        }

        // Right
        if (feats.hind_paw_speed_R > SCRATCH_THRESHOLDS.minVelocity) {
            statSpeedR.className = "threshold-indicator pass";
            statSpeedR.querySelector('span:last-child').textContent = `✓ ${SCRATCH_THRESHOLDS.minVelocity}px 超過`;
        } else {
            statSpeedR.className = "threshold-indicator fail";
            statSpeedR.querySelector('span:last-child').textContent = "✗ 低速";
        }
        if (feats.hind_to_ear_dist_R < SCRATCH_THRESHOLDS.maxHeadDist) {
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
     * Biomechanic Recalculation Engine matching Python feature_extraction.py
     */
    function recalculateFrameKinematics(fIdx, mouseId) {
        const frame = appState.data.frames[fIdx];
        const prevFrame = appState.data.frames[fIdx - 1];
        if (!frame || !frame.mice[mouseId]) return;

        const mData = frame.mice[mouseId];
        const kps = mData.keypoints;

        const headX = kps["head_center"].x;
        const headY = kps["head_center"].y;
        const earLX = kps["left_ear"].x;
        const earLY = kps["left_ear"].y;
        const earRX = kps["right_ear"].x;
        const earRY = kps["right_ear"].y;

        // Recalculate spatial distances
        mData.features.body_length_px = Math.round(Math.sqrt(Math.pow(kps["neck"].x - kps["tail_base"].x, 2) + Math.pow(kps["neck"].y - kps["tail_base"].y, 2)));
        mData.features.head_to_nose_dist = Math.round(Math.sqrt(Math.pow(kps["nose"].x - headX, 2) + Math.pow(kps["nose"].y - headY, 2)));
        
        mData.features.hind_to_head_dist_L = Math.round(Math.sqrt(Math.pow(kps["left_hind_paw"].x - headX, 2) + Math.pow(kps["left_hind_paw"].y - headY, 2)));
        mData.features.hind_to_head_dist_R = Math.round(Math.sqrt(Math.pow(kps["right_hind_paw"].x - headX, 2) + Math.pow(kps["right_hind_paw"].y - headY, 2)));
        mData.features.hind_to_ear_dist_L = Math.round(Math.sqrt(Math.pow(kps["left_hind_paw"].x - earLX, 2) + Math.pow(kps["left_hind_paw"].y - earLY, 2)));
        mData.features.hind_to_ear_dist_R = Math.round(Math.sqrt(Math.pow(kps["right_hind_paw"].x - earRX, 2) + Math.pow(kps["right_hind_paw"].y - earRY, 2)));

        // Recalculate velocities based on frame displacements
        if (prevFrame && prevFrame.mice[mouseId]) {
            const prevKps = prevFrame.mice[mouseId].keypoints;
            
            const dxL = kps["left_hind_paw"].x - prevKps["left_hind_paw"].x;
            const dyL = kps["left_hind_paw"].y - prevKps["left_hind_paw"].y;
            mData.features.hind_paw_speed_L = Math.round(Math.sqrt(dxL*dxL + dyL*dyL));

            const dxR = kps["right_hind_paw"].x - prevKps["right_hind_paw"].x;
            const dyR = kps["right_hind_paw"].y - prevKps["right_hind_paw"].y;
            mData.features.hind_paw_speed_R = Math.round(Math.sqrt(dxR*dxR + dyR*dyR));
        }

        // M-6: 振動数 (hind_osc_x_L / R) の再計算ロジックを追加 (15フレームのローリング窓でのx軸方向の速度符号反転回数)
        const windowFrames = 15;
        const startF = Math.max(1, fIdx - windowFrames + 1);
        
        // 左後足の振動数
        let signChangesL = 0;
        let lastDxL = 0;
        for (let i = startF; i <= fIdx; i++) {
            const currF = appState.data.frames[i];
            const prevF = appState.data.frames[i - 1];
            if (currF && prevF && currF.mice[mouseId] && prevF.mice[mouseId]) {
                const dx = currF.mice[mouseId].keypoints["left_hind_paw"].x - prevF.mice[mouseId].keypoints["left_hind_paw"].x;
                if (dx !== 0) {
                    if (lastDxL !== 0 && ((dx > 0 && lastDxL < 0) || (dx < 0 && lastDxL > 0))) {
                        signChangesL++;
                    }
                    lastDxL = dx;
                }
            }
        }
        mData.features.hind_osc_x_L = signChangesL;

        // 右後足の振動数
        let signChangesR = 0;
        let lastDxR = 0;
        for (let i = startF; i <= fIdx; i++) {
            const currF = appState.data.frames[i];
            const prevF = appState.data.frames[i - 1];
            if (currF && prevF && currF.mice[mouseId] && prevF.mice[mouseId]) {
                const dx = currF.mice[mouseId].keypoints["right_hind_paw"].x - prevF.mice[mouseId].keypoints["right_hind_paw"].x;
                if (dx !== 0) {
                    if (lastDxR !== 0 && ((dx > 0 && lastDxR < 0) || (dx < 0 && lastDxR > 0))) {
                        signChangesR++;
                    }
                    lastDxR = dx;
                }
            }
        }
        mData.features.hind_osc_x_R = signChangesR;

        const L_scratch = (mData.features.hind_paw_speed_L > SCRATCH_THRESHOLDS.minVelocity && 
                           mData.features.hind_to_ear_dist_L < SCRATCH_THRESHOLDS.maxHeadDist && 
                           mData.features.hind_osc_x_L >= SCRATCH_THRESHOLDS.minOscillations);
        const R_scratch = (mData.features.hind_paw_speed_R > SCRATCH_THRESHOLDS.minVelocity && 
                           mData.features.hind_to_ear_dist_R < SCRATCH_THRESHOLDS.maxHeadDist && 
                           mData.features.hind_osc_x_R >= SCRATCH_THRESHOLDS.minOscillations);

        if (L_scratch || R_scratch) {
            mData.scratch_flag = 1;
            mData.scratch_prob = 0.95;
        } else {
            // Maintain user loaded/original flags if not explicitly cleared by rules
            // Let's allow clearing if speeds drop significantly
            if (mData.features.hind_to_ear_dist_L > 165 && mData.features.hind_to_ear_dist_R > 165) {
                mData.scratch_flag = 0;
                mData.scratch_prob = 0.05;
            }
        }
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

        // Perform Swap starting at startF to EOF
        for (let f = startF; f < appState.data.numFrames; f++) {
            const frame = appState.data.frames[f];
            if (frame && frame.mice[mouseA] && frame.mice[mouseB]) {
                const temp = frame.mice[mouseA];
                frame.mice[mouseA] = frame.mice[mouseB];
                frame.mice[mouseB] = temp;
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
            
            alert("✓ 補正後データのCSV出力を完了しました！");
        } catch (e) {
            console.error(e);
            alert("⚠️ CSVエクスポートに失敗しました。");
        }
    });

    function exportCorrectedCSV() {
        const headers = ["frame_idx", "track", "mouse_id", "scratch_flag", "scratch_prob"];
        
        // Add keypoint columns
        KEYPOINTS_LIST.forEach(kp => {
            headers.push(`${kp}_x`, `${kp}_y`);
        });

        // Add feature columns
        const featKeys = [
            "body_length_px", "head_to_nose_dist",
            "hind_to_head_dist_L", "hind_to_head_dist_R",
            "hind_to_ear_dist_L", "hind_to_ear_dist_R",
            "hind_paw_speed_L", "hind_paw_speed_R",
            "hind_osc_x_L", "hind_osc_y_L",
            "hind_osc_x_R", "hind_osc_y_R"
        ];
        featKeys.forEach(fk => headers.push(fk));

        const rows = [headers.join(",")];

        for (let f = 0; f < appState.data.numFrames; f++) {
            const frame = appState.data.frames[f];
            
            for (let mId = 1; mId <= 4; mId++) {
                const mData = frame.mice[mId];
                if (!mData) continue;

                const cols = [
                    f,
                    `mouse_${mId}`,
                    mId,
                    mData.scratch_flag,
                    mData.scratch_prob.toFixed(3)
                ];

                // Coordinates
                KEYPOINTS_LIST.forEach(kp => {
                    const node = mData.keypoints[kp];
                    cols.push(node ? node.x : "", node ? node.y : "");
                });

                // Features
                featKeys.forEach(fk => {
                    cols.push(mData.features[fk] !== undefined ? mData.features[fk] : "");
                });

                rows.push(cols.join(","));
            }
        }

        return rows.join("\n");
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

        // Show upload progress bar
        showUploadProgress(file.name);

        // Use FileReader to get progress events (binary data read)
        const reader = new FileReader();
        reader.onprogress = (e) => {
            if (e.lengthComputable) {
                updateUploadProgress(e.loaded, e.total, file.name);
            }
        };
        reader.onload = () => {
            completeUploadProgress(file.name);
        };
        reader.readAsArrayBuffer(file);

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

            updateUIForFrame();
            alert(`✓ 動画の同期に成功しました！\nファイル: ${file.name}\n形式: ${file.type || fileExt}\n長さ: ${durationSec.toFixed(2)} 秒`);
        };
        
        // MOVファイルがブラウザで再生できない場合のエラーハンドリング
        bgVideo.onerror = (e) => {
            const errCode = bgVideo.error ? bgVideo.error.code : '?';
            const errMessages = {
                1: 'ユーザーによる再生中断',
                2: 'ネットワークエラー',
                3: 'デコードエラー: ファイルが堤損しているか、コーデック非対応',
                4: 'ファイルが見つからないか、ファイル形式非対応'
            };
            const errMsg = errMessages[errCode] || '不明なエラー';
            
            console.error('[Video] 再生エラー:', errCode, errMsg, file.name);
            
            if (fileExt === '.mov' || file.type === 'video/quicktime') {
                alert(`⚠️ MOVファイルの再生に失敗しました。\nエラー: ${errMsg}\n\nMOVファイルはブラウザによってコーデック互换性が異なります。\n\n解決策:\n1. ブラウザを Safari または Edge に変更する\n2. FFmpeg で .mp4 に変換: ffmpeg -i input.mov output.mp4`);
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

    speedSelect.addEventListener('change', (e) => {
        appState.playbackSpeed = parseFloat(e.target.value);
        if (bgVideo && bgVideo.src) {
            bgVideo.playbackRate = appState.playbackSpeed;
        }
        if (rawVideo && rawVideo.src) {
            rawVideo.playbackRate = appState.playbackSpeed;
        }
    });

    document.querySelectorAll('.mouse-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetBtn = e.currentTarget;
            const mouseId = parseInt(targetBtn.getAttribute('data-mouse-id'));
            
            document.querySelectorAll('.mouse-btn').forEach(b => b.classList.remove('active'));
            targetBtn.classList.add('active');
            
            appState.selectedMouse = mouseId;
            updateUIForFrame();
        });
    });

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
            block.style.color = visualizer.mouseColors[mId].hex;
            block.style.backgroundColor = visualizer.mouseColors[mId].hex;
            
            block.addEventListener('click', (e) => {
                e.stopPropagation();
                appState.currentFrame = b.bout_start_frame;
                updateUIForFrame();
            });

            track.appendChild(block);
        });
    }

    function renderBoutsList() {
        panelBoutsList.innerHTML = '';
        if (appState.data.bouts.length === 0) {
            panelBoutsList.innerHTML = '<div style="color:var(--text-muted); font-size:0.85rem; text-align:center; padding: 20px 0;">検出されたひっかき行動はありません。</div>';
            return;
        }

        appState.data.bouts.forEach(b => {
            const item = document.createElement('div');
            item.className = 'bout-item';
            
            const mouseName = visualizer.mouseColors[b.mouse_id].name;
            const mouseHex = visualizer.mouseColors[b.mouse_id].hex;

            item.innerHTML = `
                <div class="bout-meta">
                    <span class="mouse-color-dot" style="color: ${mouseHex}; background-color: currentColor;"></span>
                    <div>
                        <div class="bout-title">Mouse ${b.mouse_id} (${mouseName}) - Bout #${b.bout_id}</div>
                        <div class="bout-duration">${b.bout_n_frames}f (開始: ${b.bout_start_frame}f)</div>
                    </div>
                </div>
                <div class="bout-duration">${b.bout_duration_s.toFixed(2)}s</div>
            `;

            item.addEventListener('click', () => {
                appState.currentFrame = b.bout_start_frame;
                if (!appState.isPlaying) togglePlay();
                updateUIForFrame();
            });

            panelBoutsList.appendChild(item);
        });
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

                const maxFrameIdx = Math.max(...parsedFrames.map(r => r.frame_idx));
                const totalFrames = maxFrameIdx + 1;

                const framesArray = [];
                for (let f = 0; f < totalFrames; f++) {
                    framesArray.push({
                        frame_idx: f,
                        mice: {}
                    });
                }

                parsedFrames.forEach(row => {
                    const f = row.frame_idx;
                    const mId = row.mouse_id;
                    if (f < totalFrames && mId >= 1 && mId <= 4) {
                        framesArray[f].mice[mId] = {
                            keypoints: row.keypoints,
                            scratch_flag: row.scratch_flag,
                            scratch_prob: row.scratch_prob,
                            features: row.features
                        };
                    }
                });

                for (let f = 0; f < totalFrames; f++) {
                    const frame = framesArray[f];
                    for (let mId = 1; mId <= 4; mId++) {
                        if (!frame.mice[mId]) {
                            if (f > 0 && framesArray[f-1].mice[mId]) {
                                frame.mice[mId] = JSON.parse(JSON.stringify(framesArray[f-1].mice[mId]));
                                frame.mice[mId].scratch_flag = 0;
                            } else {
                                const defaultCenters = { 1: [200, 150], 2: [600, 150], 3: [200, 400], 4: [600, 400] };
                                const [cx, cy] = defaultCenters[mId];
                                frame.mice[mId] = createDefaultStaticMouse(cx, cy);
                            }
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

    function finalizeImport(fileName) {
        appState.currentFrame = 0;
        appState.isPlaying = false;
        playIcon.setAttribute('data-lucide', 'play');
        lucide.createIcons();

        scrubSlider.max = appState.data.numFrames - 1;

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
        const bouts = [];
        const frames = appState.data.frames;
        
        for (let mId = 1; mId <= 4; mId++) {
            let inBout = false;
            let startFrame = 0;
            let boutIdx = 1;

            for (let f = 0; f < frames.length; f++) {
                const isScratching = frames[f].mice[mId]?.scratch_flag === 1;

                if (isScratching && !inBout) {
                    inBout = true;
                    startFrame = f;
                } else if (!isScratching && inBout) {
                    let merge = false;
                    for (let peek = f; peek < Math.min(f + 10, frames.length); peek++) {
                        if (frames[peek].mice[mId]?.scratch_flag === 1) {
                            merge = true;
                            f = peek - 1;
                            break;
                        }
                    }

                    if (!merge) {
                        const durationFrames = f - startFrame;
                        if (durationFrames >= 5) {
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

            if (inBout && (frames.length - startFrame >= 5)) {
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

            const keypoints = {};
            KEYPOINTS_LIST.forEach(kp => {
                const xColName = `${kp}_x`;
                const yColName = `${kp}_y`;
                const xColIdx = headers.indexOf(xColName);
                const yColIdx = headers.indexOf(yColName);

                if (xColIdx !== -1 && yColIdx !== -1) {
                    keypoints[kp] = {
                        x: Math.round(parseFloat(cols[xColIdx])),
                        y: Math.round(parseFloat(cols[yColIdx])),
                        conf: 0.95
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

            let modifiedCount = 0;
            for (let f = startF; f <= endF; f++) {
                const frame = appState.data.frames[f];
                if (!frame || !frame.mice[mouseId]) continue;

                const kps = frame.mice[mouseId].keypoints;
                for (const kpName of Object.keys(kps)) {
                    kps[kpName].x = Math.round(kps[kpName].x + dx);
                    kps[kpName].y = Math.round(kps[kpName].y + dy);
                }
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
    btnRunPipeline.addEventListener('click', () => {
        // 新設された入力欄からパスを取得し、なければ dataset からのフォールバックを使用
        const inputVideoPath = document.getElementById('pipeline-video-path');
        const videoName = (inputVideoPath && inputVideoPath.value.trim()) || btnRunPipeline.dataset.videoName;
        
        if (!videoName) {
            alert("⚠️ 動画ファイルが指定されていません。先に動画をドロップするか、パスを直接入力してください。");
            return;
        }

        const engine = pipelineEngine.value;
        const calibration = parseFloat(pipelineCalibration.value) || 0.25;
        const modelPath = pipelineModelPath.value.trim() || null;

        // 状態を「進行中」へ変更
        btnRunPipeline.disabled = true;
        btnRunPipeline.textContent = "解析実行中...";
        pipelineStatusPanel.style.display = "block";
        pipelineStatusPanel.textContent = "[Server] 解析リクエストを送信中...\n";
        pipelineStatusPanel.style.color = "#38BDF8";

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
                px_to_mm: calibration
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

                // Extract [Progress: XX%] if present in logs
                const progressMatch = data.log_full.match(/\[Progress:\s*(\d+)%\]/g);
                if (progressMatch) {
                    const lastMatch = progressMatch[progressMatch.length - 1];
                    const pct = parseInt(lastMatch.match(/(\d+)/)[0]);
                    if (pct > 0) {
                        appState.isRealProgress = true; // Q-4: Mark that we are using real progress
                        if (appState.analysisProgressTimer) {
                            clearInterval(appState.analysisProgressTimer);
                            appState.analysisProgressTimer = null;
                        }
                        analysisProgressBar.style.width = `${pct}%`;
                        analysisProgressPct.textContent = `${pct}%`;
                    }
                }

                if (data.status === "completed") {
                    clearInterval(appState.pollingInterval);
                    appState.pollingInterval = null;
                    completeAnalysisProgressBar(true);
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

            const maxFrameIdx = Math.max(...parsedFrames.map(r => r.frame_idx));
            const totalFrames = maxFrameIdx + 1;

            const framesArray = [];
            for (let f = 0; f < totalFrames; f++) {
                framesArray.push({ frame_idx: f, mice: {} });
            }

            // モックフラグのグローバル検出
            let isMockData = false;

            parsedFrames.forEach(row => {
                const f = row.frame_idx;
                const mId = row.mouse_id;
                if (f < totalFrames && mId >= 1 && mId <= 4) {
                    framesArray[f].mice[mId] = {
                        keypoints: row.keypoints,
                        scratch_flag: row.scratch_flag,
                        scratch_prob: row.scratch_prob,
                        features: row.features
                    };
                }
                if (row.is_mock === 1) {
                    isMockData = true;
                }
            });

            // 欠損フレームの補完
            for (let f = 0; f < totalFrames; f++) {
                const frame = framesArray[f];
                for (let mId = 1; mId <= 4; mId++) {
                    if (!frame.mice[mId]) {
                        if (f > 0 && framesArray[f-1].mice[mId]) {
                            frame.mice[mId] = JSON.parse(JSON.stringify(framesArray[f-1].mice[mId]));
                            frame.mice[mId].scratch_flag = 0;
                        } else {
                            const defaultCenters = { 1: [200, 150], 2: [600, 150], 3: [200, 400], 4: [600, 400] };
                            const [cx, cy] = defaultCenters[mId];
                            frame.mice[mId] = createDefaultStaticMouse(cx, cy);
                        }
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

            Promise.all([
                fetch(frameUrl).then(res => {
                    if (!res.ok) throw new Error("CSVファイルを取得できませんでした。");
                    return res.text();
                }),
                boutsUrl ? fetch(boutsUrl).then(res => res.ok ? res.text() : '') : Promise.resolve('')
            ])
            .then(([frameText, boutsText]) => {
                const parsedFrames = parseFrameCsv(frameText);
                if (parsedFrames.length === 0) throw new Error("Parsed zero records");

                const maxFrameIdx = Math.max(...parsedFrames.map(r => r.frame_idx));
                const totalFrames = maxFrameIdx + 1;

                const framesArray = [];
                for (let f = 0; f < totalFrames; f++) {
                    framesArray.push({ frame_idx: f, mice: {} });
                }

                let isMockData = false;

                parsedFrames.forEach(row => {
                    const f = row.frame_idx;
                    const mId = row.mouse_id;
                    if (f < totalFrames && mId >= 1 && mId <= 4) {
                        framesArray[f].mice[mId] = {
                            keypoints: row.keypoints,
                            scratch_flag: row.scratch_flag,
                            scratch_prob: row.scratch_prob,
                            features: row.features
                        };
                    }
                    if (row.is_mock === 1 || resName.toLowerCase().includes('mock')) {
                        isMockData = true;
                    }
                });

                // Missing frames interpolation
                for (let f = 0; f < totalFrames; f++) {
                    const frame = framesArray[f];
                    for (let mId = 1; mId <= 4; mId++) {
                        if (!frame.mice[mId]) {
                            if (f > 0 && framesArray[f-1].mice[mId]) {
                                frame.mice[mId] = JSON.parse(JSON.stringify(framesArray[f-1].mice[mId]));
                                frame.mice[mId].scratch_flag = 0;
                            } else {
                                const defaultCenters = { 1: [200, 150], 2: [600, 150], 3: [200, 400], 4: [600, 400] };
                                const [cx, cy] = defaultCenters[mId];
                                frame.mice[mId] = createDefaultStaticMouse(cx, cy);
                            }
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

    // --- 13. Initial Kickoff ---
    updateUIForFrame();
    renderTimelineBouts();
    renderBoutsList();
    refreshServerResultsList(); // Auto-load server results list on kickoff
    lucide.createIcons();
});
