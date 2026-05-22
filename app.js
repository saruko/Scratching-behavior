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
        rollingWindowSize: 30
    };

    // --- 2. Initialize Components ---
    const visualizer = new CageVisualizer('cage-canvas');
    const charts = new DashboardCharts();
    
    // Load programmatic high-fidelity demo data on start
    appState.data = window.generateDemoData();
    
    // Initialize empty/demo signals and analytics charts
    charts.initSignalChart('chart-live-signal');
    charts.updateAnalytics(appState.data.bouts);

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
    const videoStatusText = document.getElementById('video-status-text');

    // ID Swapper widgets
    const swapMouseA = document.getElementById('swap-mouse-a');
    const swapMouseB = document.getElementById('swap-mouse-b');
    const swapStartFrame = document.getElementById('swap-start-frame');
    const btnSwapIds = document.getElementById('btn-swap-ids');
    const btnExportCsv = document.getElementById('btn-export-csv');
    
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

        if (appState.isPlaying) {
            playIcon.setAttribute('data-lucide', 'pause');
            appState.lastFrameTime = 0;
            
            if (hasVideo) {
                bgVideo.playbackRate = appState.playbackSpeed;
                bgVideo.play();
            }
            requestAnimationFrame(playbackTick);
        } else {
            playIcon.setAttribute('data-lucide', 'play');
            if (hasVideo) {
                bgVideo.pause();
            }
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

        // Bidirectional Video Sync
        const hasVideo = bgVideo && bgVideo.src && bgVideo.readyState >= 2;
        if (hasVideo && syncVideo) {
            bgVideo.currentTime = currSeconds;
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

        valSpeedL.textContent = `${feats.hind_paw_speed_L} px/f`;
        valSpeedR.textContent = `${feats.hind_paw_speed_R} px/f`;
        valDistL.textContent = `${feats.hind_to_ear_dist_L} px`;
        valDistR.textContent = `${feats.hind_to_ear_dist_R} px`;
        valOscL.textContent = `${feats.hind_osc_x_L} rev`;
        valOscR.textContent = `${feats.hind_osc_x_R} rev`;

        // Check threshold boundaries
        // Left
        if (feats.hind_paw_speed_L > 30) {
            statSpeedL.className = "threshold-indicator pass";
            statSpeedL.querySelector('span:last-child').textContent = "✓ 30px 超過";
        } else {
            statSpeedL.className = "threshold-indicator fail";
            statSpeedL.querySelector('span:last-child').textContent = "✗ 低速";
        }
        if (feats.hind_to_ear_dist_L < 150) {
            statDistL.className = "threshold-indicator pass";
            statDistL.querySelector('span:last-child').textContent = "✓ 接近中";
        } else {
            statDistL.className = "threshold-indicator fail";
            statDistL.querySelector('span:last-child').textContent = "✗ 遠方";
        }

        // Right
        if (feats.hind_paw_speed_R > 30) {
            statSpeedR.className = "threshold-indicator pass";
            statSpeedR.querySelector('span:last-child').textContent = "✓ 30px 超過";
        } else {
            statSpeedR.className = "threshold-indicator fail";
            statSpeedR.querySelector('span:last-child').textContent = "✗ 低速";
        }
        if (feats.hind_to_ear_dist_R < 150) {
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

        // Apply rule-based scratch classifier matching features_extraction.py:rule_based_scratch_flag
        // Speed > 30 & EarDistance < 150 & Reversals >= 3
        const L_scratch = (mData.features.hind_paw_speed_L > 30 && mData.features.hind_to_ear_dist_L < 150 && mData.features.hind_osc_x_L >= 3);
        const R_scratch = (mData.features.hind_paw_speed_R > 30 && mData.features.hind_to_ear_dist_R < 150 && mData.features.hind_osc_x_R >= 3);

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

    function loadBackgroundVideo(file) {
        if (!file.type.startsWith('video/')) {
            alert("⚠️ 動画ファイル（.mp4など）を選択してください。");
            return;
        }

        const url = URL.createObjectURL(file);
        
        // Load into html5 video tag
        bgVideo.src = url;
        bgVideo.load();
        
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
            
            // Adjust slider max to match video frames if FPS is known (assume 30)
            const videoFramesCount = Math.floor(durationSec * appState.data.fps);
            if (videoFramesCount > 0 && Math.abs(videoFramesCount - appState.data.numFrames) > 5) {
                // If loaded video duration differs significantly, scale our frame array
                adjustFrameDuration(videoFramesCount);
            }

            updateUIForFrame();
            alert(`✓ 動画の同期に成功しました！\nファイル: ${file.name}\n長さ: ${durationSec.toFixed(2)} 秒`);
        };
    }

    function adjustFrameDuration(newFramesCount) {
        const oldFrames = appState.data.frames;
        const scaledFrames = [];

        for (let i = 0; i < newFramesCount; i++) {
            // Map index
            const oldIdx = Math.min(
                oldFrames.length - 1,
                Math.floor((i / newFramesCount) * oldFrames.length)
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
        if (appState.isPlaying && bgVideo && bgVideo.src) {
            bgVideo.playbackRate = appState.playbackSpeed;
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

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (file.name.includes('bouts')) {
                boutsFile = file;
            } else if (file.name.includes('frame') || file.name.endsWith('.csv')) {
                frameFile = file;
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
                hind_osc_x_L: getFloatVal(headers, cols, 'hind_osc_x_L', 0),
                hind_osc_y_L: getFloatVal(headers, cols, 'hind_osc_y_L', 0),
                hind_osc_x_R: getFloatVal(headers, cols, 'hind_osc_x_R', 0),
                hind_osc_y_R: getFloatVal(headers, cols, 'hind_osc_y_R', 0)
            };

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
                keypoints,
                features
            });
        }

        return records;
    }

    function parseBoutsCsv(csvText) {
        const lines = csvText.split(/\r?\n/);
        if (lines.length < 2) return [];

        const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
        const bouts = [];

        const mIdx = headers.indexOf('mouse_id');
        const bIdx = headers.indexOf('bout_id');
        const sfIdx = headers.indexOf('bout_start_frame');
        const efIdx = headers.indexOf('bout_end_frame');
        const durIdx = headers.indexOf('bout_duration_s');

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
                bout_start_time_s: parseFloat(cols[headers.indexOf('bout_start_time_s')] || startF / 30),
                bout_end_time_s: parseFloat(cols[headers.indexOf('bout_end_time_s')] || endF / 30),
                bout_duration_s: parseFloat(cols[durIdx] || nFrames / 30),
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

    // --- 13. Initial Kickoff ---
    updateUIForFrame();
    renderTimelineBouts();
    renderBoutsList();
    lucide.createIcons();
});
