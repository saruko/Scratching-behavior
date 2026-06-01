/**
 * Mouse Scratch Detection System - Canvas Visualizer (拡張版)
 * ==========================================================
 * Manages the HTML5 Canvas drawing cycle. Renders the 800x550 cage grid,
 * mouse silhouettes, skeletons, and warning alarms.
 * Added: Responsive coordinates mapping, keypoint hover selectors,
 * drag-and-drop node controllers, and drag callback channels.
 */

class CageVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error(`Canvas element with ID '${canvasId}' not found.`);
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        
        // Dynamic drag states
        this.dragTarget = null; // { mouseId, kpName }
        this.hoverTarget = null; // { mouseId, kpName }
        this.onNodeDrag = null; // Callback: (mouseId, kpName, x, y)
        this.onNodeDragEnd = null; // Callback: ()

        // ── A-2 キャリブレーション計測モード ───────────────────────────────
        // measureMode が true の間はキーポイントのドラッグを無効化し、
        // クリックで既知長の2点を打つ。座標は canvas 内部座標系（= 動画ピクセル空間、
        // resizeCanvasToVideo により canvas.width/height = 動画ネイティブ解像度）で取得するため、
        // 2点間のピクセル距離は動画上の真のピクセル距離に一致する。
        this.measureMode = false;
        this.measurePoints = [];           // [{x, y}, ...] 最大2点（動画ピクセル座標）
        this.onMeasureUpdate = null;       // Callback: (points[]) 点が追加/リセットされるたびに呼ぶ
        
        // Animation timing trackers (B-9)
        this.animTime = 0;
        this.lastTimestamp = null;
        
        // 動的カラーパレット: constants.js の MOUSE_COLOR_PALETTE を参照
        // 最大 MAX_MICE_LIMIT (8匹) まで対応
        this.mouseColors = typeof MOUSE_COLOR_PALETTE !== 'undefined'
            ? MOUSE_COLOR_PALETTE
            : {
                1: { hex: "#EF4444", name: "赤" },
                2: { hex: "#3B82F6", name: "青" },
                3: { hex: "#EAB308", name: "黄" },
                4: { hex: "#9CA3AF", name: "灰" },
                5: { hex: "#F0F0F0", name: "白" },
                6: { hex: "#F97316", name: "橙" },
                7: { hex: "#22C55E", name: "緑" },
                8: { hex: "#A855F7", name: "紫" },
            };

        // SLEAP Skeleton edges mapping
        this.edges = [
            ["nose",          "head_center"],
            ["head_center",   "left_ear"],
            ["head_center",   "right_ear"],
            ["head_center",   "neck"],
            ["neck",          "left_fore_paw"],
            ["neck",          "right_fore_paw"],
            ["neck",          "spine_mid"],
            ["spine_mid",     "left_hind_paw"],
            ["spine_mid",     "right_hind_paw"],
            ["spine_mid",     "tail_base"],
        ];

        // Cached current frame data for mouse collision detection
        this.currentFrameData = null;
        this.selectedMouseId = null;

        // ── B-2 キーポイント信頼度カラー表示 ───────────────────────────────
        // 各キーポイントの推定信頼度（conf, 0..1）をリング色で可視化する。
        // 学術判定では config.py の MIN_CONFIDENCE 未満の点は補間前にマスクされるため、
        // 描画される点は閾値以上だが、閾値近傍（低信頼）の点を研究者が一目で識別できるようにする。
        // conf が null（エンジンが信頼度を出力しない／取得不能）の場合はリングを描かない。
        this.showConfidence = true;

        // ── フェーズ1 初期化確認モード ─────────────────────────────────────
        // 'bbox': バウンディングボックスのみ強調（骨格はうっすら）
        // 'pose': 骨格を強調（bbox はうっすら）
        // null  : 通常モード（bbox 非表示）
        this.confirmationMode = null;
        this.previewFrames = null;     // プレビューJSON の frames 辞書
        this.previewSampledFrames = null; // サンプリングされたフレーム番号配列

        // Bind interactive mouse events
        this.initMouseEvents();
    }

    /**
     * 確認モード設定。
     * mode: 'bbox' | 'pose' | null
     * previewFrames: pipeline.py が出力した _preview.json の frames 辞書
     * sampledFrames: サンプリング対象のフレーム番号配列（昇順）
     */
    setConfirmationMode(mode, previewFrames = null, sampledFrames = null) {
        this.confirmationMode = mode;
        this.previewFrames = previewFrames;
        this.previewSampledFrames = sampledFrames;
    }

    /**
     * A-2 キャリブレーション計測モードの ON/OFF。
     * ON にするとキーポイントドラッグは無効化され、クリックで2点を打てる。
     * OFF にすると計測点はクリアされる。
     */
    setMeasureMode(on) {
        this.measureMode = !!on;
        this.measurePoints = [];
        this.canvas.style.cursor = on ? 'crosshair' : 'default';
        if (this.onMeasureUpdate) this.onMeasureUpdate(this.measurePoints);
        this._redraw();
    }

    /** 現在の計測点をクリアして再計測できるようにする。 */
    clearMeasurePoints() {
        this.measurePoints = [];
        if (this.onMeasureUpdate) this.onMeasureUpdate(this.measurePoints);
        this._redraw();
    }

    /** 2点間のピクセル距離を返す（2点未満なら null）。 */
    getMeasuredPixelDistance() {
        if (this.measurePoints.length < 2) return null;
        const [a, b] = this.measurePoints;
        return Math.sqrt(Math.pow(b.x - a.x, 2) + Math.pow(b.y - a.y, 2));
    }

    /**
     * 計測オーバーレイのみを再描画する軽量ヘルパー。
     * 再生停止中（render ループが回っていない）でもクリックに即応するために使用する。
     * 直近に draw() に渡された currentFrameData をそのまま使って全体を描き直す。
     */
    _redraw() {
        this.draw(this.currentFrameData, this.selectedMouseId);
    }

    /**
     * 任意フレームに最も近い「サンプリング済みフレーム」のバウンディングボックス情報を返す。
     * プレビューは 1 秒間隔のサンプリングのため、ユーザーがシークしたフレームに対して
     * 最近接のサンプルを採用する（前方サンプルを優先：手前のフレームを基準にする方が
     * 検出と表示のずれが視認しやすい）。
     */
    _lookupPreviewDetections(frameIdx) {
        if (!this.previewFrames || !this.previewSampledFrames || this.previewSampledFrames.length === 0) {
            return null;
        }
        // 二分探索で最近接の前方サンプルを取得
        const arr = this.previewSampledFrames;
        let lo = 0, hi = arr.length - 1;
        let best = arr[0];
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (arr[mid] <= frameIdx) {
                best = arr[mid];
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        const entry = this.previewFrames[String(best)];
        return entry ? { frame: best, detections: entry.detections } : null;
    }

    /**
     * Map client mouse click offsets into internal 800x550 canvas coordinate boundaries
     */
    getMousePos(evt) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (evt.clientX - rect.left) * scaleX,
            y: (evt.clientY - rect.top) * scaleY
        };
    }

    /**
     * Set up mouse listeners to enable real-time keypoint dragging
     */
    initMouseEvents() {
        this.canvas.addEventListener('mousedown', (e) => {
            // ── A-2 計測モード: クリックで既知長の2点を取得（ドラッグ処理より優先） ──
            if (this.measureMode) {
                const mpos = this.getMousePos(e);
                if (this.measurePoints.length >= 2) {
                    // 既に2点ある場合は新しい計測を開始する
                    this.measurePoints = [];
                }
                this.measurePoints.push({ x: mpos.x, y: mpos.y });
                if (this.onMeasureUpdate) this.onMeasureUpdate(this.measurePoints);
                this._redraw();
                e.preventDefault();
                return;
            }

            if (!this.currentFrameData) return;
            const pos = this.getMousePos(e);

            // Look for closest node to the click location
            const target = this.findClosestNode(pos.x, pos.y);
            if (target && target.dist < 12) { // 12px active radius
                const mData = this.currentFrameData.mice[target.mouseId];
                const kp = mData.keypoints[target.kpName];
                
                this.dragTarget = {
                    mouseId: target.mouseId,
                    kpName: target.kpName
                };
                
                // M-4: ドラッグ前の元座標を保存（キャンセル時のロールバック用）
                this.originalPos = { x: kp.x, y: kp.y };
                this.canvas.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.currentFrameData) return;
            const pos = this.getMousePos(e);

            if (this.dragTarget) {
                // Trigger live coordinate drag update callback
                if (this.onNodeDrag) {
                    // Clamp to canvas borders
                    const cx = Math.max(10, Math.min(this.canvas.width - 10, pos.x));
                    const cy = Math.max(10, Math.min(this.canvas.height - 10, pos.y));
                    this.onNodeDrag(this.dragTarget.mouseId, this.dragTarget.kpName, Math.round(cx), Math.round(cy));
                }
                e.preventDefault();
            } else {
                // Change cursor to pointer if hovering over an editable node
                const target = this.findClosestNode(pos.x, pos.y);
                if (target && target.dist < 12) {
                    this.canvas.style.cursor = 'pointer';
                    this.hoverTarget = { mouseId: target.mouseId, kpName: target.kpName };
                } else {
                    this.canvas.style.cursor = 'default';
                    this.hoverTarget = null;
                }
            }
        });

        const stopDragging = () => {
            if (this.dragTarget) {
                this.dragTarget = null;
                this.originalPos = null;
                this.canvas.style.cursor = 'default';
                if (this.onNodeDragEnd) {
                    this.onNodeDragEnd();
                }
            }
        };

        const cancelDragging = () => {
            if (this.dragTarget && this.originalPos) {
                // M-4: mouseleave 時は元の座標にロールバックする
                if (this.onNodeDrag) {
                    this.onNodeDrag(this.dragTarget.mouseId, this.dragTarget.kpName, this.originalPos.x, this.originalPos.y);
                }
                this.dragTarget = null;
                this.originalPos = null;
                this.canvas.style.cursor = 'default';
                if (this.onNodeDragEnd) {
                    this.onNodeDragEnd();
                }
            }
        };

        this.canvas.addEventListener('mouseup', stopDragging);
        this.canvas.addEventListener('mouseleave', cancelDragging);
    }

    /**
     * Proximity helper: Find the closest keypoint node in the current frame to (x, y)
     * 動的マウス数対応: frameData.mice のキーから自動的に全マウスを探索する
     */
    findClosestNode(x, y) {
        if (!this.currentFrameData) return null;

        let closest = null;
        let minDist = Infinity;

        // 選択中のマウスを優先し、それ以外はすべてのアクティブなマウスを探索
        const allIds = Object.keys(this.currentFrameData.mice).map(Number);
        const mouseOrder = [];
        if (this.selectedMouseId) mouseOrder.push(this.selectedMouseId);
        allIds.forEach(mId => {
            if (mId !== this.selectedMouseId) mouseOrder.push(mId);
        });

        mouseOrder.forEach(mId => {
            const mData = this.currentFrameData.mice[mId];
            // 無効化（disabled）されたマウスのノードはドラッグ対象外
            if (!mData || mData.disabled) return;

            for (const [kpName, kp] of Object.entries(mData.keypoints)) {
                if (isNaN(kp.x)) continue;
                const d = Math.sqrt(Math.pow(kp.x - x, 2) + Math.pow(kp.y - y, 2));
                if (d < minDist) {
                    minDist = d;
                    closest = { mouseId: mId, kpName, dist: d };
                }
            }
        });

        return closest;
    }

    /**
     * B-2: キーポイント信頼度（0..1）を3段階のリング色に変換する。
     * 閾値は config.py の MIN_CONFIDENCE(0.40) を学術的基準に合わせて設定。
     *   高(≥0.70): 緑 — 信頼できる推定
     *   中(0.50–0.70): 琥珀 — 注意
     *   低(<0.50): 赤 — マスク閾値近傍、位置が不確実
     * conf が null（信頼度データなし）の場合は null を返しリングを描かない。
     */
    _confidenceRingColor(conf) {
        if (conf == null || isNaN(conf)) return null;
        if (conf >= 0.70) return "#22C55E";
        if (conf >= 0.50) return "#EAB308";
        return "#EF4444";
    }

    draw(frameData, selectedMouseId = null, timestamp = null) {
        this.currentFrameData = frameData;
        this.selectedMouseId = selectedMouseId;

        if (timestamp !== null) {
            if (this.lastTimestamp !== null) {
                const dt = timestamp - this.lastTimestamp;
                this.animTime += Math.min(dt, 100);
            }
            this.lastTimestamp = timestamp;
        } else {
            this.animTime = Date.now();
            this.lastTimestamp = null;
        }

        if (!frameData) {
            // 計測モード時は frameData が無くてもオーバーレイだけは描く
            if (this.measureMode) {
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this._drawMeasureOverlay();
            }
            return;
        }

        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;

        // 1. Draw grid background (only if no background video is loaded)
        // Check if there is an active video running behind
        const bgVideo = document.getElementById('bg-video');
        const hasVideo = bgVideo && bgVideo.src && bgVideo.readyState >= 2;

        ctx.clearRect(0, 0, w, h);

        if (!hasVideo) {
            ctx.fillStyle = "#060911";
            ctx.fillRect(0, 0, w, h);

            ctx.strokeStyle = "rgba(255, 255, 255, 0.02)";
            ctx.lineWidth = 1;
            const gridSize = 40;
            for (let x = 0; x < w; x += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, h);
                ctx.stroke();
            }
            for (let y = 0; y < h; y += gridSize) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(w, y);
                ctx.stroke();
            }

            // Draw cage boundary
            ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            ctx.lineWidth = 4;
            ctx.strokeRect(10, 10, w - 20, h - 20);
        }

        // ── フェーズ1 確認モード: バウンディングボックス描画 ──────────────
        // confirmationMode === 'bbox' / 'pose' どちらでも bbox を描画する
        // （'pose' モードではうっすら、'bbox' モードでは強調）
        if (this.confirmationMode && frameData.frame_idx !== undefined) {
            this._drawPreviewBoundingBoxes(frameData.frame_idx);
        }

        // 2. Draw each mouse (動的マウス数対応: frameData.mice のエントリを全て描画)
        // 'bbox' モードでは骨格・キーポイント描画をスキップ（バウンディングボックス確認に集中）
        const drawSkeletons = this.confirmationMode !== 'bbox';
        if (drawSkeletons) {
            for (const [id, mouseData] of Object.entries(frameData.mice)) {
                const mId = parseInt(id);
                const isSelected = mId === selectedMouseId;
                const isScratching = mouseData.scratch_flag === 1;
                // 動的カラーパレット: 定義されていないIDはデフォルト色にフォールバック
                const colorEntry = this.mouseColors[mId] || { hex: "#AAAAAA", name: `M${mId}` };
                const colorHex = colorEntry.hex;

                this.drawMouse(mouseData, mId, isSelected, isScratching, colorHex);
            }
        }

        // ── A-2 キャリブレーション計測オーバーレイ（最前面に描画） ──
        if (this.measureMode) {
            this._drawMeasureOverlay();
        }
    }

    /**
     * 計測モードのオーバーレイ描画。
     * 打たれた点（最大2点）を十字マーカーで、2点揃ったら結ぶ線とピクセル距離を表示する。
     * 座標は canvas 内部座標系（= 動画ピクセル空間）でそのまま描画する。
     */
    _drawMeasureOverlay() {
        const ctx = this.ctx;
        const pts = this.measurePoints;
        if (!pts || pts.length === 0) return;

        ctx.save();
        const accent = '#22D3EE'; // シアン: 既存のマウス色と被りにくい計測専用色

        // 2点を結ぶ線
        if (pts.length >= 2) {
            ctx.strokeStyle = accent;
            ctx.lineWidth = 2;
            ctx.setLineDash([8, 5]);
            ctx.beginPath();
            ctx.moveTo(pts[0].x, pts[0].y);
            ctx.lineTo(pts[1].x, pts[1].y);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // 各点の十字マーカー
        pts.forEach((p, i) => {
            ctx.strokeStyle = accent;
            ctx.fillStyle = accent;
            ctx.lineWidth = 2;
            const r = 9;
            ctx.beginPath();
            ctx.moveTo(p.x - r, p.y);
            ctx.lineTo(p.x + r, p.y);
            ctx.moveTo(p.x, p.y - r);
            ctx.lineTo(p.x, p.y + r);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
            ctx.fill();
            // 点番号ラベル
            ctx.font = 'bold 12px Outfit, sans-serif';
            ctx.fillStyle = '#000';
            ctx.fillRect(p.x + 8, p.y - 18, 16, 14);
            ctx.fillStyle = accent;
            ctx.textBaseline = 'middle';
            ctx.fillText(String(i + 1), p.x + 12, p.y - 10);
        });

        // 距離ラベル（線の中点に表示）
        const dist = this.getMeasuredPixelDistance();
        if (dist != null) {
            const mx = (pts[0].x + pts[1].x) / 2;
            const my = (pts[0].y + pts[1].y) / 2;
            const label = `${dist.toFixed(1)} px`;
            ctx.font = 'bold 13px Outfit, sans-serif';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = 'rgba(0,0,0,0.78)';
            ctx.fillRect(mx - tw / 2 - 6, my - 22, tw + 12, 20);
            ctx.fillStyle = accent;
            ctx.textBaseline = 'middle';
            ctx.fillText(label, mx - tw / 2, my - 12);
        }

        ctx.restore();
    }

    /**
     * バウンディングボックスと検出 ID ラベルを描画する。
     * 'bbox' モード: 太線・実線・大ラベルで強調
     * 'pose' モード: 細線・破線・控えめなラベル
     *
     * バウンディングボックスはキーポイントの凸包から計算されているため、
     * 動画上のマウスを正確に囲む。ラベルには track 名 と suggested mouse_id を表示し、
     * 研究者が「この検出をどのマウスに割り当てるか」を判断できる。
     */
    _drawPreviewBoundingBoxes(frameIdx) {
        const lookup = this._lookupPreviewDetections(frameIdx);
        if (!lookup) return;

        const ctx = this.ctx;
        const isBboxMode = this.confirmationMode === 'bbox';

        for (const det of lookup.detections) {
            if (!det.bbox) continue;
            const [x1, y1, x2, y2] = det.bbox;
            const w = x2 - x1;
            const h = y2 - y1;

            // track → assigned mouse_id を appState 経由で取得（外部状態を参照）
            // window.appState がない場合は det.mouse_id（カラーID自動推定値）を使用
            // mouse_id=0 は「誤検出（除外）」を表す特別値（解析対象外）。
            let assignedId = det.mouse_id;
            let excluded = false;
            if (typeof window !== 'undefined' && window.appState && window.appState.confirmedMapping) {
                const cm = window.appState.confirmedMapping;
                if (Object.prototype.hasOwnProperty.call(cm, det.track)) {
                    const mapped = cm[det.track];
                    if (mapped === 0) { excluded = true; assignedId = null; }
                    else if (mapped) { assignedId = mapped; }
                }
            }

            const colorEntry = excluded
                ? { hex: '#9CA3AF', name: '誤検出' }
                : (assignedId != null && this.mouseColors[assignedId])
                    ? this.mouseColors[assignedId]
                    : { hex: '#FFFFFF', name: '未割当' };

            ctx.save();
            ctx.strokeStyle = colorEntry.hex;
            ctx.lineWidth = isBboxMode ? 4 : 2;
            // 誤検出は破線で控えめに表示（除外対象であることを視覚的に示す）
            if (!isBboxMode || excluded) {
                ctx.setLineDash([6, 4]);
            }
            if (excluded) ctx.globalAlpha = 0.55;
            ctx.strokeRect(x1, y1, w, h);
            ctx.globalAlpha = 1.0;

            // ID ラベル（左上に配置）
            const fontSize = isBboxMode ? 18 : 12;
            ctx.font = `bold ${fontSize}px Outfit, sans-serif`;
            const label = excluded
                ? `🚫 誤検出（除外）`
                : (assignedId != null
                    ? `Mouse ${assignedId} (${colorEntry.name})`
                    : `${det.track} → ID未割当`);
            const tw = ctx.measureText(label).width;
            const padX = 8, padY = 5;
            const labelW = tw + padX * 2;
            const labelH = fontSize + padY * 2;
            const labelX = Math.max(0, x1);
            const labelY = Math.max(labelH, y1) - labelH;

            ctx.fillStyle = colorEntry.hex;
            ctx.globalAlpha = isBboxMode ? 0.92 : 0.6;
            ctx.fillRect(labelX, labelY, labelW, labelH);
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = '#000';
            ctx.textBaseline = 'middle';
            ctx.fillText(label, labelX + padX, labelY + labelH / 2);

            ctx.restore();
        }

        // 'bbox' モードでは、サンプリングフレームからずれている場合に注意を表示
        if (isBboxMode && lookup.frame !== frameIdx) {
            ctx.save();
            ctx.fillStyle = 'rgba(251, 191, 36, 0.85)';
            ctx.font = '11px Outfit, sans-serif';
            ctx.fillText(`サンプル frame=${lookup.frame} を表示中 (現在 ${frameIdx})`, 10, this.canvas.height - 10);
            ctx.restore();
        }
    }

    /**
     * Draw individual mouse contour, skeleton, tail color, and warning labels
     * disabled フラグが true の場合は半透明のグレーで描画し、判定除外であることを示す
     */
    drawMouse(mouseData, mouseId, isSelected, isScratching, colorHex) {
        const ctx = this.ctx;
        const kps = mouseData.keypoints;

        // Check if essential points are visible
        if (!kps["head_center"] || !kps["spine_mid"] || !kps["tail_base"]) return;

        // 無効化（disabled）されたマウスはグレーでバツ印付きに描画
        if (mouseData.disabled) {
            this._drawDisabledMouse(mouseData, mouseId);
            return;
        }

        // Base opacity settings
        const baseOpacity = isSelected ? 1.0 : 0.45;

        // 1. Draw stylized glow backdrop for selected or scratching mice
        if (isSelected || isScratching) {
            ctx.save();
            ctx.beginPath();
            ctx.arc(kps["head_center"].x, kps["head_center"].y, 45, 0, Math.PI * 2);
            ctx.arc(kps["spine_mid"].x, kps["spine_mid"].y, 50, 0, Math.PI * 2);
            ctx.fillStyle = isScratching 
                ? "rgba(239, 68, 68, 0.05)" 
                : `${colorHex}08`;
            ctx.filter = "blur(12px)";
            ctx.fill();
            ctx.restore();
        }

        // 2. Draw Translucent Mouse Silhouette Body Contour
        ctx.save();
        ctx.beginPath();
        const headNode = kps["head_center"];
        const noseNode = kps["nose"];
        const earL = kps["left_ear"];
        const earR = kps["right_ear"];
        const spine = kps["spine_mid"];
        const tailB = kps["tail_base"];
        
        ctx.moveTo(noseNode.x, noseNode.y);
        ctx.quadraticCurveTo(earL.x, earL.y, spine.x + (earL.x - spine.x)*0.3, spine.y + (earL.y - spine.y)*0.3);
        ctx.lineTo(tailB.x, tailB.y);
        ctx.quadraticCurveTo(spine.x + (earR.x - spine.x)*0.3, spine.y + (earR.y - spine.y)*0.3, earR.x, earR.y);
        ctx.closePath();

        ctx.fillStyle = isScratching 
            ? `rgba(239, 68, 68, ${0.12 * baseOpacity})` 
            : `${colorHex}${Math.round(15 * baseOpacity).toString(16).padStart(2, '0')}`;
        ctx.fill();
        ctx.restore();

        // 3. Draw Skeleton Edges
        ctx.save();
        ctx.lineWidth = isSelected ? 3 : 2.2;
        this.edges.forEach(([from, to]) => {
            const kpFrom = kps[from];
            const kpTo = kps[to];
            if (kpFrom && kpTo && !isNaN(kpFrom.x) && !isNaN(kpTo.x)) {
                ctx.beginPath();
                ctx.moveTo(kpFrom.x, kpFrom.y);
                ctx.lineTo(kpTo.x, kpTo.y);
                
                if (isScratching) {
                    ctx.strokeStyle = "rgba(239, 68, 68, 0.9)";
                } else if (isSelected) {
                    ctx.strokeStyle = colorHex;
                } else {
                    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
                }
                ctx.stroke();
            }
        });
        ctx.restore();

        // 4. Draw Keypoints Nodes
        ctx.save();
        for (const [kpName, kp] of Object.entries(kps)) {
            if (isNaN(kp.x)) continue;

            let size = 4.5;
            let fillStyle = "rgba(255, 255, 255, 0.5)";
            let strokeStyle = "rgba(0, 0, 0, 0.6)";
            let lineWidth = 1.2;

            // Highlight dragging / hovered node
            const isHovered = this.hoverTarget && this.hoverTarget.mouseId === mouseId && this.hoverTarget.kpName === kpName;
            const isDragged = this.dragTarget && this.dragTarget.mouseId === mouseId && this.dragTarget.kpName === kpName;

            if (kpName === "tail_base") {
                size = 11;
                fillStyle = colorHex;
                strokeStyle = isDragged || isHovered ? "#00F2FE" : "#FFFFFF";
                lineWidth = isDragged ? 3.5 : 2.5;
            } else if (kpName === "head_center") {
                size = 5.5;
                fillStyle = "#FFFFFF";
            } else if (kpName === "nose") {
                size = 4;
                fillStyle = "#FF8080";
            } else if (kpName.includes("hind_paw")) {
                size = isScratching ? 8 : 5.5;
                fillStyle = isScratching ? "#EF4444" : "#00F2FE";
                if (isScratching) {
                    strokeStyle = "#FFFFFF";
                    lineWidth = 1.8;
                }
            }

            // Drag overlays
            if (isDragged) {
                fillStyle = "#00F2FE";
                strokeStyle = "#FFFFFF";
                lineWidth = 2.5;
                size += 4;
            } else if (isHovered) {
                fillStyle = "#FFF";
                strokeStyle = "#00F2FE";
                lineWidth = 2.0;
                size += 2;
            }

            ctx.beginPath();
            ctx.arc(kp.x, kp.y, size, 0, Math.PI * 2);
            ctx.fillStyle = fillStyle;
            ctx.strokeStyle = strokeStyle;
            ctx.lineWidth = lineWidth;
            ctx.fill();
            ctx.stroke();

            // ── B-2 信頼度リング: ノード外周に conf 段階色のリングを重畳描画 ──
            // ドラッグ/ホバー中はシアンの強調表示が優先されるためリングを省略する。
            if (this.showConfidence && !isDragged && !isHovered) {
                const ringColor = this._confidenceRingColor(kp.conf);
                if (ringColor) {
                    ctx.beginPath();
                    ctx.arc(kp.x, kp.y, size + 3, 0, Math.PI * 2);
                    ctx.strokeStyle = ringColor;
                    ctx.lineWidth = 2;
                    ctx.globalAlpha = isSelected ? 0.95 : 0.6;
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;
                }
            }

            // Glow overlay around dragging node
            if (isDragged) {
                ctx.beginPath();
                ctx.arc(kp.x, kp.y, size + 6, 0, Math.PI * 2);
                ctx.strokeStyle = "rgba(0, 242, 254, 0.4)";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }
        ctx.restore();

        // 5. Draw Warning Alert Overlay when Scratching is detected
        if (isScratching) {
            ctx.save();
            const head = kps["head_center"];
            
            const ringRadius = 38 + Math.sin(this.animTime * 0.015) * 8;
            ctx.beginPath();
            ctx.arc(head.x, head.y, ringRadius, 0, Math.PI * 2);
            ctx.strokeStyle = "rgba(239, 68, 68, 0.6)";
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Label text above head
            ctx.fillStyle = "#EF4444";
            ctx.font = "bold 11px Outfit";
            ctx.textAlign = "center";
            ctx.fillText("⚠️ ひっかき検出", head.x, head.y - 48);

            ctx.restore();
        }

        // Draw Mouse Tag ID Label near tail
        ctx.save();
        ctx.fillStyle = isSelected ? "#FFFFFF" : "rgba(255, 255, 255, 0.55)";
        ctx.font = "600 10.5px Outfit";
        ctx.textAlign = "center";
        const colorEntry = this.mouseColors[mouseId] || { name: `M${mouseId}` };
        ctx.fillText(`Mouse ${mouseId} (${colorEntry.name})`, tailB.x, tailB.y + 24);
        ctx.restore();
    }

    /**
     * 無効化されたマウス（disabled=true）を半透明グレーでバツ印付きに描画する。
     * この個体の検出がその フレームで手動除外されていることを視覚的に示す。
     */
    _drawDisabledMouse(mouseData, mouseId) {
        const ctx = this.ctx;
        const kps = mouseData.keypoints;
        const headKp = kps["head_center"];
        const tailKp = kps["tail_base"];
        if (!headKp || !tailKp) return;

        const cx = (headKp.x + tailKp.x) / 2;
        const cy = (headKp.y + tailKp.y) / 2;
        const R = 22;

        ctx.save();
        ctx.globalAlpha = 0.4;
        // グレーの半透明円
        ctx.beginPath();
        ctx.arc(cx, cy, R, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(100, 100, 100, 0.3)";
        ctx.strokeStyle = "rgba(180, 180, 180, 0.6)";
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();
        // バツ印
        ctx.strokeStyle = "rgba(255, 80, 80, 0.8)";
        ctx.lineWidth = 2.5;
        const d = R * 0.6;
        ctx.beginPath();
        ctx.moveTo(cx - d, cy - d); ctx.lineTo(cx + d, cy + d);
        ctx.moveTo(cx + d, cy - d); ctx.lineTo(cx - d, cy + d);
        ctx.stroke();
        // ラベル
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = "#FF6060";
        ctx.font = "600 9px Outfit";
        ctx.textAlign = "center";
        ctx.fillText(`M${mouseId} [除外]`, cx, cy + R + 12);
        ctx.restore();
    }
}

// Export class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CageVisualizer;
} else {
    window.CageVisualizer = CageVisualizer;
}
