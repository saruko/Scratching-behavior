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
        
        // Animation timing trackers (B-9)
        this.animTime = 0;
        this.lastTimestamp = null;
        
        // Define colors matching index.css variables
        this.mouseColors = {
            1: { hex: "#EF4444", name: "赤" },
            2: { hex: "#3B82F6", name: "青" },
            3: { hex: "#EAB308", name: "黄" },
            4: { hex: "#9CA3AF", name: "黒" }
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

        // Bind interactive mouse events
        this.initMouseEvents();
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
     */
    findClosestNode(x, y) {
        if (!this.currentFrameData) return null;

        let closest = null;
        let minDist = Infinity;

        // Search the selected mouse first, then other mice
        const mouseOrder = [];
        if (this.selectedMouseId) mouseOrder.push(this.selectedMouseId);
        for (let mId = 1; mId <= 4; mId++) {
            if (mId !== this.selectedMouseId) mouseOrder.push(mId);
        }

        mouseOrder.forEach(mId => {
            const mData = this.currentFrameData.mice[mId];
            if (!mData) return;

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

        if (!frameData) return;

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

        // 2. Draw each mouse
        for (const [id, mouseData] of Object.entries(frameData.mice)) {
            const mId = parseInt(id);
            const isSelected = mId === selectedMouseId;
            const isScratching = mouseData.scratch_flag === 1;
            const colorHex = this.mouseColors[mId].hex;

            this.drawMouse(mouseData, mId, isSelected, isScratching, colorHex);
        }
    }

    /**
     * Draw individual mouse contour, skeleton, tail color, and warning labels
     */
    drawMouse(mouseData, mouseId, isSelected, isScratching, colorHex) {
        const ctx = this.ctx;
        const kps = mouseData.keypoints;

        // Check if essential points are visible
        if (!kps["head_center"] || !kps["spine_mid"] || !kps["tail_base"]) return;

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
        ctx.fillText(`Mouse ${mouseId} (${this.mouseColors[mouseId].name})`, tailB.x, tailB.y + 24);
        ctx.restore();
    }
}

// Export class
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CageVisualizer;
} else {
    window.CageVisualizer = CageVisualizer;
}
