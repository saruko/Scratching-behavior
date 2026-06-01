/**
 * Mouse Scratch Detection System - Demo Data Generator
 * ====================================================
 * Programmatically generates 20 seconds (600 frames at 30 FPS) of high-fidelity
 * mouse skeleton tracks in a 800x550 open field cage.
 * Includes natural locomotion, walking cycles, and realistic scratching bouts:
 *   - Mouse 1 (Red): Scratching with LEFT hind paw near left ear (Frames 100-165)
 *   - Mouse 3 (Yellow): Scratching with RIGHT hind paw near right ear (Frames 320-385)
 */


function generateDemoData() {
    const numFrames = 600;
    const fps = 30;
    const width = 800;
    const height = 550;

    // Base coordinates & angles for each mouse
    const mice = {
        1: { name: "赤", color: "hsl(354, 90%, 55%)", x: 200, y: 150, angle: 0, speed: 2.5, size: 45 },
        2: { name: "青", color: "hsl(210, 95%, 55%)", x: 600, y: 150, angle: Math.PI / 2, speed: 1.8, size: 42 },
        3: { name: "黄", color: "hsl(45, 95%, 50%)", x: 200, y: 400, angle: Math.PI, speed: 2.2, size: 46 },
        4: { name: "灰", color: "hsl(220, 15%, 45%)", x: 600, y: 400, angle: -Math.PI / 2, speed: 1.2, size: 40 }
    };

    const frames = [];

    for (let f = 0; f < numFrames; f++) {
        const frameData = {
            frame_idx: f,
            mice: {}
        };

        for (const [id, mouse] of Object.entries(mice)) {
            const mId = parseInt(id);
            let isScratching = false;
            let scratchingSide = null; // "L" or "R"

            // --- Define Scratching Bout intervals ---
            if (mId === 1 && f >= 100 && f <= 165) {
                isScratching = true;
                scratchingSide = "L";
            } else if (mId === 3 && f >= 320 && f <= 385) {
                isScratching = true;
                scratchingSide = "R";
            }

            // --- Mouse Locomotion & Path Planning ---
            if (isScratching) {
                // Scratching mice freeze in space
                // Slight head vibration
                mouse.angle += Math.sin(f * 0.8) * 0.02;
            } else {
                // Natural random walk with boundary collision avoidance
                mouse.angle += (Math.sin(f * 0.05) * 0.1) + (Math.random() - 0.5) * 0.15;
                
                // Keep away from walls
                const padding = 80;
                if (mouse.x < padding) mouse.angle = 0 + (Math.random() - 0.5) * 0.5;
                if (mouse.x > width - padding) mouse.angle = Math.PI + (Math.random() - 0.5) * 0.5;
                if (mouse.y < padding) mouse.angle = Math.PI/2 + (Math.random() - 0.5) * 0.5;
                if (mouse.y > height - padding) mouse.angle = -Math.PI/2 + (Math.random() - 0.5) * 0.5;

                // Move forward
                mouse.x += Math.cos(mouse.angle) * mouse.speed;
                mouse.y += Math.sin(mouse.angle) * mouse.speed;
            }

            // --- Skeleton Calculations ---
            const cX = mouse.x;
            const cY = mouse.y;
            const size = mouse.size;
            const angle = mouse.angle;
            
            // Locomotion walk cycle (paws phase)
            const walkPhase = f * 0.4;
            const walkAmp = isScratching ? 0 : 8;

            // Base template positions (relative to neck = 0, 0, facing +X)
            const template = {};

            // 1. Spine line
            template["neck"] = { rx: -size * 0.2, ry: 0 };
            template["head_center"] = { rx: size * 0.1, ry: 0 };
            template["nose"] = { rx: size * 0.45, ry: 0 };
            template["spine_mid"] = { rx: -size * 0.55, ry: 0 };
            template["tail_base"] = { rx: -size * 0.9, ry: 0 };

            // 2. Ears
            template["left_ear"] = { rx: size * 0.1, ry: -size * 0.2 };
            template["right_ear"] = { rx: size * 0.1, ry: size * 0.2 };

            // 3. Fore paws (alternate walking movement)
            template["left_fore_paw"] = { 
                rx: -size * 0.1 + Math.sin(walkPhase) * walkAmp * 0.5, 
                ry: -size * 0.28 + Math.cos(walkPhase) * walkAmp * 0.3 
            };
            template["right_fore_paw"] = { 
                rx: -size * 0.1 - Math.sin(walkPhase) * walkAmp * 0.5, 
                ry: size * 0.28 - Math.cos(walkPhase) * walkAmp * 0.3 
            };

            // 4. Hind paws
            if (isScratching && scratchingSide === "L") {
                // Scratching: LEFT hind paw reaches left ear and vibrates rapidly (e.g. 8Hz)
                const vibX = Math.sin(f * 1.5) * 12;
                const vibY = Math.cos(f * 1.5) * 8;
                
                template["left_hind_paw"] = { 
                    rx: size * 0.1 + vibX, 
                    ry: -size * 0.18 + vibY 
                };
                template["right_hind_paw"] = { 
                    rx: -size * 0.8 + Math.sin(walkPhase) * walkAmp * 0.3, 
                    ry: size * 0.3 
                };
            } else if (isScratching && scratchingSide === "R") {
                // Scratching: RIGHT hind paw reaches right ear and vibrates rapidly
                const vibX = Math.sin(f * 1.5) * 12;
                const vibY = Math.cos(f * 1.5) * 8;
                
                template["left_hind_paw"] = { 
                    rx: -size * 0.8, 
                    ry: -size * 0.3 
                };
                template["right_hind_paw"] = { 
                    rx: size * 0.1 + vibX, 
                    ry: size * 0.18 + vibY 
                };
            } else {
                // Normal walking
                template["left_hind_paw"] = { 
                    rx: -size * 0.75 + Math.cos(walkPhase + Math.PI) * walkAmp * 0.7, 
                    ry: -size * 0.32 + Math.sin(walkPhase + Math.PI) * walkAmp * 0.3 
                };
                template["right_hind_paw"] = { 
                    rx: -size * 0.75 - Math.cos(walkPhase + Math.PI) * walkAmp * 0.7, 
                    ry: size * 0.32 - Math.sin(walkPhase + Math.PI) * walkAmp * 0.3 
                };
            }

            // Convert template coordinates (rotate and shift to absolute)
            const kps = {};
            const cosA = Math.cos(angle);
            const sinA = Math.sin(angle);

            for (const [kpName, raw] of Object.entries(template)) {
                // Rotate
                const ax = raw.rx * cosA - raw.ry * sinA;
                const ay = raw.rx * sinA + raw.ry * cosA;
                // B-2 デモ用の信頼度: 実際の姿勢推定を模し、後肢（オクルージョンしやすい）は
                // やや低め・体幹/頭部は高めの代表値を割り当て、信頼度カラー表示（高/中/低）が
                // デモモードでも視覚的に確認できるようにする。フレームごとに乱数を使うと
                // リングが点滅するため、キーポイント種別ごとの固定代表値とする。
                let demoConf = 0.93;
                if (kpName.includes("hind_paw")) {
                    demoConf = kpName.includes("left") ? 0.62 : 0.46; // 中 / 低 を再現
                } else if (kpName.includes("fore_paw")) {
                    demoConf = 0.78;
                } else if (kpName === "nose") {
                    demoConf = 0.71;
                }
                // Translate
                kps[kpName] = {
                    x: Math.round(cX + ax),
                    y: Math.round(cY + ay),
                    conf: demoConf
                };
            }

            // --- Features Extraction Matching feature_extraction.py ---
            // Calculate actual distances for plotting
            const headX = kps["head_center"].x;
            const headY = kps["head_center"].y;
            const earLX = kps["left_ear"].x;
            const earLY = kps["left_ear"].y;
            const earRX = kps["right_ear"].x;
            const earRY = kps["right_ear"].y;

            // Hind paw distances
            const distL = Math.sqrt(Math.pow(kps["left_hind_paw"].x - headX, 2) + Math.pow(kps["left_hind_paw"].y - headY, 2));
            const distR = Math.sqrt(Math.pow(kps["right_hind_paw"].x - headX, 2) + Math.pow(kps["right_hind_paw"].y - headY, 2));
            const distEarL = Math.sqrt(Math.pow(kps["left_hind_paw"].x - earLX, 2) + Math.pow(kps["left_hind_paw"].y - earLY, 2));
            const distEarR = Math.sqrt(Math.pow(kps["right_hind_paw"].x - earRX, 2) + Math.pow(kps["right_hind_paw"].y - earRY, 2));

            // Generate kinematic curves (velocities, oscillations)
            let speedL = isScratching && scratchingSide === "L" ? 45 + Math.sin(f * 2.0) * 10 : 8 + Math.random() * 5;
            let speedR = isScratching && scratchingSide === "R" ? 45 + Math.sin(f * 2.0) * 10 : 8 + Math.random() * 5;
            
            // Oscillation signals
            let oscL = 0;
            let oscR = 0;
            if (isScratching && scratchingSide === "L") {
                // Scratching triggers rapid sign reversals
                oscL = 8 + Math.floor(Math.sin(f * 0.1) * 2);
            } else if (isScratching && scratchingSide === "R") {
                oscR = 8 + Math.floor(Math.sin(f * 0.1) * 2);
            } else {
                oscL = Math.random() > 0.85 ? 1 : 0;
                oscR = Math.random() > 0.85 ? 1 : 0;
            }

            frameData.mice[mId] = {
                keypoints: kps,
                scratch_flag: isScratching ? 1 : 0,
                scratch_prob: isScratching ? 0.92 + Math.random() * 0.05 : 0.01 + Math.random() * 0.04,
                features: {
                    body_length_px: size,
                    head_to_nose_dist: Math.round(Math.sqrt(Math.pow(kps["nose"].x - headX, 2) + Math.pow(kps["nose"].y - headY, 2))),
                    
                    hind_to_head_dist_L: Math.round(distL),
                    hind_to_head_dist_R: Math.round(distR),
                    hind_to_ear_dist_L: Math.round(distEarL),
                    hind_to_ear_dist_R: Math.round(distEarR),
                    
                    hind_paw_speed_L: Math.round(speedL),
                    hind_paw_speed_R: Math.round(speedR),
                    
                    hind_osc_x_L: oscL,
                    hind_osc_y_L: oscL,
                    hind_osc_x_R: oscR,
                    hind_osc_y_R: oscR
                }
            };
        }

        frames.push(frameData);
    }

    // --- Bouts Summary matching pipeline.py output ---
    const bouts = [
        {
            mouse_id: 1,
            bout_id: 1,
            bout_start_frame: 100,
            bout_end_frame: 165,
            bout_start_time_s: 3.333,
            bout_end_time_s: 5.500,
            bout_duration_s: 2.167,
            bout_n_frames: 66,
            details: "左後足による左耳周辺のひっかき（ oscillation: 強 ）"
        },
        {
            mouse_id: 3,
            bout_id: 1,
            bout_start_frame: 320,
            bout_end_frame: 385,
            bout_start_time_s: 10.667,
            bout_end_time_s: 12.833,
            bout_duration_s: 2.167,
            bout_n_frames: 66,
            details: "右後足による右耳周辺のひっかき（ oscillation: 強 ）"
        }
    ];

    return {
        width,
        height,
        fps,
        numFrames,
        frames,
        bouts
    };
}

// Export for ES modules and standard script loading
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { generateDemoData, KEYPOINTS_LIST };
} else {
    window.generateDemoData = generateDemoData;
    window.KEYPOINTS_LIST = KEYPOINTS_LIST;
}
