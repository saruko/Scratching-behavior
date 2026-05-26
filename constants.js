/**
 * Mouse Scratch Detection System - Shared Constants
 */
const KEYPOINTS_LIST = [
    "nose",
    "left_ear",
    "right_ear",
    "head_center",
    "neck",
    "left_fore_paw",
    "right_fore_paw",
    "spine_mid",
    "left_hind_paw",
    "right_hind_paw",
    "tail_base"
];

// Thresholds for scratch behavior classification
const SCRATCH_THRESHOLDS = {
    minVelocity: 30,
    maxHeadDist: 150,
    minOscillations: 3
};

