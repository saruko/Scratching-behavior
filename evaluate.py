"""
Mouse Scratch Detection System — Ground Truth Evaluation Module (Fix 7)
=======================================================================
本モジュールは自動検出結果と手動アノテーション（グラウンドトゥルース）を比較し、
フレームレベルおよび Bout レベルの精度指標を算出します。

フレームレベル指標: Precision, Recall, F1, Cohen's κ (per-mouse)
Bout レベル指標: Sensitivity, FDR, F1 (temporal IoU ≥ threshold)
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Optional


# ── Ground truth loader ──────────────────────────────────────────────────────

def load_ground_truth(gt_path: str, fmt: str = "frame") -> pd.DataFrame:
    """
    グラウンドトゥルース CSV を読み込み、内部標準フォーマットに変換する。

    fmt='frame': 列 [frame_idx, mouse_id, is_scratch(0/1)]
    fmt='bout':  列 [mouse_id, start_frame, end_frame]  (または start_time_s/end_time_s)
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"グラウンドトゥルースファイルが見つかりません: {gt_path}")

    df = pd.read_csv(gt_path)
    df.columns = df.columns.str.strip().str.lower()

    if fmt == "frame":
        required = {"frame_idx", "mouse_id", "is_scratch"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"frame フォーマットに必要な列がありません: {missing}")
        df["is_scratch"] = df["is_scratch"].astype(int)
        return df[["frame_idx", "mouse_id", "is_scratch"]]

    elif fmt == "bout":
        # start_frame/end_frame または start_time_s/end_time_s を受け付ける
        if "start_frame" in df.columns and "end_frame" in df.columns:
            df["start_frame"] = df["start_frame"].astype(int)
            df["end_frame"] = df["end_frame"].astype(int)
        elif "start_time_s" in df.columns and "end_time_s" in df.columns:
            # 時刻→フレーム変換は evaluate_bouts() で行う（fps が必要）
            pass
        else:
            raise ValueError("bout フォーマットには start_frame/end_frame または start_time_s/end_time_s が必要です。")
        if "mouse_id" not in df.columns:
            raise ValueError("bout フォーマットには mouse_id 列が必要です。")
        return df

    else:
        raise ValueError(f"未知のフォーマット指定: {fmt}。'frame' か 'bout' を指定してください。")


# ── Cohen's κ (manual implementation — no sklearn dependency) ────────────────

def _cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """2値ラベル列に対して Cohen's κ を計算する。"""
    n = len(y_true)
    if n == 0:
        return float("nan")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    po = (tp + tn) / n  # 実際の一致率

    # 期待一致率
    p_pos = ((tp + fn) / n) * ((tp + fp) / n)
    p_neg = ((tn + fp) / n) * ((tn + fn) / n)
    pe = p_pos + p_neg

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ── Frame-level evaluation ───────────────────────────────────────────────────

def evaluate_frames(
    auto_df: pd.DataFrame,
    gt_frames_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    フレームレベルで自動検出 vs グラウンドトゥルースを比較する。

    auto_df     : pipeline 出力の *_frame_labels.csv を読み込んだ DataFrame。
                  必要列: [frame_idx, mouse_id, is_scratch]
    gt_frames_df: load_ground_truth(fmt='frame') の出力

    Returns: per-mouse DataFrame with columns
             [mouse_id, n_frames, precision, recall, f1, kappa, tp, fp, fn, tn]
    """
    # 自動結果を標準化
    # パイプライン出力は scratch_flag 列を使うため、is_scratch が無ければ scratch_flag を採用する。
    auto_df = auto_df.copy()
    if "is_scratch" not in auto_df.columns:
        if "scratch_flag" in auto_df.columns:
            auto_df["is_scratch"] = auto_df["scratch_flag"]
        else:
            raise ValueError(
                "自動検出CSVに 'is_scratch' も 'scratch_flag' も見つかりません。"
                "パイプライン出力の *_frame_labels.csv を指定してください。"
            )
    auto = auto_df[["frame_idx", "mouse_id", "is_scratch"]].copy()
    auto["is_scratch"] = auto["is_scratch"].fillna(0).astype(int)

    gt = gt_frames_df[["frame_idx", "mouse_id", "is_scratch"]].copy()
    gt["is_scratch"] = gt["is_scratch"].astype(int)

    # 全マウス ID (GT 優先、自動にしか居ないものも加える)
    all_mice = sorted(set(gt["mouse_id"].unique()) | set(auto["mouse_id"].unique()))

    rows = []
    for mid in all_mice:
        auto_m = auto[auto["mouse_id"] == mid].set_index("frame_idx")["is_scratch"]
        gt_m   = gt[gt["mouse_id"] == mid].set_index("frame_idx")["is_scratch"]

        # 共通フレームのみ評価
        common_frames = auto_m.index.intersection(gt_m.index)
        if len(common_frames) == 0:
            rows.append({
                "mouse_id": mid, "n_frames": 0,
                "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "kappa": float("nan"),
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            })
            continue

        y_pred = auto_m.loc[common_frames].values
        y_true = gt_m.loc[common_frames].values

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")

        kappa = _cohens_kappa(y_true, y_pred)

        rows.append({
            "mouse_id": mid,
            "n_frames": len(common_frames),
            "precision": round(precision, 4) if not np.isnan(precision) else float("nan"),
            "recall":    round(recall,    4) if not np.isnan(recall)    else float("nan"),
            "f1":        round(f1,        4) if not np.isnan(f1)        else float("nan"),
            "kappa":     round(kappa,     4) if not np.isnan(kappa)     else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return pd.DataFrame(rows)


# ── Bout-level evaluation ────────────────────────────────────────────────────

def _temporal_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    """フレーム区間 [s1,e1] と [s2,e2] の Temporal IoU を返す (閉区間)。"""
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = max(e1, e2) - min(s1, s2) + 1
    return inter / union if union > 0 else 0.0


def evaluate_bouts(
    auto_bouts: pd.DataFrame,
    gt_bouts: pd.DataFrame,
    fps: float = 30.0,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Bout レベルで自動検出 vs グラウンドトゥルースを比較する。

    auto_bouts, gt_bouts : 列 [mouse_id, start_frame, end_frame] を持つ DataFrame。
                           start_time_s/end_time_s が含まれる場合は fps でフレームに変換する。
    fps             : 変換に使用する FPS。
    iou_threshold   : 一致とみなす Temporal IoU の下限 (デフォルト 0.5)。

    Returns: per-mouse DataFrame with columns
             [mouse_id, n_gt_bouts, n_auto_bouts, tp_bouts, fp_bouts, fn_bouts,
              sensitivity, fdr, precision_bout, f1_bout]
    """
    def _to_frame_bouts(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "start_frame" not in df.columns:
            df["start_frame"] = (df["start_time_s"] * fps).round().astype(int)
            df["end_frame"]   = (df["end_time_s"]   * fps).round().astype(int)
        return df[["mouse_id", "start_frame", "end_frame"]]

    auto = _to_frame_bouts(auto_bouts)
    gt   = _to_frame_bouts(gt_bouts)

    all_mice = sorted(set(gt["mouse_id"].unique()) | set(auto["mouse_id"].unique()))

    rows = []
    for mid in all_mice:
        auto_m = auto[auto["mouse_id"] == mid][["start_frame", "end_frame"]].values.tolist()
        gt_m   = gt[gt["mouse_id"] == mid][["start_frame", "end_frame"]].values.tolist()

        matched_auto = set()
        matched_gt   = set()

        for gi, (gs, ge) in enumerate(gt_m):
            for ai, (as_, ae) in enumerate(auto_m):
                if ai in matched_auto:
                    continue
                if _temporal_iou(gs, ge, as_, ae) >= iou_threshold:
                    matched_gt.add(gi)
                    matched_auto.add(ai)
                    break  # greedy match — each GT bout matched at most once

        tp = len(matched_gt)
        fn = len(gt_m) - tp
        fp = len(auto_m) - len(matched_auto)

        sensitivity      = tp / len(gt_m)   if len(gt_m)   > 0 else float("nan")
        fdr              = fp / len(auto_m)  if len(auto_m) > 0 else float("nan")
        precision_bout   = tp / len(auto_m)  if len(auto_m) > 0 else float("nan")

        if not (np.isnan(sensitivity) or np.isnan(precision_bout)) and (sensitivity + precision_bout) > 0:
            f1_bout = 2 * sensitivity * precision_bout / (sensitivity + precision_bout)
        else:
            f1_bout = float("nan")

        rows.append({
            "mouse_id":       mid,
            "n_gt_bouts":     len(gt_m),
            "n_auto_bouts":   len(auto_m),
            "tp_bouts":       tp,
            "fp_bouts":       fp,
            "fn_bouts":       fn,
            "sensitivity":    round(sensitivity,    4) if not np.isnan(sensitivity)    else float("nan"),
            "fdr":            round(fdr,            4) if not np.isnan(fdr)            else float("nan"),
            "precision_bout": round(precision_bout, 4) if not np.isnan(precision_bout) else float("nan"),
            "f1_bout":        round(f1_bout,        4) if not np.isnan(f1_bout)        else float("nan"),
        })

    return pd.DataFrame(rows)


# ── Top-level runner ─────────────────────────────────────────────────────────

def run_evaluation(
    auto_frame_csv: str,
    gt_csv: str,
    gt_format: str = "frame",
    fps: float = 30.0,
    iou_threshold: float = 0.5,
    output_dir: Optional[str] = None,
    auto_bouts_csv: Optional[str] = None,
) -> dict:
    """
    自動検出結果 CSV とグラウンドトゥルース CSV を読み込み、
    フレーム・Bout レベルの評価を実施してサマリーを返す。

    Returns dict with keys:
        "frame_metrics"   : pd.DataFrame  (evaluate_frames の出力)
        "bout_metrics"    : pd.DataFrame or None
        "output_files"    : list[str]
    """
    auto_df = pd.read_csv(auto_frame_csv)
    gt_df   = load_ground_truth(gt_csv, fmt=gt_format)

    output_files = []

    # ── フレームレベル評価 ──
    if gt_format == "frame":
        frame_metrics = evaluate_frames(auto_df, gt_df)
    else:
        # GT が bout フォーマットの場合はフレームレベル評価をスキップ
        frame_metrics = pd.DataFrame()

    # ── Bout レベル評価 ──
    bout_metrics = None
    if gt_format == "bout" or auto_bouts_csv is not None:
        # 自動 bouts CSV の特定
        if auto_bouts_csv is None:
            # frame_labels.csv → bouts.csv の命名規則から推測
            auto_bouts_csv = auto_frame_csv.replace("_frame_labels.csv", "_bouts.csv")
        if os.path.exists(auto_bouts_csv):
            auto_bouts_df = pd.read_csv(auto_bouts_csv)
            gt_bouts_df   = gt_df if gt_format == "bout" else load_ground_truth(gt_csv, fmt="bout")
            bout_metrics  = evaluate_bouts(auto_bouts_df, gt_bouts_df, fps=fps, iou_threshold=iou_threshold)
        else:
            print(f"[evaluate] [Warning] Bout CSVが見つかりません: {auto_bouts_csv} — Bout評価をスキップします。")

    # ── 出力 ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(auto_frame_csv))[0]

        if not frame_metrics.empty:
            frame_out = os.path.join(output_dir, f"{base}_eval_frame.csv")
            frame_metrics.to_csv(frame_out, index=False)
            output_files.append(frame_out)
            print(f"[evaluate] フレーム評価結果: {frame_out}")

        if bout_metrics is not None and not bout_metrics.empty:
            bout_out = os.path.join(output_dir, f"{base}_eval_bout.csv")
            bout_metrics.to_csv(bout_out, index=False)
            output_files.append(bout_out)
            print(f"[evaluate] Bout評価結果: {bout_out}")

    return {
        "frame_metrics": frame_metrics,
        "bout_metrics":  bout_metrics,
        "output_files":  output_files,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="自動検出結果とグラウンドトゥルースを比較して精度指標を算出します。"
    )
    parser.add_argument("--auto",      required=True,  help="自動検出 *_frame_labels.csv のパス")
    parser.add_argument("--gt",        required=True,  help="グラウンドトゥルース CSV のパス")
    parser.add_argument("--format",    default="frame", choices=["frame", "bout"],
                        help="GT CSVのフォーマット (frame または bout, デフォルト: frame)")
    parser.add_argument("--fps",       type=float, default=30.0,
                        help="動画 FPS (bout フォーマット時の時刻変換に使用, デフォルト: 30)")
    parser.add_argument("--iou",       type=float, default=0.5,
                        help="Bout 一致判定の Temporal IoU 閾値 (デフォルト: 0.5)")
    parser.add_argument("--out_dir",   default=None,
                        help="評価 CSV の出力先ディレクトリ (省略時は標準出力のみ)")
    parser.add_argument("--auto_bouts", default=None,
                        help="自動検出 *_bouts.csv のパス (省略時は --auto から自動推測)")
    args = parser.parse_args()

    result = run_evaluation(
        auto_frame_csv=args.auto,
        gt_csv=args.gt,
        gt_format=args.format,
        fps=args.fps,
        iou_threshold=args.iou,
        output_dir=args.out_dir,
        auto_bouts_csv=args.auto_bouts,
    )

    print("\n=== フレームレベル評価 ===")
    if not result["frame_metrics"].empty:
        print(result["frame_metrics"].to_string(index=False))
    else:
        print("(フレームレベル評価なし)")

    print("\n=== Bout レベル評価 ===")
    if result["bout_metrics"] is not None and not result["bout_metrics"].empty:
        print(result["bout_metrics"].to_string(index=False))
    else:
        print("(Bout レベル評価なし)")
