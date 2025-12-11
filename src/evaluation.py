# src/evaluation.py
"""
Evaluation module for the Loan Approval pipeline.

Usage:
    python src/evaluation.py

Outputs (under repo_root/artifacts/evaluation/):
 - metrics_summary.json
 - threshold_costs.csv
 - roc_curve.png, pr_curve.png, calibration_curve.png, threshold_vs_cost.png
 - feature_importance.png
 - shap_summary.png, shap_sample_waterfall.png (if explainer exists)
"""
from pathlib import Path
import json
import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc as auc_metric,
    brier_score_loss, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# Logging
logger = logging.getLogger("evaluation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler("logs/evaluation.log") if Path("logs").exists() else logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# Resolve repo-root
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
PREPROCESSOR_PATH = REPO_ROOT / "artifacts" / "preprocessor.joblib"
EXPLAINER_PATH = REPO_ROOT / "artifacts" / "shap_explainer.joblib"
METADATA_PATH = REPO_ROOT / "artifacts" / "metadata.json"
PROCESSED_DATA = REPO_ROOT / "data" / "processed" / "processed_loan_data.csv"
EVAL_DIR = REPO_ROOT / "artifacts" / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Default business costs (fallback)
DEFAULT_C_FP = 500
DEFAULT_C_FN = 5000

# ------------------ Helpers ------------------
def load_artifacts():
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not PREPROCESSOR_PATH.exists():
        missing.append(str(PREPROCESSOR_PATH))
    if missing:
        msg = "Missing artifact(s):\n  - " + "\n  - ".join(missing) + \
              "\nRun training first: python src/train_model.py"
        logger.error(msg)
        raise FileNotFoundError(msg)

    model = joblib.load(str(MODEL_PATH))
    preprocessor = joblib.load(str(PREPROCESSOR_PATH))

    explainer = None
    if EXPLAINER_PATH.exists():
        try:
            explainer = joblib.load(str(EXPLAINER_PATH))
        except Exception as e:
            logger.warning("Could not load SHAP explainer: %s", e)
            explainer = None

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    return model, preprocessor, explainer, metadata

def load_processed_df():
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA}. Run data_preprocessing.py first.")
    df = pd.read_csv(str(PROCESSED_DATA))
    if "target" not in df.columns:
        raise ValueError("Processed CSV must contain 'target' column.")
    X = df.drop(columns=["target"])
    y = df["target"].values
    return X, y, df

# --------------- Metric & plot functions ---------------
def compute_threshold_costs(y_true, y_prob, thresholds, C_fp, C_fn):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = int(fp * C_fp + fn * C_fn)
        rows.append({"threshold": float(t), "cost": int(cost), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return pd.DataFrame(rows)

def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_pr(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc_metric(recall, precision)
    plt.figure(figsize=(6,6))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_calibration(y_true, y_prob, out_path, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker="o", label="Calibration")
    plt.plot([0,1],[0,1], "k--", alpha=0.6)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration curve"); plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_threshold_vs_cost(cost_df, out_path):
    plt.figure(figsize=(8,5))
    plt.plot(cost_df['threshold'], cost_df['cost'], marker='o')
    plt.xlabel("Threshold"); plt.ylabel("Expected business cost")
    plt.title("Threshold vs Business Cost")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_feature_importance(model, feature_names, out_path, top_n=30):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        names = np.array(feature_names)[idx]
        vals = importances[idx]
        plt.figure(figsize=(8, max(4, 0.2*len(names))))
        plt.barh(np.arange(len(names))[::-1], vals[::-1])
        plt.yticks(np.arange(len(names)), names[::-1])
        plt.xlabel("Feature importance"); plt.title("Top feature importances")
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    else:
        logger.info("No feature_importances_ attribute on model; skipping importance plot.")

# --------------- SHAP helpers ---------------
def shap_global_summary(explainer, X_df, out_path, sample_n=1000):
    try:
        Xs = X_df.sample(n=min(sample_n, X_df.shape[0]), random_state=42)
        try:
            shap_vals = explainer.shap_values(Xs)
        except Exception:
            shap_vals = explainer(Xs)
        import shap as _shap
        plt.figure(figsize=(8,6))
        _shap.summary_plot(shap_vals, Xs, show=False)
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        logger.info("Saved SHAP global summary to %s", out_path)
    except Exception as e:
        logger.exception("Failed to build SHAP global summary: %s", e)

def shap_single_waterfall(explainer, X_row_df, out_path):
    try:
        import shap as _shap
        try:
            shap_vals = explainer.shap_values(X_row_df)
            base = explainer.expected_value if hasattr(explainer, "expected_value") else None
        except Exception:
            # newer API
            out = explainer(X_row_df)
            shap_vals = out.values if hasattr(out, "values") else out
            base = out.base_values if hasattr(out, "base_values") else None

        # shap.plots.waterfall expects shap.Explanation in newer versions or old args
        try:
            _shap.plots.waterfall(_shap.Explanation(values=shap_vals[0],
                                                   base_values=base,
                                                   data=X_row_df.iloc[0]), show=False)
            plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        except Exception:
            # fallback: summary barplot of top features
            vals = np.array(shap_vals[0])
            idx = np.argsort(np.abs(vals))[::-1][:10]
            feat = X_row_df.columns[idx]
            plt.figure(figsize=(6,4))
            plt.barh(feat[::-1], vals[idx][::-1])
            plt.title("Top SHAP contributions (approx)"); plt.tight_layout()
            plt.savefig(out_path, dpi=150); plt.close()

        logger.info("Saved SHAP waterfall/sample to %s", out_path)
    except Exception as e:
        logger.exception("Failed to build SHAP waterfall: %s", e)

# --------------- Main evaluation flow ---------------
def evaluate_all(save_metadata_back=True):
    logger.info("Starting evaluation...")
    model, preprocessor, explainer, metadata = load_artifacts()
    X_df, y, full_df = load_processed_df()

    # train/holdout split (consistent with train)
    X_train, X_holdout, y_train, y_holdout = train_test_split(X_df, y, test_size=0.20, stratify=y, random_state=42)
    probs = model.predict_proba(X_holdout)[:,1]

    # metrics
    auc = float(roc_auc_score(y_holdout, probs))
    precision, recall, _ = precision_recall_curve(y_holdout, probs)
    pr_auc = float(auc_metric(recall, precision))
    brier = float(brier_score_loss(y_holdout, probs))
    y_pred = (probs >= metadata.get("threshold", 0.5)).astype(int)
    class_report = classification_report(y_holdout, y_pred, output_dict=True)

    # threshold vs cost
    C_fp = metadata.get("business_cost", {}).get("C_fp", DEFAULT_C_FP)
    C_fn = metadata.get("business_cost", {}).get("C_fn", DEFAULT_C_FN)
    thresholds = np.linspace(0.01, 0.99, 99)
    cost_df = compute_threshold_costs(y_holdout, probs, thresholds, C_fp, C_fn)
    cost_df.to_csv(EVAL_DIR / "threshold_costs.csv", index=False)

    best_row = cost_df.loc[cost_df['cost'].idxmin()]
    best_threshold = float(best_row['threshold'])
    best_cost = int(best_row['cost'])

    # Save metrics summary
    metrics = {
        "holdout_roc_auc": auc,
        "holdout_pr_auc": pr_auc,
        "brier_score": brier,
        "best_threshold_by_cost": best_threshold,
        "best_cost": best_cost,
        "C_fp": int(C_fp),
        "C_fn": int(C_fn)
    }
    with open(EVAL_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics summary to %s", str(EVAL_DIR / "metrics_summary.json"))

    # Save plots
    try:
        plot_roc(y_holdout, probs, EVAL_DIR / "roc_curve.png")
        plot_pr(y_holdout, probs, EVAL_DIR / "pr_curve.png")
        plot_calibration(y_holdout, probs, EVAL_DIR / "calibration_curve.png")
        plot_threshold_vs_cost(cost_df, EVAL_DIR / "threshold_vs_cost.png")
        plot_feature_importance(model, list(X_df.columns), EVAL_DIR / "feature_importance.png")
    except Exception as e:
        logger.exception("Plotting failed: %s", e)

    # SHAP visuals
    if explainer is not None:
        try:
            shap_global_summary(explainer, X_df, EVAL_DIR / "shap_summary.png")
            sample_idx = int(np.argmax(probs))  # highest-prob sample
            X_sample_row = X_holdout.iloc[[sample_idx]]
            shap_single_waterfall(explainer, X_sample_row, EVAL_DIR / "shap_sample_waterfall.png")
        except Exception as e:
            logger.exception("SHAP generation failed: %s", e)
    else:
        logger.info("No SHAP explainer available; skipping SHAP plots.")

    # Optionally update metadata.json with best threshold (but preserve existing fields)
    if save_metadata_back:
        md = metadata.copy()
        md.update({"evaluation": {"best_threshold_by_cost": best_threshold, "best_cost": best_cost}})
        # write back safely
        with open(METADATA_PATH, "w") as f:
            json.dump(md, f, indent=2)
        logger.info("Updated metadata.json with evaluation results.")

    logger.info("Evaluation complete. Artifacts saved under %s", EVAL_DIR)
    return metrics, cost_df

# CLI
if __name__ == "__main__":
    metrics, cost_df = evaluate_all(save_metadata_back=True)
    print("Evaluation finished. Metrics:")
    print(json.dumps(metrics, indent=2))
    print("Threshold costs saved to:", str(EVAL_DIR / "threshold_costs.csv"))
