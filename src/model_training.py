# src/train_model.py
"""
Train pipeline with K-Fold CV + RandomizedSearchCV for RandomForest and XGBoost.
Saves:
 - best model (models/model.joblib)
 - CV results for both searches (artifacts/rf_cv_results.csv, artifacts/xgb_cv_results.csv)
 - metadata (artifacts/metadata.json)
 - shap explainer (artifacts/shap_explainer.joblib)
 - threshold costs (artifacts/threshold_costs.csv)
"""
import os
import json
import joblib
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import shap
import time
import scipy.stats as stats

# ---------- Config ----------
SEED = 42
N_JOBS = -1
CV_FOLDS = 5
N_ITER_RF = 40       # number of random search iterations for RF
N_ITER_XGB = 40      # number of random search iterations for XGB
RANDOM_STATE = SEED

C_FP = 500
C_FN = 5000

# near top of src/train_model.py (replace existing path constants / os.makedirs)
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA = REPO_ROOT / "data" / "processed" / "processed_loan_data.csv"
MODEL_OUT = REPO_ROOT / "models" / "model.joblib"
METADATA_OUT = REPO_ROOT / "artifacts" / "metadata.json"
RF_CV_OUT = REPO_ROOT / "artifacts" / "rf_cv_results.csv"
XGB_CV_OUT = REPO_ROOT / "artifacts" / "xgb_cv_results.csv"
THRESH_COSTS_OUT = REPO_ROOT / "artifacts" / "threshold_costs.csv"
SHAP_EXPLAINER_OUT = REPO_ROOT / "artifacts" / "shap_explainer.joblib"

os.makedirs(REPO_ROOT / "models", exist_ok=True)
os.makedirs(REPO_ROOT / "artifacts", exist_ok=True)
os.makedirs(REPO_ROOT / "logs", exist_ok=True)


# ---------- Logging ----------
logger = logging.getLogger("train_model")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/train_model.log")
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter); ch.setFormatter(formatter)
logger.addHandler(fh); logger.addHandler(ch)

# ---------- Helpers ----------
def load_processed_data(path=PROCESSED_DATA):
    df = pd.read_csv(path)
    logger.info(f"Loaded processed data: {df.shape}")
    if "target" not in df.columns:
        raise ValueError("Processed file must contain 'target' column.")
    X = df.drop(columns=["target"])
    y = df["target"].values
    return X, y

def compute_expected_cost(y_true, y_prob, thresholds, C_fp=C_FP, C_fn=C_FN):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = int(fp * C_fp + fn * C_fn)
        rows.append({"threshold": float(t), "cost": int(cost), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return pd.DataFrame(rows)

# replace RF estimator creation
rf = RandomForestClassifier(random_state=SEED, n_jobs=1)  # avoid nested parallelism; RandomizedSearchCV will parallelize

# replace save_cv_results function
def save_cv_results(cv_result, out_csv):
    import pandas as pd, json
    res = pd.DataFrame(cv_result)
    cols = [c for c in res.columns if c.startswith("param_")] + ["mean_test_score","std_test_score","rank_test_score"]
    cols = [c for c in cols if c in res.columns]
    tidy = res[cols].copy()
    params = res[[c for c in res.columns if c.startswith("param_")]].to_dict(orient="records")
    tidy = tidy.drop(columns=[c for c in tidy.columns if c.startswith("param_")], errors="ignore")
    tidy["params"] = [json.dumps(p) for p in params]
    tidy.to_csv(out_csv, index=False)
    logger.info(f"Saved tidy CV results to {out_csv}")



# ---------- Hyperparameter spaces ----------
def rf_param_dist():
    return {
        "n_estimators": stats.randint(100, 1000),
        "max_depth": [None] + list(range(4, 21)),
        "min_samples_split": stats.randint(2, 11),
        "min_samples_leaf": stats.randint(1, 11),
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"]
    }

def xgb_param_dist():
    return {
        "n_estimators": stats.randint(100, 1500),
        "max_depth": stats.randint(3, 12),
        "learning_rate": stats.loguniform(0.01, 0.3),
        "subsample": stats.uniform(0.5, 0.5),
        "colsample_bytree": stats.uniform(0.5, 0.5),
        "gamma": stats.expon(0, 5),
        "reg_alpha": stats.loguniform(1e-8, 10),
        "reg_lambda": stats.loguniform(1e-8, 10),
    }

# ---------- Main train function ----------
def train_and_tune():
    X, y = load_processed_data(PROCESSED_DATA)

    # Split off a holdout test set to evaluate final chosen model
    X_full_train, X_holdout, y_full_train, y_holdout = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
    logger.info(f"Full train: {X_full_train.shape}, Holdout: {X_holdout.shape}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # ---------- Random Forest search ----------
    logger.info("Starting RandomForest RandomizedSearchCV...")
    rf = RandomForestClassifier(random_state=SEED, n_jobs=1)
    rf_dist = rf_param_dist()
    rf_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=rf_dist,
        n_iter=N_ITER_RF,
        scoring="roc_auc",
        cv=skf,
        verbose=2,
        random_state=SEED,
        n_jobs=N_JOBS,
        return_train_score=True
    )
    t0 = time.time()
    rf_search.fit(X_full_train, y_full_train)
    t_rf = time.time() - t0
    logger.info(f"RF RandomSearch finished in {t_rf/60:.2f} min. Best ROC AUC (cv): {rf_search.best_score_:.4f}")
    # save rf cv results
    save_cv_results(rf_search.cv_results_, RF_CV_OUT)

    # ---------- XGBoost search ----------
    logger.info("Starting XGBoost RandomizedSearchCV...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=SEED, n_jobs=1)  # n_jobs=1 here; RandomizedSearchCV will parallelize
    xgb_dist = xgb_param_dist()
    xgb_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=xgb_dist,
        n_iter=N_ITER_XGB,
        scoring="roc_auc",
        cv=skf,
        verbose=2,
        random_state=SEED,
        n_jobs=N_JOBS,
        return_train_score=True
    )
    t0 = time.time()
    xgb_search.fit(X_full_train, y_full_train)
    t_xgb = time.time() - t0
    logger.info(f"XGB RandomSearch finished in {t_xgb/60:.2f} min. Best ROC AUC (cv): {xgb_search.best_score_:.4f}")
    # save xgb cv results
    save_cv_results(xgb_search.cv_results_, XGB_CV_OUT)

    # ---------- Choose best estimator ----------
    if xgb_search.best_score_ >= rf_search.best_score_:
        best_model = xgb_search.best_estimator_
        best_model_name = "xgboost"
        best_cv_score = float(xgb_search.best_score_)
        best_params = xgb_search.best_params_
        # save cv_results tidied summary
    else:
        best_model = rf_search.best_estimator_
        best_model_name = "random_forest"
        best_cv_score = float(rf_search.best_score_)
        best_params = rf_search.best_params_

    logger.info(f"Selected best model: {best_model_name} with CV ROC AUC: {best_cv_score:.4f}")

    # ---------- Fit best model on full training data ----------
    logger.info("Fitting best model on full training set (X_full_train)...")
    best_model.set_params(random_state=SEED)
    best_model.fit(X_full_train, y_full_train)

    # Save model artifact
    joblib.dump(best_model, MODEL_OUT)
    logger.info(f"Saved best model to {MODEL_OUT}")

    # ---------- Evaluate on holdout set ----------
    probs_holdout = best_model.predict_proba(X_holdout)[:, 1]
    auc_holdout = roc_auc_score(y_holdout, probs_holdout)
    logger.info(f"Holdout ROC AUC: {auc_holdout:.4f}")

    # ---------- Threshold search using business cost ----------
    thresholds = np.linspace(0.01, 0.99, 99)
    cost_df = compute_expected_cost(y_holdout, probs_holdout, thresholds, C_fp=C_FP, C_fn=C_FN)
    cost_df.to_csv(THRESH_COSTS_OUT, index=False)
    best = cost_df.loc[cost_df['cost'].idxmin()]
    best_threshold = float(best['threshold'])
    logger.info(f"Best threshold by business cost on holdout: {best_threshold} (cost={best['cost']})")

    # ---------- SHAP explainer ----------
        # ---------- SHAP explainer ----------
    try:
        logger.info("Building SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(best_model)

        # sample training rows for memory
        sample_for_shap = pd.DataFrame(X_full_train).sample(
            n=min(2000, X_full_train.shape[0]),
            random_state=SEED
        )

        # precompute a shap_values call (old & new API handling)
        try:
            _ = explainer.shap_values(sample_for_shap)
        except Exception:
            _ = explainer(sample_for_shap)  # newer SHAP API

        joblib.dump(explainer, SHAP_EXPLAINER_OUT)
        logger.info(f"Saved SHAP explainer to {SHAP_EXPLAINER_OUT}")

    except Exception as e:
        logger.exception("SHAP explainer generation failed: %s", e)

    # ---------- Save metadata ----------
    metadata = {
    "model_path": str(MODEL_OUT),
    "model_type": best_model_name,
    "holdout_roc_auc": float(auc_holdout),
    "cv_roc_auc": float(best_cv_score),
    "best_params": best_params,
    "threshold": best_threshold,
    "business_cost": {"C_fp": C_FP, "C_fn": C_FN},
    "rf_cv_results": str(RF_CV_OUT),
    "xgb_cv_results": str(XGB_CV_OUT),
    "threshold_costs": str(THRESH_COSTS_OUT),
    "shap_explainer": str(SHAP_EXPLAINER_OUT)
}

    with open(METADATA_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {METADATA_OUT}")

    logger.info("Training pipeline completed.")

if __name__ == "__main__":
    train_and_tune()
