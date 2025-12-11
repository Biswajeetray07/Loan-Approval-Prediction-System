# src/inference.py
import json
import joblib
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger("inference")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Resolve repo-root-relative artifact paths so code works regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo/src -> parents[1] == repo root
MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
PREPROCESSOR_PATH = REPO_ROOT / "artifacts" / "preprocessor.joblib"
EXPLAINER_PATH = REPO_ROOT / "artifacts" / "shap_explainer.joblib"
METADATA_PATH = REPO_ROOT / "artifacts" / "metadata.json"
PROCESSED_DATA = REPO_ROOT / "data" / "processed" / "processed_loan_data.csv"

def _load_artifacts():
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not PREPROCESSOR_PATH.exists():
        missing.append(str(PREPROCESSOR_PATH))
    if missing:
        msg = (
            "Missing required artifact(s):\n  - " + "\n  - ".join(missing) +
            "\n\nMake sure you ran training and that artifacts are saved.\n"
            "Run (from repo root):\n  python src/train_model.py\n"
            "Then re-run the app from repo root:\n  streamlit run app/streamlit_app.py\n"
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    model = joblib.load(str(MODEL_PATH))
    preprocessor = joblib.load(str(PREPROCESSOR_PATH))

    explainer = None
    if EXPLAINER_PATH.exists():
        try:
            explainer = joblib.load(str(EXPLAINER_PATH))
        except Exception as e:
            logger.warning("Failed to load SHAP explainer: %s", e)
            explainer = None

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            metadata = json.load(f)

    logger.info(f"Loaded artifacts from repo root: {REPO_ROOT}")
    return model, preprocessor, explainer, metadata

def _apply_cleaning(raw_input: Dict[str,Any]) -> pd.DataFrame:
    try:
        from data_preprocessing import apply_cleaning_and_feature_engineering
        df_raw = pd.DataFrame([raw_input])
        return apply_cleaning_and_feature_engineering(df_raw)
    except Exception as e:
        logger.info("apply_cleaning_and_feature_engineering not available or failed: %s. Assuming input engineered.", e)
        return pd.DataFrame([raw_input])

def _align_to_expected(df_engineered: pd.DataFrame) -> pd.DataFrame:
    if PROCESSED_DATA.exists():
        proc = pd.read_csv(str(PROCESSED_DATA))
        expected_cols = [c for c in proc.columns if c != "target"]
        return df_engineered.reindex(columns=expected_cols, fill_value=0)
    return df_engineered

def predict_single(raw_input: Dict[str,Any], top_n:int=5) -> Dict[str,Any]:
    model, preprocessor, explainer, metadata = _load_artifacts()

    df_engineered = _apply_cleaning(raw_input)
    df_aligned = _align_to_expected(df_engineered)

    try:
        X = preprocessor.transform(df_aligned)
    except Exception as e:
        logger.warning("Preprocessor.transform failed: %s. Falling back to df_aligned.values", e)
        X = df_aligned.values

    proba = float(model.predict_proba(X)[:,1][0])
    threshold = float(metadata.get("threshold", 0.5))
    decision = "REJECT" if proba >= threshold else "APPROVE"
    result = {"probability_default": proba, "decision": decision, "threshold": threshold}

    top_features = []
    if explainer is not None:
        try:
            # handle old/new SHAP API
            try:
                shap_vals = explainer.shap_values(X)
            except Exception:
                shap_vals = explainer(X)

            # determine feature names
            try:
                proc = pd.read_csv(str(PROCESSED_DATA))
                feat_names = [c for c in proc.columns if c != "target"]
            except Exception:
                try:
                    num_cols = preprocessor.transformers_[0][2]
                    cat_cols = preprocessor.transformers_[1][2]
                    cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
                    feat_names = list(num_cols) + list(cat_names)
                except Exception:
                    feat_names = [f"f_{i}" for i in range(np.array(shap_vals).shape[1])]

            sample_shap = np.array(shap_vals[0])
            idx = np.argsort(np.abs(sample_shap))[::-1][:top_n]
            for i in idx:
                top_features.append({
                    "feature": feat_names[i],
                    "shap_value": float(sample_shap[i]),
                    "impact": "+" if sample_shap[i] > 0 else "-"
                })
            result["top_features"] = top_features
            result["shap_values"] = sample_shap.tolist()
        except Exception as e:
            logger.exception("SHAP explanation failed: %s", e)
            result["top_features"] = []
            result["shap_values"] = None
    else:
        result["top_features"] = []
        result["shap_values"] = None

    return result

def predict_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    model, preprocessor, explainer, metadata = _load_artifacts()
    try:
        from data_preprocessing import apply_cleaning_and_feature_engineering
        df_engineered = apply_cleaning_and_feature_engineering(df_raw)
    except Exception as e:
        logger.info("Batch engineering not applied: %s. Assuming dataframe already engineered.", e)
        df_engineered = df_raw

    df_aligned = _align_to_expected(df_engineered)
    try:
        X = preprocessor.transform(df_aligned)
    except Exception as e:
        logger.warning("Preprocessor.transform failed for batch: %s. Using df_aligned.values", e)
        X = df_aligned.values

    probs = model.predict_proba(X)[:,1]
    threshold = float(metadata.get("threshold", 0.5))
    decisions = np.where(probs >= threshold, "REJECT", "APPROVE")

    out = df_raw.copy().reset_index(drop=True)
    out["probability_default"] = probs
    out["decision"] = decisions
    return out

if __name__ == "__main__":
    example = {}
    if PROCESSED_DATA.exists():
        df_proc = pd.read_csv(str(PROCESSED_DATA))
        for c in df_proc.columns[:20]:
            if c != "target":
                example[c] = 0.0
    try:
        res = predict_single(example)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print("Predict failed:", e)
