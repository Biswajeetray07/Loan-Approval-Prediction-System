# app/streamlit_app.py
import streamlit as st
import pandas as pd
import json
import os
import sys
import pathlib

# Ensure repo root is on sys.path (helps when Streamlit invoked from different CWD)
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.inference import predict_single, predict_batch

st.set_page_config(page_title="Loan Approval App", layout="wide")
st.title("Loan Approval — Demo")

meta_path = repo_root / "artifacts" / "metadata.json"
metadata = {}
if meta_path.exists():
    with open(meta_path) as f:
        metadata = json.load(f)

st.sidebar.header("Model info")
st.sidebar.text(f"Model: {metadata.get('model_type', 'unknown')}")
st.sidebar.text(f"Threshold: {metadata.get('threshold', 0.5)}")
st.sidebar.text(f"Holdout AUC: {metadata.get('holdout_roc_auc', 'n/a')}")

st.header("Single applicant prediction")
st.markdown("Enter applicant features (raw values). If you have not implemented feature engineering, use the processed column names found in `data/processed/processed_loan_data.csv`.")

processed_path = repo_root / "data" / "processed" / "processed_loan_data.csv"
if processed_path.exists():
    proc_df = pd.read_csv(processed_path, nrows=1)
    if "target" in proc_df.columns:
        proc_cols = [c for c in proc_df.columns if c != "target"]
    else:
        proc_cols = list(proc_df.columns)
else:
    proc_cols = []

single_input = {}
with st.form("single_form"):
    if proc_cols:
        st.markdown("Fill values for processed features (first 20 shown).")
        for c in proc_cols[:20]:
            single_input[c] = st.text_input(c, value="")
    else:
        st.markdown("No processed schema found — enter JSON-like key/value under 'Manual input' below.")
        raw_kv = st.text_area("Manual input (JSON)", value='{"person_age": 30, "person_income": 500000, "loan_amnt": 200000}')
    submitted = st.form_submit_button("Predict single")
    if submitted:
        if not proc_cols:
            try:
                raw_input = json.loads(raw_kv)
            except Exception as e:
                st.error(f"Failed to parse JSON input: {e}")
                raw_input = {}
        else:
            raw_input = {}
            for k,v in single_input.items():
                if v == "":
                    raw_input[k] = 0.0
                else:
                    try:
                        val = float(v)
                        raw_input[k] = val
                    except:
                        raw_input[k] = v

        try:
            res = predict_single(raw_input, top_n=8)
            st.metric("Probability of default", f"{res['probability_default']:.4f}")
            st.markdown(f"### Decision: **{res['decision']}** (threshold={res.get('threshold',0.5)})")
            if res.get("top_features"):
                st.subheader("Top SHAP features")
                tf = pd.DataFrame(res["top_features"])
                st.dataframe(tf)
            else:
                st.info("No SHAP explanation available for this model.")
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Run training first: `python src/train_model.py` from repo root, then re-run this app.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown("---")
st.header("Batch prediction (CSV upload)")
st.markdown("Upload a CSV of raw rows. The pipeline will try to apply feature engineering automatically.")
uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])
if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(df_raw.head())
    if st.button("Run batch predictions"):
        try:
            out = predict_batch(df_raw)
            st.success("Predictions completed.")
            st.dataframe(out.head(50))
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions", csv_bytes, "predictions.csv")
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("Run training first: `python src/train_model.py` from repo root, then re-run this app.")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.markdown("---")
st.header("Evaluation artifacts")
eval_metrics_path = repo_root / "artifacts" / "evaluation" / "metrics_summary.json"
if eval_metrics_path.exists():
    with open(eval_metrics_path) as f:
        m = json.load(f)
    st.json(m)
    roc = repo_root / "artifacts" / "evaluation" / "roc_curve.png"
    pr = repo_root / "artifacts" / "evaluation" / "pr_curve.png"
    if roc.exists(): st.image(str(roc))
    if pr.exists(): st.image(str(pr))
else:
    st.info("Run evaluation pipeline first (`python src/evaluate_model.py`) to generate metrics and plots.")
