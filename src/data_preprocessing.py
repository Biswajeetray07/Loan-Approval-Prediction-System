import os
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.debug(f"Loaded raw data from: {path}")
        return df
    except Exception as e:
        logger.error(f"Error while loading raw data: {e}")
        raise


def apply_cleaning_and_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    logger.debug("Starting outlier cleaning and feature engineering...")

    # Outlier handling
    df['person_age'] = df['person_age'].clip(18, 100)
    df['person_emp_exp'] = df['person_emp_exp'].clip(0, 60)
    income_cap = df['person_income'].quantile(0.99)
    df['person_income'] = np.where(df['person_income'] > income_cap,
                                   income_cap,
                                   df['person_income'])

    # Feature engineering
    df['person_income_log1p'] = np.log1p(df['person_income'])
    df['loan_percent_income_recomputed'] = df['loan_amnt'] / (df['person_income'] + 1e-9)
    df['age_bucket'] = pd.cut(df['person_age'],
                              bins=[18, 24, 30, 40, 60, 100],
                              labels=['18-24', '25-30', '31-40', '41-60', '60+'])

    df['credit_score_bin'] = pd.cut(df['credit_score'],
                                    bins=[300, 580, 670, 740, 800, 900],
                                    labels=['poor', 'fair', 'good', 'very_good', 'exceptional'],
                                    right=False)

    df['income_decile'] = pd.qcut(df['person_income'], 10,
                                  labels=False, duplicates='drop')

    logger.debug("Feature engineering completed.")
    return df


def apply_preprocessing(df: pd.DataFrame, save_preprocessor_path: str = "artifacts/preprocessor.joblib") -> pd.DataFrame:
    logger.debug("Starting preprocessing...")

    numeric_features = [
        'person_age', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
        'loan_percent_income_recomputed', 'cb_person_cred_hist_length',
        'credit_score', 'person_income_log1p'
    ]

    categorical_features = [
        'person_education', 'person_home_ownership', 'loan_intent',
        'previous_loan_defaults_on_file', 'age_bucket', 'credit_score_bin'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    # Fit the preprocessor and transform
    df_processed_np = preprocessor.fit_transform(df)

    # Column names
    num_cols = numeric_features
    cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    final_cols = np.concatenate([num_cols, cat_cols])

    df_processed = pd.DataFrame(df_processed_np, columns=final_cols)
    logger.debug("Preprocessing completed.")

    # ensure artifacts dir exists and save preprocessor
    os.makedirs(os.path.dirname(save_preprocessor_path) or ".", exist_ok=True)
    joblib.dump(preprocessor, save_preprocessor_path)
    logger.debug(f"Saved preprocessor to: {save_preprocessor_path}")

    return df_processed


def save_processed_data(df_processed: pd.DataFrame, save_dir: str) -> None:
    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "processed_loan_data.csv")
        df_processed.to_csv(save_path, index=False)
        logger.debug(f"Processed data saved at: {save_path}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


def main():
    try:
        raw_data_path = "data/raw/loan_data.csv"
        processed_data_dir = "data/processed"
        preproc_artifact_path = "artifacts/preprocessor.joblib"

        # 1. Load raw data
        df_raw = load_data(raw_data_path)

        # 2. Extract target column BEFORE preprocessing
        if "loan_status" in df_raw.columns:
            target = df_raw["loan_status"]              # if your target column is named loan_status
        elif "target" in df_raw.columns:
            target = df_raw["target"]
        else:
            raise ValueError("❌ Target column not found in raw dataset.")

        # 3. Apply cleaning + feature engineering
        df_features = apply_cleaning_and_feature_engineering(df_raw)

        # 4. Apply preprocessing transform (scaler + OHE)
        df_processed = apply_preprocessing(df_features, save_preprocessor_path=preproc_artifact_path)

        # 5. Add the target column back
        df_processed["target"] = target.values

        # 6. Save processed file
        os.makedirs(processed_data_dir, exist_ok=True)
        save_path = os.path.join(processed_data_dir, "processed_loan_data.csv")
        df_processed.to_csv(save_path, index=False)

        logger.debug(f"Processed data saved at: {save_path}")
        print("✔ Processed data created successfully.")

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise



if __name__ == "__main__":
    main()
