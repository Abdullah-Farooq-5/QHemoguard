import pandas as pd
import numpy as np

FEATURE_COLS = ['GENDER', 'RBC Count (mil/cmm)', 'Hemoglobin Level (g/dL)',
                 'Hematocrit (%)', 'MCV (fL)', 'MCH (pg)', 'MCHC (g/dL)',
                 'RDW-CV (%)', 'Hemoglobin A (%)', 'Hemoglobin A2 (%)']
LABEL_COL = 'Hematologist Remarks'

# Two experiment feature sets, per Dr. Madiha's request
TOP5_FEATURES_FULL = ['Hemoglobin A2 (%)', 'Hemoglobin A (%)', 'RBC Count (mil/cmm)', 'MCV (fL)', 'MCH (pg)']
CBC_ONLY_FEATURES  = ['RBC Count (mil/cmm)', 'Hemoglobin Level (g/dL)', 'Hematocrit (%)', 'MCV (fL)', 'MCH (pg)']

def load_raw_data(path="../data/Hb. Electrophoresis Data from 13-08-2020 to 24-06-2024 Preprocessed Train All with HB.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['GENDER'] = df['GENDER'].map({'Female': 0, 'Male': 1})
    return df

def rescale_features(df, feature_cols):
    """Rescale each feature to (0, 2*pi] for angle/phase encoding."""
    df = df.copy()
    for col in feature_cols:
        df[col] = 2*np.pi * (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

def get_X_y(df, feature_cols, label_col=LABEL_COL):
    X = df[feature_cols].values
    y = df[label_col].values
    return X, y