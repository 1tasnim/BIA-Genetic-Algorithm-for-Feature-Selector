import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def safe_read_csv(file_storage):
    try_encodings = ["utf-8", "cp1256", "latin-1"]
    for enc in try_encodings:
        try:
            file_storage.stream.seek(0)
            df = pd.read_csv(file_storage, encoding=enc)
            return df
        except Exception:
            continue
    file_storage.stream.seek(0)
    df = pd.read_csv(file_storage, sep=None, engine="python")
    return df

def impute_missing(df):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mean())
        else:
            if df[c].isna().any():
                mode = df[c].mode(dropna=True)
                fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                df[c] = df[c].fillna(fill_val)
    return df

def detect_target_column(df):
    common_targets = ["target", "label", "class", "y", "outcome", "diagnosis"]
    for name in df.columns:
        if name.strip().lower() in common_targets:
            return name
    candidate = None
    best_card = 1e9
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if 2 <= n_unique <= 10:
            if n_unique < best_card:
                best_card = n_unique
                candidate = col
    return candidate if candidate is not None else df.columns[-1]

def encode_features(X):
    # One-Hot للفئات + إبقاء الأرقام
    return pd.get_dummies(X, drop_first=True)

def split_scale(X_enc, y_enc, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=test_size, random_state=random_state,
        stratify=y_enc if len(np.unique(y_enc)) > 1 else None
    )
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test

def summarize_dataframe(df):
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_perc": (df.isna().mean() * 100).round(2).to_dict(),
    }
    desc_num = df.select_dtypes(include=[np.number]).describe().T
    desc_num_html = desc_num.to_html(classes="data-table", border=0)
    return info, desc_num_html
