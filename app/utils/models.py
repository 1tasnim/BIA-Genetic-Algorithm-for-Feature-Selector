import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from .data import impute_missing, encode_features, split_scale

def train_baselines(df, target_col, random_state=42):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
        y_enc, _ = pd.factorize(y)
    else:
        y_enc = y.values if isinstance(y, pd.Series) else y

    if len(np.unique(y_enc)) < 2:
        return {
            "Logistic Regression": float("nan"),
            "Random Forest": float("nan"),
            "Gradient Boosting": float("nan"),
        }

    X = impute_missing(X)
    X_enc = encode_features(X)

    stratify=y_enc if len(np.unique(y_enc)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=0.3, random_state=random_state,
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
       "Random Forest": RandomForestClassifier(n_estimators=80, random_state=random_state),
       "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
       "Logistic Regression": LogisticRegression(max_iter=500)
    }
    performance = {}
    for name, model in models.items():
        try:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            performance[name] = float(accuracy_score(y_test, y_pred))
        except Exception:
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                performance[name] = float(accuracy_score(y_test, y_pred))
            except Exception:
                performance[name] = float("nan")

    return performance
