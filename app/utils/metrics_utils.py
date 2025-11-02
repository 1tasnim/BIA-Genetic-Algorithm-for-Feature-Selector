import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import numpy as np
import base64, io

def evaluate_classification_metrics(df, target_col):
    """تقييم الأداء عبر مصفوفة الالتباس وF1 وPrecision"""
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]))
    X = X.select_dtypes(include='number')
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=60, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    metrics_df = pd.DataFrame(report).T.round(3)

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    conf_plot64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return metrics_df.to_html(classes='table table-bordered table-sm text-center', border=0), conf_plot64
