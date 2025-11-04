import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_heatmap(df):
    """خريطة الارتباطات بين الميزات الرقمية"""
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(7,5))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _fig_to_b64(dpi=120, bbox=True):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if bbox else None)
    buf.seek(0)
    img64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img64

def plot_top_numeric_distributions(df, max_cols=4):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return None
    top = num_cols[:max_cols]
    plt.figure(figsize=(8, 6))
    for i, c in enumerate(top, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df[c].dropna(), kde=True, bins=30, color="#0B3954")
        plt.title(c)
    plt.tight_layout()
    return _fig_to_b64()

def plot_target_distribution(y, target_name):
    plt.figure(figsize=(5, 4))
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
        sns.histplot(y.dropna(), kde=True, bins=30, color="#8E6C88")
    else:
        sns.countplot(x=y.astype(str), color="#8E6C88")
        plt.xticks(rotation=20, ha="right")
    plt.title(f"Distribution of {target_name}")
    plt.tight_layout()
    return _fig_to_b64()

def make_bar_plot(performance_dict, title):
    plt.figure(figsize=(6, 4))
    names = list(performance_dict.keys())
    vals = list(performance_dict.values())
    sns.barplot(x=names, y=vals, palette="coolwarm")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.tight_layout()
    return _fig_to_b64()

def make_line_plot(xs, ys, title, xlabel, ylabel):
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=xs, y=ys, marker="o", color="seagreen")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    return _fig_to_b64()
