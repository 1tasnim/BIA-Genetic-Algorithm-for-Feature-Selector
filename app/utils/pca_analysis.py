import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import base64, io, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run_pca(df, target_col):
    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]))

    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "missing"
            X[col] = X[col].fillna(mode_val)
    
    X = X.select_dtypes(include='number')
    X_scaled = StandardScaler().fit_transform(X)
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)  # حماية إضافية ضد NaN

    pca = PCA(n_components=min(5, X_scaled.shape[1]))
    components = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_

    # مخطط التباين
    plt.figure(figsize=(6,4))
    plt.plot(np.cumsum(var_ratio)*100, marker='o')
    plt.title("Cumulative Explained Variance (%)")
    plt.xlabel("Number of Components")
    plt.ylabel("Variance (%)")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    pca_var64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 2D
    plt.figure(figsize=(6,5))
    plt.scatter(components[:,0], components[:,1], c=pd.factorize(y)[0], cmap='tab10')
    plt.title("PCA 2D Scatter")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    pca_2d64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 3D
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(components[:,0], components[:,1], components[:,2], c=pd.factorize(y)[0], cmap='tab10')
    ax.set_title("PCA 3D Visualization")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    pca_3d64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    info = {
        "explained_variance": np.round(var_ratio * 100, 2).tolist(),
        "total_variance": round(np.sum(var_ratio) * 100, 2)
    }
    return info, pca_var64, pca_2d64, pca_3d64
