import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Lasso


from .data import impute_missing, encode_features, split_scale

# ====================   الخوارزمية الجينية GA =====================

def init_population(n_individuals, n_features, rng=None):
    rng = rng or random
    pop = []
    for _ in range(n_individuals):
        chrom = [rng.choice([0, 1]) for _ in range(n_features)]
        if not any(chrom):
            chrom[rng.randrange(n_features)] = 1
        pop.append(chrom)
    return pop

def build_model(name, random_state=42):
    name = (name or "logistic").lower()
    if name in ["logistic", "lr", "logreg"]:
        return LogisticRegression(max_iter=2000)
    if name in ["lda", "linear discriminant analysis"]:
        return LinearDiscriminantAnalysis()
    if name in ["tree", "decisiontree", "decision tree"]:
        return DecisionTreeClassifier(random_state=random_state)
    if name in ["rf", "randomforest", "random forest"]:
        return RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1)
    if name in ["svm", "svm_linear", "svm-linear", "linear_svm"]:
        return SVC(kernel="linear", C=1.0, probability=False)
    return LogisticRegression(max_iter=2000)

def fitness_of(chrom, df, feature_cols, y, model_name="logistic", random_state=42):
    selected = [f for f, b in zip(feature_cols, chrom) if b == 1]
    if not selected:
        return 0.0

    X = df[selected].copy()
    X = impute_missing(X)

    # ترميز الميزات
    X_enc = encode_features(X)
    # احتفاظ بالأعمدة العددية فقط بعد الترميز
    X_enc = X_enc.select_dtypes(include=[np.number])
    if X_enc.shape[1] == 0:
        return 0.0
    if not np.isfinite(X_enc.values).all():
        return 0.0

    # ترميز الهدف
    y_series = y if isinstance(y, pd.Series) else pd.Series(y)
    if y_series.dtype == "object" or not pd.api.types.is_numeric_dtype(y_series):
        y_enc, _ = pd.factorize(y_series)
    else:
        y_enc = y_series.values

    # يجب توافر صنفين على الأقل
    if len(np.unique(y_enc)) < 2:
        return 0.0

    # تقسيم وتقييس
    try:
        X_train_sc, X_test_sc, y_train, y_test = split_scale(X_enc, y_enc, random_state=random_state)
    except ValueError:
        return 0.0

    model = build_model(model_name, random_state=random_state)
    try:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        return float(accuracy_score(y_test, y_pred))
    except Exception:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_enc, y_enc, test_size=0.3, random_state=random_state,
                stratify=y_enc if len(np.unique(y_enc)) > 1 else None
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return float(accuracy_score(y_test, y_pred))
        except Exception:
            return 0.0

# Roulette Wheel Selection
def roulette_wheel_selection(pop, fitnesses, rng=None):
    rng = rng or random
    fits = np.array(fitnesses, dtype=float)
    total = fits.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(pop)) / len(pop)
    else:
        probs = fits / total
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return pop[i][:]
    return pop[-1][:]

def single_point_crossover(p1, p2, cx_rate=0.9, rng=None):
    rng = rng or random
    if rng.random() > cx_rate or len(p1) < 2:
        return p1[:], p2[:]
    point = rng.randrange(1, len(p1))
    c1 = p1[:point] + p2[point:]
    c2 = p2[:point] + p1[point:]
    if not any(c1):
        c1[rng.randrange(len(c1))] = 1
    if not any(c2):
        c2[rng.randrange(len(c2))] = 1
    return c1, c2

def mutate(chrom, mut_rate=0.02, rng=None):
    rng = rng or random
    for i in range(len(chrom)):
        if rng.random() < mut_rate:
            chrom[i] = 1 - chrom[i]
    if not any(chrom):
        chrom[rng.randrange(len(chrom))] = 1
    return chrom

def genetic_feature_selection(
    df, target_col, pop_size=20, generations=25,
    cx_rate=0.9, mut_rate=0.02, elitism=2, random_state=42,
    selection_method="roulette",  
    model_name="logistic"         
):
    rng = random.Random(random_state)
    y = df[target_col]
    feature_cols = [c for c in df.columns if c != target_col]
    n_features = len(feature_cols)
    if n_features == 0:
        return [], float("nan"), []

    pop = init_population(pop_size, n_features, rng=rng)

    history = []
    best_chrom = None
    best_fit = -1.0

    for gen in range(1, generations + 1):
        fitnesses = [
            fitness_of(chrom, df, feature_cols, y, model_name=model_name, random_state=random_state)
            for chrom in pop
        ]

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = float(fitnesses[gen_best_idx])
        gen_best_features = [f for f, b in zip(feature_cols, pop[gen_best_idx]) if b == 1]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_chrom = pop[gen_best_idx][:]

        stats = {
            "AVG": float(np.mean(fitnesses)),
            "STD": float(np.std(fitnesses)),
            "MIN": float(np.min(fitnesses)),
            "MAX": float(np.max(fitnesses))
        }
        history.append({
            "generation": gen,
            "score": round(gen_best_fit, 4),
            "AVG": round(stats["AVG"], 4),
            "STD": round(stats["STD"], 4),
            "MIN": round(stats["MIN"], 4),
            "MAX": round(stats["MAX"], 4),
            "features": gen_best_features
        })

        elite_indices = list(np.argsort(fitnesses))[::-1][:elitism]
        new_pop = [pop[i][:] for i in elite_indices]

        def select_one():
            if selection_method == "roulette":
                return roulette_wheel_selection(pop, fitnesses, rng=rng)
            # tournament fallback
            k = min(3, len(pop))
            idxs = [rng.randrange(len(pop)) for _ in range(k)]
            best_idx = max(idxs, key=lambda i: fitnesses[i])
            return pop[best_idx][:]

        while len(new_pop) < pop_size:
            p1 = select_one()
            p2 = select_one()
            c1, c2 = single_point_crossover(p1, p2, cx_rate=cx_rate, rng=rng)
            c1 = mutate(c1, mut_rate=mut_rate, rng=rng)
            if len(new_pop) < pop_size:
                new_pop.append(c1)
            c2 = mutate(c2, mut_rate=mut_rate, rng=rng)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop

    best_features = [f for f, b in zip(feature_cols, best_chrom) if b == 1] if best_chrom else []
    return history, float(best_fit), best_features


# ===================== Chi2 Feature Selection =====================
def chi2_feature_selection(df, target_col, k=10):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # تعويض القيم المفقودة
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "missing"
            X[col] = X[col].fillna(mode_val)

    # ترميز المتغيرات الفئوية
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    # فقط الميزات الرقمية
    X = X.select_dtypes(include='number')

    # التأكد من عدم وجود NaN بعد الترميز
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # مقياس القيم بين 0 و 1 لأن chi2 يحتاج قيم موجبة
    X_scaled = MinMaxScaler().fit_transform(X)

    # تطبيق اختبار Chi2
    chi2_selector = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
    chi2_selector.fit(X_scaled, y)
    scores = chi2_selector.scores_
    feature_names = X.columns

    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, scores, color='skyblue')
    plt.title("Chi² Feature Importance")
    plt.xlabel("Score")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    chi_plot64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    top_features = feature_names[np.argsort(scores)[::-1][:k]].tolist()
    return top_features, scores, chi_plot64


# ===================== Lasso / Logistic L1 Feature Selection =====================
def lasso_feature_selection(df, target_col):
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # تعويض القيم المفقودة
    X = X.copy()
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].mean())
        else:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else "missing"
            X[col] = X[col].fillna(mode_val)

    # ترميز الفئات
    X = pd.get_dummies(X, drop_first=True, dtype=float)

    # إزالة القيم غير المنتهية أو المفقودة
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # توحيد القيم (مهم للـ L1)
    X_scaled = StandardScaler().fit_transform(X)

    # تدريب نموذج Logistic L1
    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
    model.fit(X_scaled, y)

    coef = np.abs(model.coef_[0])
    importance = pd.Series(coef, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    importance.head(15).plot(kind='barh', color='purple')
    plt.title("Lasso (L1) Feature Coefficients")
    plt.xlabel("|Coefficient|")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    lasso_plot64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    top_features = importance.head(10).index.tolist()
    return top_features, coef, lasso_plot64


