
from flask import Blueprint, render_template, request, make_response
import pandas as pd
import numpy as np

# استيراد الدوال مباشرة دون استخدام __init__.py
from .utils.data import safe_read_csv, impute_missing, detect_target_column, summarize_dataframe, split_scale
from .utils.plotting import plot_top_numeric_distributions, plot_target_distribution, make_bar_plot, make_line_plot, plot_heatmap 
from .utils.ga import genetic_feature_selection, chi2_feature_selection, lasso_feature_selection
from .utils.models import train_baselines
from .utils.pca_analysis import run_pca
from .utils.metrics_utils import evaluate_classification_metrics


bp = Blueprint("main", __name__)

@bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    file = request.files.get("csv_file")
    if not file:
        return "<h2> لم يتم استلام ملف</h2>"

    try:
        df = safe_read_csv(file)
    except Exception as e:
        return f"<h2> خطأ في قراءة الملف: {e}</h2>"

    if df.empty or df.shape[1] < 2:
        return "<h2> الملف فارغ أو لا يحتوي ميزات كافية</h2>"

    df = impute_missing(df)

    info, desc_num_html = summarize_dataframe(df)

    target_hint = request.form.get("target_hint", "").strip()
    target_col = target_hint if target_hint and target_hint in df.columns else detect_target_column(df)

    rows, cols = df.shape
    num_cols_count = df.select_dtypes(include=[np.number]).shape[1]
    cat_cols_count = df.select_dtypes(exclude=[np.number]).shape[1]
    preview_html = df.head(200).to_html(classes="data-table", index=False, border=0)
    hist_num64 = plot_top_numeric_distributions(df)
    target_dist64 = plot_target_distribution(df[target_col], target_col)

    # إعدادات GA من النموذج (يمكنك إضافة حقول في الواجهة للخيارات)
    def get_num(name, default, t=float):
        val = request.form.get(name, str(default))
        try:
            return t(val)
        except Exception:
            return default

    pop_size = int(get_num("pop_size", 20, int))
    generations = int(get_num("generations", 25, int))
    cx_rate = float(get_num("cx_rate", 0.9, float))
    mut_rate = float(get_num("mut_rate", 0.02, float))
    elitism = int(get_num("elitism", 2, int))


    # خيار اختيار النموذج داخل GA
    ga_model_name = (request.form.get("ga_model", "logistic") or "logistic").strip()

    # حراسة القيم
    pop_size = max(4, min(pop_size, 200))
    generations = max(5, min(generations, 200))
    cx_rate = min(max(cx_rate, 0.0), 1.0)
    mut_rate = min(max(mut_rate, 0.0), 1.0)
    elitism = max(0, min(elitism, max(1, pop_size // 2)))

    # تشغيل الخوارزمية الجينية
    ga_history, ga_best, ga_best_features = genetic_feature_selection(
        df, target_col, pop_size=pop_size, generations=generations,
        cx_rate=cx_rate, mut_rate=mut_rate, elitism=elitism,
        random_state=42, selection_method="roulette",
        model_name=ga_model_name
    )

      # خريطة الارتباطات
    heatmap64 = plot_heatmap(df) 

    # نماذج الأساس
    performance = train_baselines(df, target_col)
    perf_for_plot = dict(performance)
    perf_for_plot["Genetic Algorithm (Best)"] = ga_best

     # Chi-square
    chi_features, chi_scores, chi_plot64 = chi2_feature_selection(df, target_col)

    # Lasso / Logistic L1
    lasso_features, lasso_scores, lasso_plot64 = lasso_feature_selection(df, target_col)

    # PCA
    pca_info, pca_var64, pca_2d64, pca_3d64 = run_pca(df, target_col)
     
    # تقارير الأداء
    class_metrics, conf64 = evaluate_classification_metrics(df, target_col)

    # رسم المقارنة
    perf_for_plot = dict(performance)
    perf_for_plot["Genetic Algorithm (Best)"] = ga_best
    graph_models64 = make_bar_plot(perf_for_plot, "Comparison of Models vs Genetic Algorithm")

    # خط تطور دقة GA عبر الأجيال
    ga_scores = [h["score"] for h in ga_history] if ga_history else []
    ga_line64 = make_line_plot(
        xs=list(range(1, len(ga_scores) + 1)),
        ys=ga_scores,
        title="GA Best Score per Generation",
        xlabel="Generation",
        ylabel="Accuracy",
    ) if ga_scores else None

    return render_template(
        "result.html",
        rows=rows, cols=cols, target=target_col,
        columns=info["columns"], dtypes=info["dtypes"], missing_perc=info["missing_perc"],
        desc_num_html=desc_num_html,
        num_cols=num_cols_count, cat_cols=cat_cols_count,
        preview=preview_html,
        hist_num=hist_num64,
        target_dist=target_dist64,
        generations=ga_history,
        best_score=ga_best,
        best_features=ga_best_features,
        performance=performance,
        graph_models=graph_models64,
        ga_line=ga_line64,
        chi_features=chi_features, chi_plot=chi_plot64,
        lasso_features=lasso_features, lasso_plot=lasso_plot64,
        pca_info=pca_info, pca_var=pca_var64, pca_2d=pca_2d64, pca_3d=pca_3d64,
        class_metrics=class_metrics, conf_plot=conf64,
        heatmap=heatmap64,
    )

@bp.route("/download_features", methods=["POST"])
def download_features():
    features = request.form.get("features", "")
    features_list = [f for f in features.split(",") if f]
    features_df = pd.DataFrame(features_list, columns=["Selected Features"])

    csv = features_df.to_csv(index=False, encoding="utf-8-sig")
    resp = make_response(csv)
    resp.headers["Content-Disposition"] = "attachment; filename=selected_features.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp
