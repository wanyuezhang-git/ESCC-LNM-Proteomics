import json
import warnings
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def format_median_iqr(values):
    arr = np.array(values, dtype=float)
    med = np.median(arr)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    return f"{med:.3f} ({q1:.3f}, {q3:.3f})"


def get_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def main():
    data_path = "clinical_gene_CT_clinical.csv"
    output_metrics = "nested_cv_metrics_summary.csv"
    output_metrics_raw = "nested_cv_metrics_raw.csv"
    output_params = "nested_cv_best_params.csv"
    output_plot = "nested_cv_combined_roc.png"

    features = ["OBSCN", "CEMIP", "PFN2", "TMBIM1", "P4HA2", "ITGB4", "SURF4", "BPI"]
    target = "N stage"

    df = pd.read_csv(data_path)
    X = df[features].copy()
    y = df[target].astype(int).copy()

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    models = {
        "LR": (
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                C=0.1,  # fixed to enforce moderate regularization as requested
                random_state=42,
                max_iter=3000,
            ),
            {"clf__C": [0.1]},
        ),
        "SVM": (
            SVC(probability=True, random_state=42),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.1], "clf__kernel": ["rbf"]},
        ),
        "RF": (
            RandomForestClassifier(random_state=42, n_jobs=1),
            {"clf__n_estimators": [200], "clf__max_depth": [None, 5], "clf__min_samples_leaf": [1, 2]},
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=1,
                tree_method="hist",
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [2, 3],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=42, n_jobs=1, verbose=-1),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [-1, 5],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
        "CatBoost": (
            CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False, thread_count=1),
            {
                "clf__iterations": [200],
                "clf__depth": [3, 5],
                "clf__learning_rate": [0.03, 0.1],
                "clf__l2_leaf_reg": [3, 5],
            },
        ),
        "GBDT": (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.5, 1.0]},
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(random_state=42, n_jobs=1),
            {"clf__n_estimators": [200], "clf__max_depth": [None, 5], "clf__min_samples_leaf": [1, 2]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
        ),
        "MLP": (
            MLPClassifier(random_state=42, max_iter=3000),
            {
                "clf__hidden_layer_sizes": [(16,), (32,), (16, 8)],
                "clf__alpha": [1e-4, 1e-3],
            },
        ),
        "GaussianNB": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
    }

    metric_store = defaultdict(list)
    raw_rows = []
    best_params_rows = []
    roc_pool = defaultdict(lambda: {"y_true": [], "y_score": []})

    for model_name, (estimator, param_grid) in models.items():
        print(f"Running nested CV for {model_name} ...")
        fold_best_params = []

        for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(estimator))])
            grid = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_proba = best_model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            auc = roc_auc_score(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            sens = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0.0
            spec = get_specificity(y_test, y_pred)

            metric_store[model_name].append(
                {
                    "AUC": auc,
                    "Accuracy": acc,
                    "Precision": prec,
                    "F1 score": f1,
                    "Sensitivity": sens,
                    "Specificity": spec,
                }
            )

            raw_rows.append(
                {
                    "Model": model_name,
                    "Outer Fold": fold_id,
                    "AUC": auc,
                    "Accuracy": acc,
                    "Precision": prec,
                    "F1 score": f1,
                    "Sensitivity": sens,
                    "Specificity": spec,
                }
            )

            fold_best_params.append(grid.best_params_)
            best_params_rows.append(
                {
                    "Model": model_name,
                    "Outer Fold": fold_id,
                    "Best Params": json.dumps(grid.best_params_, ensure_ascii=True),
                }
            )

            roc_pool[model_name]["y_true"].extend(y_test.tolist())
            roc_pool[model_name]["y_score"].extend(y_proba.tolist())

        params_as_text = [json.dumps(p, sort_keys=True, ensure_ascii=True) for p in fold_best_params]
        dominant = Counter(params_as_text).most_common(1)[0][0]
        best_params_rows.append({"Model": model_name, "Outer Fold": "Dominant", "Best Params": dominant})

    summary_rows = []
    metric_names = ["AUC", "Accuracy", "Precision", "F1 score", "Sensitivity", "Specificity"]
    for model_name in models.keys():
        row = {"Model": model_name}
        for metric in metric_names:
            metric_values = [x[metric] for x in metric_store[model_name]]
            row[metric] = format_median_iqr(metric_values)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    raw_df = pd.DataFrame(raw_rows)
    params_df = pd.DataFrame(best_params_rows)

    summary_df.to_csv(output_metrics, index=False)
    raw_df.to_csv(output_metrics_raw, index=False)
    params_df.to_csv(output_params, index=False)

    # Morandi-inspired palette (soft but distinguishable)
    morandi_colors = [
        "#6D6875",
        "#B5838D",
        "#E5989B",
        "#99A799",
        "#7D8F69",
        "#A5A58D",
        "#8E9AAF",
        "#C9ADA7",
        "#84A59D",
        "#9C89B8",
        "#6C757D",
        "#52796F",
    ]

    plt.figure(figsize=(10, 8), dpi=150)
    for idx, model_name in enumerate(models.keys()):
        y_true_all = np.array(roc_pool[model_name]["y_true"])
        y_score_all = np.array(roc_pool[model_name]["y_score"])
        fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
        auc_val = roc_auc_score(y_true_all, y_score_all)
        plt.plot(
            fpr,
            tpr,
            color=morandi_colors[idx % len(morandi_colors)],
            linewidth=2,
            label=f"{model_name} (AUC={auc_val:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curves (10-fold Nested CV, 8-protein Panel)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()

    print("\nDone.")
    print(f"Saved: {output_metrics}")
    print(f"Saved: {output_metrics_raw}")
    print(f"Saved: {output_params}")
    print(f"Saved: {output_plot}")


if __name__ == "__main__":
    main()
