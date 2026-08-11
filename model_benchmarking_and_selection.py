

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:  
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:  
    CatBoostClassifier = None


SEED = 2026
OUTER_SPLITS = 5
OUTER_REPEATS = 10
INNER_SPLITS_FOR_MODEL_COMPARISON = 3
INNER_SPLITS_FOR_FINAL_MODEL = 5

DEFAULT_DATA_DIR = Path("/Users/zhangwy/Documents")
DEFAULT_KEY_GENES_FILE = "key genes.xlsx"
DEFAULT_EXPRESSION_FILE = "preprocess data.csv"
DEFAULT_SPLIT_FILE = "final_60_40_split.csv"


GAUSSIANNB_FINAL_GRID = {"clf__var_smoothing": np.logspace(-12, -3, 10).tolist()}


HIGH_NPV_MIN_SENSITIVITY = 0.50


@dataclass(frozen=True)
class CohortData:
    table: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_file_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_").lower()


def normalize_id_series(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace("\ufeff", "", regex=False)
    )


def read_key_genes(path: Path) -> list[str]:
    """Read exactly four protein names from key genes.xlsx:key genes."""
    if not path.exists():
        raise FileNotFoundError(f"Cannot find key gene file: {path}")
    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"{path.name} must be an Excel file.")
    raw = pd.read_excel(path)
    required_col = "key genes"
    if required_col not in raw.columns:
        raise ValueError(f"{path.name} must contain column '{required_col}'. Available columns: {list(raw.columns)}")
    genes = [str(v).strip() for v in raw[required_col].dropna().tolist() if str(v).strip()]
    if len(genes) != 4:
        raise ValueError(f"Expected exactly 4 proteins in {path.name}:{required_col}, found {len(genes)}: {genes}")
    if len(set(genes)) != 4:
        raise ValueError(f"Duplicated protein names found in {path.name}: {genes}")
    return genes


def ln_status_to_binary(values: pd.Series) -> pd.Series:
    labels = values.astype(str).str.strip()
    expected = {"LN_negative", "LN_positive"}
    unexpected = sorted(set(labels.dropna().unique()) - expected)
    if unexpected:
        raise ValueError(f"LN_status must contain only LN_negative/LN_positive. Unexpected values: {unexpected}")
    return labels.eq("LN_positive").astype(int)


def load_cohort(
    data_dir: Path,
    genes: list[str],
    cohort_name: str,
    expression_file: str,
    split_file: str,
) -> CohortData:
    expression = pd.read_csv(data_dir / expression_file, encoding="utf-8-sig")
    split = pd.read_csv(data_dir / split_file, encoding="utf-8-sig")

    required_expression_cols = ["Sample", *genes]
    missing_expression_cols = [col for col in required_expression_cols if col not in expression.columns]
    if missing_expression_cols:
        raise ValueError(f"{expression_file} is missing required columns: {missing_expression_cols}")

    required_split_cols = ["patient_id", "cohort_assignment", "LN_status"]
    missing_split_cols = [col for col in required_split_cols if col not in split.columns]
    if missing_split_cols:
        raise ValueError(f"{split_file} is missing required columns: {missing_split_cols}")

    expression = expression.copy()
    split = split.copy()
    expression["sample_id"] = normalize_id_series(expression["Sample"])
    split["sample_id"] = normalize_id_series(split["patient_id"])

    cohort_mask = split["cohort_assignment"].astype(str).str.strip().eq(cohort_name)
    cohort_rows = split.loc[cohort_mask, ["sample_id", "LN_status"]].copy()
    if cohort_rows.empty:
        raise ValueError(f"No samples found with cohort_assignment == {cohort_name!r}.")
    if cohort_rows["sample_id"].duplicated().any():
        duplicates = cohort_rows.loc[cohort_rows["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"Duplicate sample IDs in split file: {duplicates[:10]}")
    cohort_rows = cohort_rows.rename(columns={"LN_status": "target_original"})

    merged = cohort_rows.merge(
        expression[["sample_id", *genes]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    missing = merged[["sample_id", "target_original", *genes]].isna().any(axis=1)
    if len(merged) != len(cohort_rows) or missing.any():
        bad = merged.loc[missing, "sample_id"].tolist()
        raise ValueError(f"Merged cohort contains missing target/expression values: {bad[:10]}")

    X = merged[genes].astype(float)
    y = ln_status_to_binary(merged["target_original"])
    y.name = "outcome_binary_positive"
    if y.nunique() != 2:
        raise ValueError("The cohort must contain both negative and positive classes.")
    merged["outcome_binary_positive"] = y
    return CohortData(table=merged, X=X, y=y)


def require_12_model_dependencies() -> None:
    missing = []
    if XGBClassifier is None:
        missing.append("xgboost")
    if LGBMClassifier is None:
        missing.append("lightgbm")
    if CatBoostClassifier is None:
        missing.append("catboost")
    if missing:
        raise ImportError(
            "The protocol requires comparison of all 12 models, but these packages are missing: "
            + ", ".join(missing)
            + ". Install them before running the full pipeline."
        )


def model_specs() -> dict[str, tuple[object, dict[str, list[object]]]]:
    """Return the 12 candidate models and their Development tuning grids."""
    require_12_model_dependencies()
    return {
        "LR": (
            LogisticRegression(penalty="l2", solver="liblinear", random_state=SEED, max_iter=3000),
            {"clf__C": [0.03, 0.1, 0.3, 1.0]},
        ),
        "SVM": (
            SVC(kernel="rbf", probability=True, random_state=SEED),
            {"clf__C": [0.1, 1.0], "clf__gamma": ["scale", "auto"]},
        ),
        "RF": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {"clf__n_estimators": [200], "clf__max_depth": [None, 3], "clf__min_samples_leaf": [1, 2]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1, tree_method="hist", verbosity=0),
            {"clf__n_estimators": [50], "clf__max_depth": [1, 2], "clf__learning_rate": [0.03, 0.1]},
        ),
        "LightGBM": (
            LGBMClassifier(random_state=SEED, n_jobs=1, verbose=-1, feature_pre_filter=False),
            {
                "clf__n_estimators": [50],
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2],
                "clf__num_leaves": [3],
                "clf__min_child_samples": [3, 5],
            },
        ),
        "CatBoost": (
            CatBoostClassifier(random_state=SEED, verbose=0, allow_writing_files=False, thread_count=1),
            {"clf__iterations": [50], "clf__depth": [2, 3], "clf__learning_rate": [0.03, 0.1]},
        ),
        "GBDT": (
            GradientBoostingClassifier(random_state=SEED),
            {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.1], "clf__max_depth": [1, 2]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=SEED),
            {"clf__n_estimators": [50, 100], "clf__learning_rate": [0.03, 0.1]},
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(random_state=SEED, n_jobs=1),
            {"clf__n_estimators": [200], "clf__max_depth": [None, 3], "clf__min_samples_leaf": [1, 2]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5], "clf__weights": ["uniform", "distance"]},
        ),
        "MLP": (
            MLPClassifier(random_state=SEED, max_iter=2000),
            {"clf__hidden_layer_sizes": [(4,), (8,)], "clf__alpha": [1e-3, 1e-2], "clf__activation": ["relu"]},
        ),
        "GaussianNB": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
        ),
    }


def final_grid_for_selected_model(model_name: str, comparison_grid: dict[str, list[object]]) -> dict[str, list[object]]:
    
    if model_name == "GaussianNB":
        return GAUSSIANNB_FINAL_GRID
    return comparison_grid


def build_pipeline(estimator: object) -> Pipeline:
   
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_threshold_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    ppv = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "precision": ppv,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "youden_index": float(sensitivity + specificity - 1),
    }


def fit_grid_search(
    estimator: object,
    param_grid: dict[str, list[object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    inner_cv: StratifiedKFold,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=build_pipeline(estimator),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search


def select_best_by_average_rank(summary_df: pd.DataFrame) -> pd.Series:
    """Prespecified selection rule from Development rank analysis.

    Primary rule:
        Lowest average rank across AUC, accuracy, precision, F1, sensitivity,
        and specificity.

    Stability rule:
        If several models are within <0.5 average-rank units of the top model,
        choose the one with lower AUC SD.
    """
    sorted_by_rank = summary_df.sort_values(["Average rank", "Model"]).reset_index(drop=True)
    min_rank = float(sorted_by_rank.loc[0, "Average rank"])
    close = sorted_by_rank[sorted_by_rank["Average rank"] - min_rank < 0.5].copy()
    if len(close) > 1:
        close = close.sort_values(["SD AUC", "Average rank", "Model"]).reset_index(drop=True)
        return close.iloc[0]
    return sorted_by_rank.iloc[0]


def compare_12_models(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: Path,
    specs: dict[str, tuple[object, dict[str, list[object]]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS_FOR_MODEL_COMPARISON,
        shuffle=True,
        random_state=SEED,
    )

    raw_rows: list[dict[str, object]] = []
    best_param_rows: list[dict[str, object]] = []

    print(f"Stage 1: comparing {len(specs)} candidate models in Development cohort ...", flush=True)
    for model_name, (estimator, grid) in specs.items():
        print(f"  - {model_name}", flush=True)
        for split_idx, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
            repeat = (split_idx - 1) // OUTER_SPLITS + 1
            fold = (split_idx - 1) % OUTER_SPLITS + 1
            search = fit_grid_search(
                estimator=clone(estimator),
                param_grid=grid,
                X_train=X.iloc[train_idx],
                y_train=y.iloc[train_idx],
                inner_cv=inner_cv,
            )
            probability = search.best_estimator_.predict_proba(X.iloc[valid_idx])[:, 1]
            y_valid = y.iloc[valid_idx].to_numpy()
            fold_metrics = binary_threshold_metrics(y_valid, probability, threshold=0.5)

            raw_rows.append(
                {
                    "Model": model_name,
                    "Repeat": repeat,
                    "Fold": fold,
                    "AUC": float(roc_auc_score(y_valid, probability)),
                    "Brier score": float(brier_score_loss(y_valid, probability)),
                    "Accuracy": fold_metrics["accuracy"],
                    "Precision": fold_metrics["precision"],
                    "Sensitivity": fold_metrics["sensitivity"],
                    "Specificity": fold_metrics["specificity"],
                    "F1": fold_metrics["f1"],
                    "threshold_for_fold_metrics": 0.5,
                }
            )
            best_param_rows.append(
                {
                    "Model": model_name,
                    "Repeat": repeat,
                    "Fold": fold,
                    "Best Params": json.dumps(search.best_params_, ensure_ascii=False, sort_keys=True),
                    "Best inner CV AUC": float(search.best_score_),
                }
            )

    raw_df = pd.DataFrame(raw_rows)
    params_df = pd.DataFrame(best_param_rows)

    rows = []
    summary_metrics = ["AUC", "Brier score", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1"]
    for model_name, group in raw_df.groupby("Model", sort=False):
        row: dict[str, object] = {"Model": model_name}
        for metric in summary_metrics:
            row[f"Mean {metric}"] = float(group[metric].mean())
            row[f"SD {metric}"] = float(group[metric].std(ddof=1))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    rank_metrics = ["AUC", "Accuracy", "Precision", "F1", "Sensitivity", "Specificity"]
    for metric in rank_metrics:
        summary_df[f"Rank {metric}"] = summary_df[f"Mean {metric}"].rank(
            ascending=False,
            method="average",
        )
    summary_df["Average rank"] = summary_df[[f"Rank {metric}" for metric in rank_metrics]].mean(axis=1)
    summary_df = summary_df.sort_values(["Average rank", "SD AUC", "Model"]).reset_index(drop=True)

    best_row = select_best_by_average_rank(summary_df)
    best_model = str(best_row["Model"])

    raw_df.to_csv(out_dir / "development_12_models_cv_raw.csv", index=False)
    params_df.to_csv(out_dir / "development_12_models_cv_best_params.csv", index=False)
    summary_df.to_csv(out_dir / "development_12_models_cv_summary.csv", index=False)
    return raw_df, params_df, summary_df, best_model


def tune_selected_model(
    X: pd.DataFrame,
    y: pd.Series,
    selected_model_name: str,
    selected_estimator: object,
    final_grid: dict[str, list[object]],
    out_dir: Path,
) -> tuple[Pipeline, dict[str, object], pd.DataFrame]:
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS_FOR_FINAL_MODEL,
        shuffle=True,
        random_state=SEED,
    )
    search = fit_grid_search(
        estimator=clone(selected_estimator),
        param_grid=final_grid,
        X_train=X,
        y_train=y,
        inner_cv=inner_cv,
    )
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(out_dir / "selected_model_final_tuning_cv_results.csv", index=False)
    cv_results.to_csv(out_dir / f"{clean_file_stem(selected_model_name)}_final_tuning_cv_results.csv", index=False)
    return search.best_estimator_, search.best_params_, cv_results


def repeated_oof_selected_model(
    X: pd.DataFrame,
    y: pd.Series,
    selected_model_name: str,
    selected_estimator: object,
    final_params: dict[str, object],
    sample_ids: pd.Series,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )
    sum_prob = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    long_rows: list[dict[str, object]] = []

    for split_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        repeat = (split_idx - 1) // OUTER_SPLITS + 1
        fold = (split_idx - 1) % OUTER_SPLITS + 1
        model = build_pipeline(clone(selected_estimator))
        model.set_params(**final_params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probability = model.predict_proba(X.iloc[valid_idx])[:, 1]

        sum_prob[valid_idx] += probability
        counts[valid_idx] += 1
        for local_idx, prob in zip(valid_idx, probability):
            long_rows.append(
                {
                    "sample_id": sample_ids.iloc[local_idx],
                    "selected_model": selected_model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "y_true": int(y.iloc[local_idx]),
                    "raw_oof_probability": float(prob),
                }
            )

    if (counts == 0).any():
        raise RuntimeError("Some samples did not receive out-of-fold probabilities.")

    long_df = pd.DataFrame(long_rows)
    avg_df = pd.DataFrame(
        {
            "sample_id": sample_ids.to_numpy(),
            "selected_model": selected_model_name,
            "y_true": y.to_numpy(dtype=int),
            "raw_oof_probability": sum_prob / counts,
            "oof_prediction_count": counts,
        }
    )
    long_df.to_csv(out_dir / "development_selected_model_oof_predictions_long.csv", index=False)
    avg_df.to_csv(out_dir / "development_selected_model_oof_predictions.csv", index=False)
    return long_df, avg_df


def logit_probability(probability: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_probability_calibrator(oof_df: pd.DataFrame, out_dir: Path) -> tuple[LogisticRegression, pd.DataFrame]:
    y = oof_df["y_true"].to_numpy(dtype=int)
    raw_prob = oof_df["raw_oof_probability"].to_numpy(dtype=float)
    z = logit_probability(raw_prob).reshape(-1, 1)

    calibrator = LogisticRegression(solver="lbfgs", max_iter=5000)
    calibrator.fit(z, y)
    calibrated = calibrator.predict_proba(z)[:, 1]

    out = oof_df.copy()
    out["calibrated_oof_probability"] = calibrated
    out.to_csv(out_dir / "development_selected_model_oof_predictions_calibrated.csv", index=False)
    return calibrator, out


def apply_probability_calibrator(calibrator: LogisticRegression, raw_probability: np.ndarray) -> np.ndarray:
    z = logit_probability(raw_probability).reshape(-1, 1)
    return calibrator.predict_proba(z)[:, 1]


def threshold_grid(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    candidate_thresholds = np.unique(np.r_[0.0, probability, 1.0])
    rows = [binary_threshold_metrics(y_true, probability, float(threshold)) for threshold in candidate_thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def choose_thresholds_from_development(
    y_true: np.ndarray,
    probability: np.ndarray,
    out_dir: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    grid = threshold_grid(y_true, probability)
    grid.to_csv(out_dir / "development_selected_model_threshold_grid.csv", index=False)

    youden_row = (
        grid.sort_values(
            ["youden_index", "sensitivity", "specificity", "threshold"],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
        .iloc[0]
    )

    eligible = grid[grid["sensitivity"] >= HIGH_NPV_MIN_SENSITIVITY].copy()
    if eligible.empty:
        high_npv_row = youden_row
    else:
        high_npv_row = (
            eligible.sort_values(
                ["npv", "specificity", "sensitivity", "threshold"],
                ascending=[False, False, False, True],
            )
            .reset_index(drop=True)
            .iloc[0]
        )

    thresholds = {
        "Youden": float(youden_row["threshold"]),
        "high_NPV": float(high_npv_row["threshold"]),
    }
    pd.DataFrame(
        [{"threshold_name": name, "threshold": value} for name, value in thresholds.items()]
    ).to_csv(out_dir / "development_selected_model_selected_thresholds.csv", index=False)
    return thresholds, grid


def predict_positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def exact_interventional_shapley_values(model: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    
    feature_names = list(X.columns)
    values = X.reset_index(drop=True).to_numpy(dtype=float)
    background = values.copy()
    n_samples, n_features = values.shape
    if n_features != 4:
        raise ValueError(f"Exact SHAP routine expects 4 features, found {n_features}.")

    def mean_model_output(array: np.ndarray) -> float:
        frame = pd.DataFrame(array, columns=feature_names)
        return float(model.predict_proba(frame)[:, 1].mean())

    expected_value = mean_model_output(background)
    shap_values = np.zeros((n_samples, n_features), dtype=float)
    all_indices = list(range(n_features))

    for i in range(n_samples):
        x = values[i]
        for j in range(n_features):
            others = [idx for idx in all_indices if idx != j]
            phi = 0.0
            for subset_size in range(n_features):
                for subset in combinations(others, subset_size):
                    subset_idx = list(subset)
                    subset_with_j = subset_idx + [j]

                    data_with_j = background.copy()
                    data_without_j = background.copy()
                    if subset_with_j:
                        data_with_j[:, subset_with_j] = x[subset_with_j]
                    if subset_idx:
                        data_without_j[:, subset_idx] = x[subset_idx]

                    weight = (
                        math.factorial(subset_size)
                        * math.factorial(n_features - subset_size - 1)
                        / math.factorial(n_features)
                    )
                    phi += weight * (mean_model_output(data_with_j) - mean_model_output(data_without_j))
            shap_values[i, j] = phi

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    raw_probability = predict_positive_probability(model, X)
    expected_plus_shap_sum = expected_value + shap_df.sum(axis=1).to_numpy()
    check_df = pd.DataFrame(
        {
            "raw_probability": raw_probability,
            "expected_value": expected_value,
            "expected_plus_shap_sum": expected_plus_shap_sum,
            "additivity_residual": raw_probability - expected_plus_shap_sum,
        }
    )
    return shap_df, expected_value, check_df


def save_shap_outputs(
    selected_model_name: str,
    shap_df: pd.DataFrame,
    shap_check_df: pd.DataFrame,
    X: pd.DataFrame,
    cohort_table: pd.DataFrame,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    expected_value: float,
    out_dir: Path,
) -> pd.DataFrame:
    prefix = clean_file_stem(selected_model_name)
    importance = (
        pd.DataFrame(
            {
                "protein": shap_df.columns,
                "mean_abs_shap": shap_df.abs().mean(axis=0).to_numpy(),
                "mean_shap": shap_df.mean(axis=0).to_numpy(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    shap_values_out = pd.concat(
        [
            cohort_table[["sample_id", "target_original", "outcome_binary_positive"]].reset_index(drop=True),
            pd.DataFrame(
                {
                    "selected_model": selected_model_name,
                    "raw_probability": raw_probability,
                    "calibrated_probability": calibrated_probability,
                    "shap_expected_value": expected_value,
                    "shap_additivity_residual": shap_check_df["additivity_residual"],
                }
            ),
            X.reset_index(drop=True).add_prefix("feature_"),
            shap_df.reset_index(drop=True).add_prefix("SHAP_"),
        ],
        axis=1,
    )
    shap_values_out.to_csv(out_dir / "selected_model_shapley_values.csv", index=False)
    shap_values_out.to_csv(out_dir / f"{prefix}_shapley_values.csv", index=False)
    importance.to_csv(out_dir / "selected_model_shapley_importance.csv", index=False)
    importance.to_csv(out_dir / f"{prefix}_shapley_importance.csv", index=False)
    shap_check_df.to_csv(out_dir / "selected_model_shapley_additivity_check.csv", index=False)

    plot_shap_bar(selected_model_name, importance, out_dir / "selected_model_shap_summary_bar.png")
    plot_shap_beeswarm(selected_model_name, shap_df, X, importance["protein"].tolist(), out_dir / "selected_model_shap_beeswarm.png")
    for protein in importance["protein"]:
        plot_shap_dependence(
            selected_model_name=selected_model_name,
            shap_df=shap_df,
            X=X,
            protein=protein,
            path=out_dir / f"selected_model_shap_dependence_{clean_file_stem(protein)}.png",
        )
    return importance


def plot_shap_bar(selected_model_name: str, importance: pd.DataFrame, path: Path) -> None:
    plot_df = importance.sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(7, 4))
    plt.barh(plot_df["protein"], plot_df["mean_abs_shap"], color="#4C78A8")
    plt.xlabel("Mean absolute Shapley value")
    plt.ylabel("Protein")
    plt.title(f"{selected_model_name} SHAP global importance")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_shap_beeswarm(
    selected_model_name: str,
    shap_df: pd.DataFrame,
    X: pd.DataFrame,
    ordered_features: list[str],
    path: Path,
) -> None:
    rng = np.random.default_rng(SEED)
    plt.figure(figsize=(7, 4.5))
    for row_idx, protein in enumerate(reversed(ordered_features)):
        shap_values = shap_df[protein].to_numpy()
        feature_values = X[protein].to_numpy()
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(shap_values))
        plt.scatter(
            shap_values,
            np.full(len(shap_values), row_idx) + jitter,
            c=feature_values,
            cmap="coolwarm",
            s=28,
            alpha=0.85,
            edgecolors="none",
        )
    plt.axvline(0, color="black", linewidth=0.8)
    plt.yticks(range(len(ordered_features)), list(reversed(ordered_features)))
    plt.xlabel("Shapley value")
    plt.ylabel("Protein")
    plt.title(f"{selected_model_name} SHAP summary")
    cbar = plt.colorbar()
    cbar.set_label("Feature value")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_shap_dependence(
    selected_model_name: str,
    shap_df: pd.DataFrame,
    X: pd.DataFrame,
    protein: str,
    path: Path,
) -> None:
    plt.figure(figsize=(5.5, 4))
    plt.scatter(X[protein], shap_df[protein], color="#F58518", alpha=0.85, s=32)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel(f"{protein} expression")
    plt.ylabel(f"SHAP value for {protein}")
    plt.title(f"{selected_model_name} SHAP dependence: {protein}")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def calibration_bins(y_true: np.ndarray, probability: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probability >= left) & (probability <= right)
        else:
            mask = (probability >= left) & (probability < right)
        if not mask.any():
            continue
        rows.append(
            {
                "bin": i + 1,
                "left": float(left),
                "right": float(right),
                "n": int(mask.sum()),
                "mean_predicted_probability": float(probability[mask].mean()),
                "observed_event_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def dca_table(y_true: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    n = len(y_true)
    prevalence = float(y_true.mean())
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = probability >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        net_benefit_model = tp / n - fp / n * threshold / (1.0 - threshold)
        net_benefit_all = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
        rows.append(
            {
                "threshold_probability": float(threshold),
                "net_benefit_model": float(net_benefit_model),
                "net_benefit_treat_all": float(net_benefit_all),
                "net_benefit_treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def save_holdout_plots(
    selected_model_name: str,
    y_true: np.ndarray,
    probability: np.ndarray,
    out_dir: Path,
) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    auc_value = roc_auc_score(y_true, probability)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        out_dir / "holdout_selected_model_roc_curve.csv",
        index=False,
    )

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="#4C78A8", linewidth=2, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"Holdout Test ROC: {selected_model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "holdout_selected_model_roc_curve.png", dpi=300)
    plt.close()

    cal_df = calibration_bins(y_true, probability, n_bins=5)
    cal_df.to_csv(out_dir / "holdout_selected_model_calibration_bins.csv", index=False)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    if not cal_df.empty:
        plt.plot(
            cal_df["mean_predicted_probability"],
            cal_df["observed_event_rate"],
            marker="o",
            color="#F58518",
            linewidth=2,
        )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"Holdout Test calibration: {selected_model_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "holdout_selected_model_calibration_curve.png", dpi=300)
    plt.close()

    dca_df = dca_table(y_true, probability)
    dca_df.to_csv(out_dir / "holdout_selected_model_dca_table.csv", index=False)
    plt.figure(figsize=(6, 4.5))
    plt.plot(dca_df["threshold_probability"], dca_df["net_benefit_model"], label=selected_model_name, linewidth=2)
    plt.plot(dca_df["threshold_probability"], dca_df["net_benefit_treat_all"], label="Treat all", linestyle="--")
    plt.plot(dca_df["threshold_probability"], dca_df["net_benefit_treat_none"], label="Treat none", linestyle=":")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"Holdout Test decision curve: {selected_model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "holdout_selected_model_dca_curve.png", dpi=300)
    plt.close()


def evaluate_holdout(
    selected_model_name: str,
    test: CohortData,
    final_model: Pipeline,
    calibrator: LogisticRegression,
    thresholds: dict[str, float],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_probability = predict_positive_probability(final_model, test.X)
    calibrated_probability = apply_probability_calibrator(calibrator, raw_probability)
    y_true = test.y.to_numpy(dtype=int)

    predictions = test.table[["sample_id", "target_original", "outcome_binary_positive"]].copy()
    predictions["selected_model"] = selected_model_name
    predictions["raw_probability"] = raw_probability
    predictions["calibrated_probability"] = calibrated_probability
    for name, threshold in thresholds.items():
        predictions[f"pred_{name}"] = (calibrated_probability >= threshold).astype(int)
    predictions.to_csv(out_dir / "holdout_selected_model_predictions.csv", index=False)

    threshold_rows = []
    for name, threshold in thresholds.items():
        row = binary_threshold_metrics(y_true, calibrated_probability, threshold)
        row["threshold_name"] = name
        row["AUC"] = float(roc_auc_score(y_true, calibrated_probability))
        row["Brier score"] = float(brier_score_loss(y_true, calibrated_probability))
        threshold_rows.append(row)

    threshold_metrics_df = pd.DataFrame(threshold_rows)
    first_cols = ["threshold_name", "threshold", "AUC", "Brier score"]
    other_cols = [col for col in threshold_metrics_df.columns if col not in first_cols]
    threshold_metrics_df = threshold_metrics_df[first_cols + other_cols]
    threshold_metrics_df.to_csv(out_dir / "holdout_selected_model_threshold_metrics.csv", index=False)

    save_holdout_plots(selected_model_name, y_true, calibrated_probability, out_dir)

    overall = {
        "selected_model": selected_model_name,
        "test_n": int(len(y_true)),
        "test_positive_n": int(y_true.sum()),
        "test_negative_n": int(len(y_true) - y_true.sum()),
        "AUC": float(roc_auc_score(y_true, calibrated_probability)),
        "Brier score": float(brier_score_loss(y_true, calibrated_probability)),
        "threshold_metrics": threshold_metrics_df.to_dict(orient="records"),
    }
    return predictions, threshold_metrics_df, overall


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, np.ndarray)) else False:
        return None
    return obj


def save_run_metadata(
    out_dir: Path,
    genes: list[str],
    dev: CohortData,
    test: CohortData,
    best_model_name: str,
    model_summary: pd.DataFrame,
    final_model_params: dict[str, object],
    calibrator: LogisticRegression,
    thresholds: dict[str, float],
    development_oof_calibrated: pd.DataFrame,
    holdout_overall: dict[str, object],
    shap_importance: pd.DataFrame,
) -> dict[str, object]:
    best_summary = model_summary.loc[model_summary["Model"].eq(best_model_name)].iloc[0].to_dict()
    metadata = {
        "seed": SEED,
        "protein_panel": genes,
        "development": {
            "n": int(len(dev.y)),
            "positive_n": int(dev.y.sum()),
            "negative_n": int(len(dev.y) - dev.y.sum()),
        },
        "holdout_test": {
            "n": int(len(test.y)),
            "positive_n": int(test.y.sum()),
            "negative_n": int(len(test.y) - test.y.sum()),
        },
        "stage_1_model_selection": {
            "purpose": "Exploratory comparison of 12 candidate models in Development only.",
            "candidate_model_n": 12,
            "outer_cv": f"{OUTER_SPLITS}-fold x {OUTER_REPEATS} repeats",
            "inner_cv": f"{INNER_SPLITS_FOR_MODEL_COMPARISON}-fold",
            "fold_metric_threshold": 0.5,
            "rank_metrics": ["AUC", "Accuracy", "Precision", "F1", "Sensitivity", "Specificity"],
            "tie_rule": "If models are within <0.5 average-rank units, choose lower AUC SD.",
            "best_model_found_by_rank_analysis": best_model_name,
            "best_model_summary": best_summary,
        },
        "stage_2_selected_model_analysis": {
            "transition_note": (
                "After Development rank analysis identified the best model, all downstream "
                "tuning, calibration, SHAP interpretation and Holdout Test were performed "
                "for that selected model."
            ),
            "selected_model": best_model_name,
            "final_params": final_model_params,
            "refit_data": "all Development samples",
        },
        "probability_calibration": {
            "method": "Logistic calibration on averaged Development repeated-CV OOF raw probabilities",
            "input": "logit(raw selected-model probability)",
            "intercept": float(calibrator.intercept_[0]),
            "coefficient": float(calibrator.coef_[0, 0]),
            "development_oof_auc_after_calibration": float(
                roc_auc_score(
                    development_oof_calibrated["y_true"],
                    development_oof_calibrated["calibrated_oof_probability"],
                )
            ),
            "development_oof_brier_after_calibration": float(
                brier_score_loss(
                    development_oof_calibrated["y_true"],
                    development_oof_calibrated["calibrated_oof_probability"],
                )
            ),
        },
        "thresholds": {
            "source": "Development calibrated OOF probabilities only",
            "values": thresholds,
            "high_NPV_rule": (
                "Maximize NPV among Development thresholds with sensitivity >= "
                f"{HIGH_NPV_MIN_SENSITIVITY:.2f}; ties prefer higher specificity, "
                "higher sensitivity, then lower threshold."
            ),
        },
        "shap": {
            "model_output": "Frozen selected model raw positive-class probability before probability calibration",
            "method": "Exact interventional Shapley values using Development samples as empirical background",
            "global_importance": shap_importance.to_dict(orient="records"),
        },
        "holdout_test_once": holdout_overall,
    }
    metadata = json_safe(metadata)
    (out_dir / "integrated_pipeline_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def save_frozen_model(
    out_dir: Path,
    selected_model_name: str,
    final_model: Pipeline,
    calibrator: LogisticRegression,
    genes: list[str],
    thresholds: dict[str, float],
    final_model_params: dict[str, object],
) -> None:
    joblib.dump(
        {
            "selected_model": selected_model_name,
            "model": final_model,
            "calibrator": calibrator,
            "genes": genes,
            "thresholds": thresholds,
            "final_model_params": final_model_params,
            "seed": SEED,
            "note": (
                "Frozen object: StandardScaler + selected classifier fitted on all Development, "
                "logistic calibrator fitted on Development repeated-CV OOF probabilities, "
                "thresholds selected on Development OOF probabilities."
            ),
        },
        out_dir / "frozen_selected_model.joblib",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full protein-screening pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing input files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <data-dir>/protein_screening_results",
    )
    parser.add_argument("--key-genes-file", default=DEFAULT_KEY_GENES_FILE)
    parser.add_argument("--expression-file", default=DEFAULT_EXPRESSION_FILE)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = ensure_dir((args.out_dir or (data_dir / "protein_screening_results")).resolve())

    genes = read_key_genes(data_dir / args.key_genes_file)
    print(f"Protein panel: {genes}", flush=True)

    development = load_cohort(
        data_dir=data_dir,
        genes=genes,
        cohort_name="Development",
        expression_file=args.expression_file,
        split_file=args.split_file,
    )
    holdout = load_cohort(
        data_dir=data_dir,
        genes=genes,
        cohort_name="Holdout_Test",
        expression_file=args.expression_file,
        split_file=args.split_file,
    )
    print(
        "Development: "
        f"n={len(development.y)}, positive={int(development.y.sum())}, "
        f"negative={int(len(development.y) - development.y.sum())}",
        flush=True,
    )
    print(
        "Holdout Test: "
        f"n={len(holdout.y)}, positive={int(holdout.y.sum())}, "
        f"negative={int(len(holdout.y) - holdout.y.sum())}",
        flush=True,
    )

    specs = model_specs()

   
    _, _, model_summary, best_model_name = compare_12_models(development.X, development.y, out_dir, specs)
    print(f"Stage 1 result: best model found by average-rank analysis = {best_model_name}", flush=True)

   
    selected_estimator, comparison_grid = specs[best_model_name]
    final_grid = final_grid_for_selected_model(best_model_name, comparison_grid)
    final_model, final_model_params, _ = tune_selected_model(
        X=development.X,
        y=development.y,
        selected_model_name=best_model_name,
        selected_estimator=selected_estimator,
        final_grid=final_grid,
        out_dir=out_dir,
    )
    print(f"Stage 2 selected model: {best_model_name}", flush=True)
    print(f"Final selected-model params: {final_model_params}", flush=True)

    _, oof_avg = repeated_oof_selected_model(
        X=development.X,
        y=development.y,
        selected_model_name=best_model_name,
        selected_estimator=selected_estimator,
        final_params=final_model_params,
        sample_ids=development.table["sample_id"],
        out_dir=out_dir,
    )
    calibrator, oof_calibrated = fit_probability_calibrator(oof_avg, out_dir)
    thresholds, _ = choose_thresholds_from_development(
        y_true=oof_calibrated["y_true"].to_numpy(dtype=int),
        probability=oof_calibrated["calibrated_oof_probability"].to_numpy(dtype=float),
        out_dir=out_dir,
    )
    print(f"Development-selected thresholds: {thresholds}", flush=True)

    dev_raw_probability = predict_positive_probability(final_model, development.X)
    dev_calibrated_probability = apply_probability_calibrator(calibrator, dev_raw_probability)
    shap_df, shap_expected_value, shap_check_df = exact_interventional_shapley_values(final_model, development.X)
    shap_importance = save_shap_outputs(
        selected_model_name=best_model_name,
        shap_df=shap_df,
        shap_check_df=shap_check_df,
        X=development.X,
        cohort_table=development.table,
        raw_probability=dev_raw_probability,
        calibrated_probability=dev_calibrated_probability,
        expected_value=shap_expected_value,
        out_dir=out_dir,
    )

    _, _, holdout_overall = evaluate_holdout(
        selected_model_name=best_model_name,
        test=holdout,
        final_model=final_model,
        calibrator=calibrator,
        thresholds=thresholds,
        out_dir=out_dir,
    )
    save_frozen_model(
        out_dir=out_dir,
        selected_model_name=best_model_name,
        final_model=final_model,
        calibrator=calibrator,
        genes=genes,
        thresholds=thresholds,
        final_model_params=final_model_params,
    )
    metadata = save_run_metadata(
        out_dir=out_dir,
        genes=genes,
        dev=development,
        test=holdout,
        best_model_name=best_model_name,
        model_summary=model_summary,
        final_model_params=final_model_params,
        calibrator=calibrator,
        thresholds=thresholds,
        development_oof_calibrated=oof_calibrated,
        holdout_overall=holdout_overall,
        shap_importance=shap_importance,
    )

    print("\nFinished. Main outputs are in:", out_dir, flush=True)
    print(json.dumps(metadata["holdout_test_once"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
