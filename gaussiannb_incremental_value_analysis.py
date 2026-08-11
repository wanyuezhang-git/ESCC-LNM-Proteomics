

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 2026
OUTER_SPLITS = 5
OUTER_REPEATS = 10
INNER_SPLITS = 5
N_BOOTSTRAPS = 2000

DEFAULT_DATA_DIR = Path("/Users/zhangwy/Documents")
DEFAULT_CLINICAL_FILE = "clinical_origin.csv"
DEFAULT_EXPRESSION_FILE = "preprocess data.csv"
DEFAULT_SPLIT_FILE = "final_60_40_split.csv"
DEFAULT_KEY_GENES_FILE = "key genes.xlsx"

GNB_GRID = {"clf__var_smoothing": np.logspace(-12, -3, 10).tolist()}

MODEL_ORDER = ["CT Model", "Clinical Model", "Integrated Model"]
MODEL_PAIRS = [
    ("Clinical Model", "CT Model"),
    ("Integrated Model", "CT Model"),
    ("Integrated Model", "Clinical Model"),
]


@dataclass(frozen=True)
class Dataset:
    development: pd.DataFrame
    holdout: pd.DataFrame
    key_genes: list[str]
    feature_sets: dict[str, dict[str, list[str]]]
    column_mapping: pd.DataFrame
    target_source: str


def normalize_id_series(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )


def clean_file_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_").lower()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_input_file(data_dir: Path, filename: str) -> Path:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Cannot find required input file: {path}")
    return path


def read_key_genes(path: Path) -> list[str]:
    
    
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    if path.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"{path.name} must be an Excel file containing a 'key genes' column.")

    raw = pd.read_excel(path)
    required_col = "key genes"
    if required_col not in raw.columns:
        raise ValueError(
            f"{path.name} must contain a column named '{required_col}'. "
            f"Available columns: {list(raw.columns)}"
        )

    genes = [str(v).strip() for v in raw[required_col].dropna().tolist() if str(v).strip()]
    if len(genes) != 4:
        raise ValueError(f"Expected exactly 4 key genes in {path.name}, found {len(genes)}: {genes}")
    if len(set(genes)) != 4:
        raise ValueError(f"Duplicated key genes found in {path.name}: {genes}")
    return genes


def ln_status_to_binary(values: pd.Series) -> pd.Series:
    labels = values.astype(str).str.strip()
    expected = {"LN_negative", "LN_positive"}
    unexpected = sorted(set(labels.dropna().unique()) - expected)
    if unexpected:
        raise ValueError(f"LN_status must contain only LN_negative/LN_positive. Unexpected values: {unexpected}")
    return labels.eq("LN_positive").astype(int)


def convert_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.isna().any():
            bad = out.loc[numeric.isna(), col].tolist()
            raise ValueError(f"{col} must be numeric. Non-numeric values: {bad[:10]}")
        out[col] = numeric
    return out


def load_dataset(
    data_dir: Path,
    clinical_file: str,
    expression_file: str,
    split_file: str,
    key_genes_file: str,
) -> Dataset:
    key_gene_path = require_input_file(data_dir, key_genes_file)
    clinical_path = require_input_file(data_dir, clinical_file)
    expression_path = require_input_file(data_dir, expression_file)
    split_path = require_input_file(data_dir, split_file)

    key_genes = read_key_genes(key_gene_path)

    clinical = pd.read_csv(clinical_path, encoding="utf-8-sig")
    expression = pd.read_csv(expression_path, encoding="utf-8-sig")
    split = pd.read_csv(split_path, encoding="utf-8-sig")

    clinical_feature_columns = [
        "Preoperative CT",
        "Gender",
        "Age",
        "Tumor location",
        "Tumor size",
        "Tumor grade",
        "T stage",
        "LV invasion",
        "Venous invasion",
    ]
    required_clinical_columns = ["Unnamed: 0", *clinical_feature_columns, "N stage"]
    missing_clinical_features = [col for col in required_clinical_columns if col not in clinical.columns]
    if missing_clinical_features:
        raise ValueError(f"{clinical_file} is missing required columns: {missing_clinical_features}")
    clinical_cols = {name: name for name in clinical_feature_columns}

    required_split_columns = ["patient_id", "LN_status", "cohort_assignment"]
    missing_split_columns = [col for col in required_split_columns if col not in split.columns]
    if missing_split_columns:
        raise ValueError(f"{split_file} is missing required columns: {missing_split_columns}")

    required_expression_columns = ["Sample", *key_genes]
    missing_expression_columns = [col for col in required_expression_columns if col not in expression.columns]
    if missing_expression_columns:
        raise ValueError(f"{expression_file} is missing required columns: {missing_expression_columns}")

    clinical_work = clinical.copy()
    expression_work = expression.copy()
    split_work = split.copy()

    clinical_work["sample_id"] = normalize_id_series(clinical_work["Unnamed: 0"])
    expression_work["sample_id"] = normalize_id_series(expression_work["Sample"])
    split_work["sample_id"] = normalize_id_series(split_work["patient_id"])

    clinical_keep = ["sample_id", *clinical_cols.values()]
    clinical_keep = list(dict.fromkeys(clinical_keep))
    clinical_work = clinical_work[clinical_keep].rename(columns={v: k for k, v in clinical_cols.items()})
    expression_work = expression_work[["sample_id", *key_genes]]

    split_keep = ["sample_id", "cohort_assignment", "LN_status"]

    merged = (
        split_work[split_keep]
        .merge(clinical_work, on="sample_id", how="left", validate="one_to_one")
        .merge(expression_work, on="sample_id", how="left", validate="one_to_one")
        .rename(columns={"LN_status": "target_original"})
    )
    target_source = f"{split_file}:LN_status"

    required_columns = ["sample_id", "cohort_assignment", "target_original", "Preoperative CT", *key_genes]
    missing_required = merged[required_columns].isna().any(axis=1)
    if missing_required.any():
        bad = merged.loc[missing_required, "sample_id"].tolist()
        raise ValueError(f"Merged data contains missing required values for samples: {bad[:10]}")

    merged["N_stage_binary_positive"] = ln_status_to_binary(merged["target_original"])
    categorical_columns = [
        "Preoperative CT",
        "Gender",
        "Tumor location",
        "Tumor grade",
        "T stage",
        "LV invasion",
        "Venous invasion",
    ]
    for col in categorical_columns:
        if col in merged.columns:
            merged[col] = merged[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})

    numeric_columns = ["Age", "Tumor size", *key_genes]
    merged = convert_numeric_columns(merged, numeric_columns)

    development = merged[merged["cohort_assignment"].astype(str).str.strip().eq("Development")].copy()
    holdout = merged[merged["cohort_assignment"].astype(str).str.strip().eq("Holdout_Test")].copy()
    if development.empty or holdout.empty:
        raise ValueError("Could not identify both Development and Holdout_Test samples from split file.")
    if development["N_stage_binary_positive"].nunique() != 2:
        raise ValueError("Development cohort must contain both positive and negative N stage classes.")
    if holdout["N_stage_binary_positive"].nunique() != 2:
        raise ValueError("Holdout_Test cohort must contain both positive and negative N stage classes.")

    feature_sets = {
        "CT Model": {
            "numeric": [],
            "categorical": ["Preoperative CT"],
        },
        "Clinical Model": {
            "numeric": ["Age", "Tumor size"],
            "categorical": [
                "Preoperative CT",
                "Gender",
                "Tumor location",
                "Tumor grade",
                "T stage",
                "LV invasion",
                "Venous invasion",
            ],
        },
        "Integrated Model": {
            "numeric": ["Tumor size", *key_genes],
            "categorical": [
                "Preoperative CT",
                "Tumor location",
                "Tumor grade",
                "T stage",
                "LV invasion",
                "Venous invasion",
            ],
        },
    }

    mapping_rows = [
        {"standard_name": "sample_id", "source_file": clinical_file, "source_column": "Unnamed: 0", "role": "clinical ID"},
        {"standard_name": "sample_id", "source_file": expression_file, "source_column": "Sample", "role": "expression ID"},
        {"standard_name": "sample_id", "source_file": split_file, "source_column": "patient_id", "role": "split ID"},
        {"standard_name": "cohort_assignment", "source_file": split_file, "source_column": "cohort_assignment", "role": "cohort split"},
        {
            "standard_name": "target_original",
            "source_file": split_file,
            "source_column": "LN_status",
            "role": "binary outcome source",
        },
        {"standard_name": "key_genes", "source_file": key_gene_path.name, "source_column": "key genes", "role": "4-gene panel"},
    ]
    for standard_name, source_column in clinical_cols.items():
        mapping_rows.append(
            {
                "standard_name": standard_name,
                "source_file": clinical_file,
                "source_column": source_column,
                "role": "clinical/CT predictor",
            }
        )
    for gene in key_genes:
        mapping_rows.append(
            {
                "standard_name": gene,
                "source_file": expression_file,
                "source_column": gene,
                "role": "gene-expression predictor",
            }
        )
    column_mapping = pd.DataFrame(mapping_rows)

    return Dataset(
        development=development,
        holdout=holdout,
        key_genes=key_genes,
        feature_sets=feature_sets,
        column_mapping=column_mapping,
        target_source=target_source,
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_model_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            ("clf", GaussianNB()),
        ]
    )


def class_count_based_splits(y: pd.Series, requested_splits: int) -> int:
    min_count = int(y.value_counts().min())
    if min_count < 2:
        raise ValueError("At least two samples per class are required for stratified cross-validation.")
    return min(requested_splits, min_count)


def fit_tuned_gaussiannb(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    categorical_cols: list[str],
    requested_inner_splits: int = INNER_SPLITS,
) -> GridSearchCV:
    n_splits = class_count_based_splits(y, requested_inner_splits)
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=make_model_pipeline(numeric_cols, categorical_cols),
        param_grid=GNB_GRID,
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def get_model_features(df: pd.DataFrame, feature_spec: dict[str, list[str]]) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_cols = [col for col in feature_spec["numeric"] if col in df.columns]
    categorical_cols = [col for col in feature_spec["categorical"] if col in df.columns]
    missing = sorted(set(feature_spec["numeric"] + feature_spec["categorical"]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[numeric_cols + categorical_cols].copy()
    return X, numeric_cols, categorical_cols


def repeated_oof_predictions(
    model_name: str,
    development: pd.DataFrame,
    feature_spec: dict[str, list[str]],
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, numeric_cols, categorical_cols = get_model_features(development, feature_spec)
    y = development["N_stage_binary_positive"].astype(int)
    n_splits = class_count_based_splits(y, OUTER_SPLITS)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=OUTER_REPEATS,
        random_state=SEED,
    )

    sum_probability = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    long_rows = []
    tuning_rows = []

    for split_idx, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
        repeat = (split_idx - 1) // n_splits + 1
        fold = (split_idx - 1) % n_splits + 1
        search = fit_tuned_gaussiannb(
            X=X.iloc[train_idx],
            y=y.iloc[train_idx],
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            requested_inner_splits=INNER_SPLITS,
        )
        probability = search.best_estimator_.predict_proba(X.iloc[valid_idx])[:, 1]
        sum_probability[valid_idx] += probability
        counts[valid_idx] += 1

        tuning_rows.append(
            {
                "model": model_name,
                "repeat": repeat,
                "fold": fold,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "best_inner_cv_auc": float(search.best_score_),
            }
        )
        for row_idx, prob in zip(valid_idx, probability):
            long_rows.append(
                {
                    "dataset": "Development",
                    "model": model_name,
                    "sample_id": development.iloc[row_idx]["sample_id"],
                    "repeat": repeat,
                    "fold": fold,
                    "y_true": int(y.iloc[row_idx]),
                    "raw_probability": float(prob),
                }
            )

    if (counts == 0).any():
        raise RuntimeError(f"Some Development samples were not predicted in OOF for {model_name}.")

    avg_df = pd.DataFrame(
        {
            "dataset": "Development",
            "model": model_name,
            "sample_id": development["sample_id"].to_numpy(),
            "y_true": y.to_numpy(dtype=int),
            "raw_probability": sum_probability / counts,
            "oof_prediction_count": counts,
        }
    )
    long_df = pd.DataFrame(long_rows)
    tuning_df = pd.DataFrame(tuning_rows)
    long_df.to_csv(out_dir / f"development_oof_predictions_long_{clean_file_stem(model_name)}.csv", index=False)
    tuning_df.to_csv(out_dir / f"development_oof_tuning_{clean_file_stem(model_name)}.csv", index=False)
    return avg_df, tuning_df


def fit_final_model_and_predict_holdout(
    model_name: str,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_spec: dict[str, list[str]],
    out_dir: Path,
) -> tuple[GridSearchCV, pd.DataFrame, pd.DataFrame]:
    X_dev, numeric_cols, categorical_cols = get_model_features(development, feature_spec)
    y_dev = development["N_stage_binary_positive"].astype(int)
    X_holdout = holdout[numeric_cols + categorical_cols].copy()
    y_holdout = holdout["N_stage_binary_positive"].astype(int)

    search = fit_tuned_gaussiannb(
        X=X_dev,
        y=y_dev,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        requested_inner_splits=INNER_SPLITS,
    )
    final_model = search.best_estimator_
    holdout_probability = final_model.predict_proba(X_holdout)[:, 1]
    development_apparent_probability = final_model.predict_proba(X_dev)[:, 1]

    final_tuning = pd.DataFrame(search.cv_results_)
    final_tuning["model"] = model_name
    final_tuning.to_csv(out_dir / f"final_tuning_{clean_file_stem(model_name)}.csv", index=False)

    holdout_pred = pd.DataFrame(
        {
            "dataset": "Holdout_Test",
            "model": model_name,
            "sample_id": holdout["sample_id"].to_numpy(),
            "y_true": y_holdout.to_numpy(dtype=int),
            "raw_probability": holdout_probability,
        }
    )
    development_apparent_pred = pd.DataFrame(
        {
            "dataset": "Development_apparent",
            "model": model_name,
            "sample_id": development["sample_id"].to_numpy(),
            "y_true": y_dev.to_numpy(dtype=int),
            "raw_probability": development_apparent_probability,
        }
    )
    return search, holdout_pred, development_apparent_pred


def logit_probability(probability: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_calibrator_from_development_oof(oof_predictions: pd.DataFrame) -> LogisticRegression:
    y = oof_predictions["y_true"].to_numpy(dtype=int)
    raw_probability = oof_predictions["raw_probability"].to_numpy(dtype=float)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=5000)
    calibrator.fit(logit_probability(raw_probability).reshape(-1, 1), y)
    return calibrator


def apply_calibrator(calibrator: LogisticRegression, probability: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(logit_probability(probability).reshape(-1, 1))[:, 1]


def add_calibrated_probabilities(
    development_oof: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    development_apparent: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LogisticRegression]:
    calibrator = fit_calibrator_from_development_oof(development_oof)

    dev = development_oof.copy()
    hold = holdout_predictions.copy()
    apparent = development_apparent.copy()
    dev["calibrated_probability"] = apply_calibrator(calibrator, dev["raw_probability"].to_numpy(dtype=float))
    hold["calibrated_probability"] = apply_calibrator(calibrator, hold["raw_probability"].to_numpy(dtype=float))
    apparent["calibrated_probability"] = apply_calibrator(calibrator, apparent["raw_probability"].to_numpy(dtype=float))
    return dev, hold, apparent, calibrator


# ---------------------------------------------------------------------------
# DeLong AUC variance and test
# ---------------------------------------------------------------------------


def compute_midrank(x: np.ndarray) -> np.ndarray:
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    n = len(x)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[sorted_idx] = midranks
    return out


def fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_auc_variance(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(y_true)) != 2:
        return float("nan"), float("nan")
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    predictions_sorted = probability[np.newaxis, order]
    aucs, cov = fast_delong(predictions_sorted, label_1_count)
    return float(aucs[0]), float(cov[0, 0])


def delong_auc_ci(y_true: np.ndarray, probability: np.ndarray, alpha: float = 0.95) -> tuple[float, float, float, float]:
    auc, variance = delong_auc_variance(y_true, probability)
    if not np.isfinite(variance) or variance <= 0:
        return auc, float("nan"), float("nan"), float("nan")
    z = 1.959963984540054
    se = math.sqrt(variance)
    lower = max(0.0, auc - z * se)
    upper = min(1.0, auc + z * se)
    return auc, se, lower, upper


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob_a = np.asarray(prob_a, dtype=float)
    prob_b = np.asarray(prob_b, dtype=float)
    if len(np.unique(y_true)) != 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "delta_auc": float("nan"), "z": float("nan"), "p_value": float("nan")}

    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    predictions_sorted = np.vstack([prob_a, prob_b])[:, order]
    aucs, cov = fast_delong(predictions_sorted, label_1_count)
    delta = float(aucs[0] - aucs[1])
    variance = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if variance <= 0 or not np.isfinite(variance):
        z = float("nan")
        p_value = float("nan")
    else:
        z = abs(delta) / math.sqrt(variance)
        p_value = math.erfc(z / math.sqrt(2.0))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta_auc": delta,
        "z": float(z),
        "p_value": float(p_value),
    }


# ---------------------------------------------------------------------------
# Metrics, calibration, DCA, NRI and IDI
# ---------------------------------------------------------------------------


def calibration_slope_intercept(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(y_true)) != 2:
        return float("nan"), float("nan")
    try:
        model = LogisticRegression(solver="lbfgs", max_iter=5000)
        model.fit(logit_probability(probability).reshape(-1, 1), y_true)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def model_metrics(dataset_name: str, predictions: pd.DataFrame, probability_col: str) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        sub = predictions[predictions["model"].eq(model_name)].copy()
        y = sub["y_true"].to_numpy(dtype=int)
        p = sub[probability_col].to_numpy(dtype=float)
        auc, auc_se, auc_low, auc_high = delong_auc_ci(y, p)
        cal_intercept, cal_slope = calibration_slope_intercept(y, p)
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "n": int(len(y)),
                "positive_n": int(y.sum()),
                "negative_n": int(len(y) - y.sum()),
                "probability_col": probability_col,
                "AUC": auc,
                "AUC_SE_DeLong": auc_se,
                "AUC_95CI_low_DeLong": auc_low,
                "AUC_95CI_high_DeLong": auc_high,
                "Brier score": float(brier_score_loss(y, p)),
                "calibration_intercept": cal_intercept,
                "calibration_slope": cal_slope,
            }
        )
    return pd.DataFrame(rows)


def delong_comparisons(dataset_name: str, predictions: pd.DataFrame, probability_col: str) -> pd.DataFrame:
    rows = []
    wide = predictions.pivot(index="sample_id", columns="model", values=probability_col)
    y = predictions.drop_duplicates("sample_id").set_index("sample_id").loc[wide.index, "y_true"].to_numpy(dtype=int)
    for model_a, model_b in MODEL_PAIRS:
        result = delong_test(y, wide[model_a].to_numpy(dtype=float), wide[model_b].to_numpy(dtype=float))
        rows.append(
            {
                "dataset": dataset_name,
                "model_a": model_a,
                "model_b": model_b,
                "probability_col": probability_col,
                **result,
            }
        )
    return pd.DataFrame(rows)


def calibration_bins(dataset_name: str, predictions: pd.DataFrame, probability_col: str, n_bins: int = 5) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        sub = predictions[predictions["model"].eq(model_name)].copy()
        p = sub[probability_col].to_numpy(dtype=float)
        y = sub["y_true"].to_numpy(dtype=int)

        # Quantile bins keep small datasets reasonably balanced; duplicate edges
        # are dropped automatically when probabilities are tied.
        try:
            bin_id = pd.qcut(p, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            bin_id = pd.cut(p, bins=n_bins, labels=False, include_lowest=True)
        tmp = pd.DataFrame({"bin_id": bin_id, "p": p, "y": y}).dropna()
        for b, group in tmp.groupby("bin_id", sort=True):
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "probability_col": probability_col,
                    "bin": int(b) + 1,
                    "n": int(len(group)),
                    "p_min": float(group["p"].min()),
                    "p_max": float(group["p"].max()),
                    "mean_predicted_probability": float(group["p"].mean()),
                    "observed_event_rate": float(group["y"].mean()),
                }
            )
    return pd.DataFrame(rows)


def dca_table(dataset_name: str, predictions: pd.DataFrame, probability_col: str) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        sub = predictions[predictions["model"].eq(model_name)].copy()
        y = sub["y_true"].to_numpy(dtype=int)
        p = sub[probability_col].to_numpy(dtype=float)
        n = len(y)
        prevalence = float(y.mean())
        for threshold in np.linspace(0.01, 0.99, 99):
            pred = p >= threshold
            tp = int(((pred == 1) & (y == 1)).sum())
            fp = int(((pred == 1) & (y == 0)).sum())
            net_benefit_model = tp / n - fp / n * threshold / (1.0 - threshold)
            net_benefit_all = prevalence - (1.0 - prevalence) * threshold / (1.0 - threshold)
            rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "probability_col": probability_col,
                    "threshold_probability": float(threshold),
                    "net_benefit_model": float(net_benefit_model),
                    "net_benefit_treat_all": float(net_benefit_all),
                    "net_benefit_treat_none": 0.0,
                }
            )
    return pd.DataFrame(rows)


def continuous_nri_idi(y_true: np.ndarray, p_new: np.ndarray, p_old: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    p_new = np.asarray(p_new, dtype=float)
    p_old = np.asarray(p_old, dtype=float)
    events = y_true == 1
    nonevents = y_true == 0
    if events.sum() == 0 or nonevents.sum() == 0:
        return {"NRI": float("nan"), "NRI_events": float("nan"), "NRI_nonevents": float("nan"), "IDI": float("nan")}

    diff = p_new - p_old
    nri_events = float((diff[events] > 0).mean() - (diff[events] < 0).mean())
    nri_nonevents = float((diff[nonevents] < 0).mean() - (diff[nonevents] > 0).mean())
    nri = nri_events + nri_nonevents
    discrimination_slope_new = float(p_new[events].mean() - p_new[nonevents].mean())
    discrimination_slope_old = float(p_old[events].mean() - p_old[nonevents].mean())
    idi = discrimination_slope_new - discrimination_slope_old
    return {
        "NRI": nri,
        "NRI_events": nri_events,
        "NRI_nonevents": nri_nonevents,
        "IDI": idi,
        "discrimination_slope_new": discrimination_slope_new,
        "discrimination_slope_old": discrimination_slope_old,
    }


def bootstrap_nri_idi(
    dataset_name: str,
    predictions: pd.DataFrame,
    probability_col: str,
    n_bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    wide = predictions.pivot(index="sample_id", columns="model", values=probability_col)
    y = predictions.drop_duplicates("sample_id").set_index("sample_id").loc[wide.index, "y_true"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    rows = []

    for model_new, model_old in MODEL_PAIRS:
        p_new = wide[model_new].to_numpy(dtype=float)
        p_old = wide[model_old].to_numpy(dtype=float)
        estimate = continuous_nri_idi(y, p_new, p_old)
        boot_nri = []
        boot_idi = []
        n = len(y)
        for _ in range(n_bootstraps):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) != 2:
                continue
            boot = continuous_nri_idi(y[idx], p_new[idx], p_old[idx])
            if np.isfinite(boot["NRI"]):
                boot_nri.append(boot["NRI"])
            if np.isfinite(boot["IDI"]):
                boot_idi.append(boot["IDI"])

        boot_nri_arr = np.asarray(boot_nri, dtype=float)
        boot_idi_arr = np.asarray(boot_idi, dtype=float)
        nri_low, nri_high = np.percentile(boot_nri_arr, [2.5, 97.5]) if len(boot_nri_arr) else (np.nan, np.nan)
        idi_low, idi_high = np.percentile(boot_idi_arr, [2.5, 97.5]) if len(boot_idi_arr) else (np.nan, np.nan)

        rows.append(
            {
                "dataset": dataset_name,
                "new_model": model_new,
                "old_model": model_old,
                "probability_col": probability_col,
                "NRI": estimate["NRI"],
                "NRI_95CI_low_bootstrap": float(nri_low),
                "NRI_95CI_high_bootstrap": float(nri_high),
                "NRI_events": estimate["NRI_events"],
                "NRI_nonevents": estimate["NRI_nonevents"],
                "IDI": estimate["IDI"],
                "IDI_95CI_low_bootstrap": float(idi_low),
                "IDI_95CI_high_bootstrap": float(idi_high),
                "discrimination_slope_new": estimate["discrimination_slope_new"],
                "discrimination_slope_old": estimate["discrimination_slope_old"],
                "n_bootstrap_requested": int(n_bootstraps),
                "n_bootstrap_successful": int(min(len(boot_nri_arr), len(boot_idi_arr))),
            }
        )
    return pd.DataFrame(rows)


def save_roc_plot(dataset_name: str, predictions: pd.DataFrame, probability_col: str, out_dir: Path) -> None:
    plt.figure(figsize=(5.5, 5.2))
    for model_name in MODEL_ORDER:
        sub = predictions[predictions["model"].eq(model_name)]
        y = sub["y_true"].to_numpy(dtype=int)
        p = sub[probability_col].to_numpy(dtype=float)
        fpr, tpr, thresholds = roc_curve(y, p)
        auc_value = roc_auc_score(y, p)
        pd.DataFrame(
            {
                "dataset": dataset_name,
                "model": model_name,
                "probability_col": probability_col,
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresholds,
            }
        ).to_csv(out_dir / f"{clean_file_stem(dataset_name)}_roc_{clean_file_stem(model_name)}.csv", index=False)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"{dataset_name} ROC curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{clean_file_stem(dataset_name)}_roc_curves.png", dpi=300)
    plt.close()


def save_calibration_plot(dataset_name: str, cal_bins: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(5.5, 5.2))
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    for model_name in MODEL_ORDER:
        sub = cal_bins[cal_bins["model"].eq(model_name)]
        if sub.empty:
            continue
        plt.plot(
            sub["mean_predicted_probability"],
            sub["observed_event_rate"],
            marker="o",
            linewidth=2,
            label=model_name,
        )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"{dataset_name} calibration curves")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{clean_file_stem(dataset_name)}_calibration_curves.png", dpi=300)
    plt.close()


def save_dca_plot(dataset_name: str, dca: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6.2, 4.8))
    for model_name in MODEL_ORDER:
        sub = dca[dca["model"].eq(model_name)]
        plt.plot(sub["threshold_probability"], sub["net_benefit_model"], linewidth=2, label=model_name)
    ref = dca[dca["model"].eq(MODEL_ORDER[0])]
    plt.plot(ref["threshold_probability"], ref["net_benefit_treat_all"], color="gray", linestyle="--", label="Treat all")
    plt.plot(ref["threshold_probability"], ref["net_benefit_treat_none"], color="black", linestyle=":", label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"{dataset_name} decision curve analysis")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{clean_file_stem(dataset_name)}_dca_curves.png", dpi=300)
    plt.close()


def evaluate_dataset(
    dataset_name: str,
    predictions: pd.DataFrame,
    probability_col: str,
    n_bootstraps: int,
    out_dir: Path,
) -> dict[str, pd.DataFrame]:
    metrics = model_metrics(dataset_name, predictions, probability_col)
    delong = delong_comparisons(dataset_name, predictions, probability_col)
    cal_bins = calibration_bins(dataset_name, predictions, probability_col)
    dca = dca_table(dataset_name, predictions, probability_col)
    nri_idi = bootstrap_nri_idi(dataset_name, predictions, probability_col, n_bootstraps, seed=SEED)

    prefix = clean_file_stem(dataset_name)
    metrics.to_csv(out_dir / f"{prefix}_model_metrics.csv", index=False)
    delong.to_csv(out_dir / f"{prefix}_delong_auc_comparisons.csv", index=False)
    cal_bins.to_csv(out_dir / f"{prefix}_calibration_bins.csv", index=False)
    dca.to_csv(out_dir / f"{prefix}_dca_table.csv", index=False)
    nri_idi.to_csv(out_dir / f"{prefix}_nri_idi_bootstrap.csv", index=False)

    save_roc_plot(dataset_name, predictions, probability_col, out_dir)
    save_calibration_plot(dataset_name, cal_bins, out_dir)
    save_dca_plot(dataset_name, dca, out_dir)

    return {
        "metrics": metrics,
        "delong": delong,
        "calibration_bins": cal_bins,
        "dca": dca,
        "nri_idi": nri_idi,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GaussianNB CT/Clinical/Integrated model comparison.")
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
        help="Output directory. Default: <data-dir>/stage3_gaussiannb_model_comparison_results",
    )
    parser.add_argument("--clinical-file", default=DEFAULT_CLINICAL_FILE)
    parser.add_argument("--expression-file", default=DEFAULT_EXPRESSION_FILE)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--key-genes-file", default=DEFAULT_KEY_GENES_FILE)
    parser.add_argument(
        "--probability-mode",
        choices=["raw", "calibrated"],
        default="raw",
        help="Probability used for AUC/Brier/calibration/DCA/NRI/IDI. Default: raw.",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=N_BOOTSTRAPS,
        help="Bootstrap iterations for NRI/IDI CI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = ensure_dir((args.out_dir or (data_dir / "stage3_gaussiannb_model_comparison_results")).resolve())

    dataset = load_dataset(
        data_dir=data_dir,
        clinical_file=args.clinical_file,
        expression_file=args.expression_file,
        split_file=args.split_file,
        key_genes_file=args.key_genes_file,
    )
    dataset.column_mapping.to_csv(out_dir / "input_column_mapping.csv", index=False)

    print(f"Key genes: {dataset.key_genes}", flush=True)
    print(f"Outcome source: {dataset.target_source}", flush=True)
    print(
        "Development: "
        f"n={len(dataset.development)}, positive={int(dataset.development['N_stage_binary_positive'].sum())}, "
        f"negative={int(len(dataset.development) - dataset.development['N_stage_binary_positive'].sum())}",
        flush=True,
    )
    print(
        "Holdout_Test: "
        f"n={len(dataset.holdout)}, positive={int(dataset.holdout['N_stage_binary_positive'].sum())}, "
        f"negative={int(len(dataset.holdout) - dataset.holdout['N_stage_binary_positive'].sum())}",
        flush=True,
    )

    development_oof_all = []
    development_apparent_all = []
    holdout_all = []
    final_models = {}
    final_tuning_summary = []
    calibrators = {}

    for model_name in MODEL_ORDER:
        print(f"Running {model_name} ...", flush=True)
        feature_spec = dataset.feature_sets[model_name]

        dev_oof, _ = repeated_oof_predictions(model_name, dataset.development, feature_spec, out_dir)
        final_search, holdout_pred, dev_apparent_pred = fit_final_model_and_predict_holdout(
            model_name=model_name,
            development=dataset.development,
            holdout=dataset.holdout,
            feature_spec=feature_spec,
            out_dir=out_dir,
        )

        dev_oof, holdout_pred, dev_apparent_pred, calibrator = add_calibrated_probabilities(
            development_oof=dev_oof,
            holdout_predictions=holdout_pred,
            development_apparent=dev_apparent_pred,
        )
        calibrators[model_name] = calibrator

        development_oof_all.append(dev_oof)
        development_apparent_all.append(dev_apparent_pred)
        holdout_all.append(holdout_pred)
        final_models[model_name] = final_search.best_estimator_
        final_tuning_summary.append(
            {
                "model": model_name,
                "features_numeric": feature_spec["numeric"],
                "features_categorical": feature_spec["categorical"],
                "best_params": final_search.best_params_,
                "best_inner_cv_auc": float(final_search.best_score_),
                "calibration_intercept": float(calibrator.intercept_[0]),
                "calibration_coefficient": float(calibrator.coef_[0, 0]),
            }
        )

    development_predictions = pd.concat(development_oof_all, ignore_index=True)
    development_apparent_predictions = pd.concat(development_apparent_all, ignore_index=True)
    holdout_predictions = pd.concat(holdout_all, ignore_index=True)

    development_predictions.to_csv(out_dir / "development_oof_predictions_all_models.csv", index=False)
    development_apparent_predictions.to_csv(out_dir / "development_apparent_predictions_all_models.csv", index=False)
    holdout_predictions.to_csv(out_dir / "holdout_test_predictions_all_models.csv", index=False)
    pd.DataFrame(json_safe(final_tuning_summary)).to_csv(out_dir / "final_model_tuning_summary.csv", index=False)

    probability_col = "calibrated_probability" if args.probability_mode == "calibrated" else "raw_probability"
    print(f"Evaluation probability column: {probability_col}", flush=True)

    development_results = evaluate_dataset(
        dataset_name="Development",
        predictions=development_predictions,
        probability_col=probability_col,
        n_bootstraps=args.n_bootstraps,
        out_dir=out_dir,
    )
    holdout_results = evaluate_dataset(
        dataset_name="Holdout_Test",
        predictions=holdout_predictions,
        probability_col=probability_col,
        n_bootstraps=args.n_bootstraps,
        out_dir=out_dir,
    )

    # Save final frozen models. They are all trained only on Development.
    joblib.dump(
        {
            "models": final_models,
            "calibrators": calibrators,
            "feature_sets": dataset.feature_sets,
            "key_genes": dataset.key_genes,
            "probability_mode_used_for_report": args.probability_mode,
            "seed": SEED,
            "note": (
                "Three GaussianNB models trained on all Development samples. "
                "Development evaluation uses repeated-CV OOF probabilities; "
                "Holdout_Test evaluation applies Development-trained models once."
            ),
        },
        out_dir / "frozen_gaussiannb_ct_clinical_integrated_models.joblib",
    )

    summary = {
        "seed": SEED,
        "key_genes": dataset.key_genes,
        "target_source": dataset.target_source,
        "probability_mode": args.probability_mode,
        "probability_col": probability_col,
        "development_n": int(len(dataset.development)),
        "holdout_test_n": int(len(dataset.holdout)),
        "model_definitions": dataset.feature_sets,
        "input_column_mapping": dataset.column_mapping.to_dict(orient="records"),
        "final_model_tuning_summary": final_tuning_summary,
        "development": {
            "metrics": development_results["metrics"].to_dict(orient="records"),
            "delong": development_results["delong"].to_dict(orient="records"),
            "nri_idi": development_results["nri_idi"].to_dict(orient="records"),
        },
        "holdout_test": {
            "metrics": holdout_results["metrics"].to_dict(orient="records"),
            "delong": holdout_results["delong"].to_dict(orient="records"),
            "nri_idi": holdout_results["nri_idi"].to_dict(orient="records"),
        },
    }
    summary = json_safe(summary)
    (out_dir / "stage3_gaussiannb_model_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFinished. Results saved to:", out_dir, flush=True)
    print("\nDevelopment metrics:")
    print(development_results["metrics"].to_string(index=False))
    print("\nHoldout_Test metrics:")
    print(holdout_results["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
