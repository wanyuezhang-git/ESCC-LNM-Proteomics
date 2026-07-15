from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from math import erfc, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler, StratifiedKFold

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


TARGET = "N stage"

CT_FEATURES = ["Preoperative CT"]

CLINICAL_FEATURES = [
    "Tumor location",
    "Tumor size",
    "Tumor grade",
    "T stage",
    "LV invasion",
    "Venous invasion",
    "Preoperative CT",
]

PROTEIN_FEATURES = [
    "OBSCN",
    "CEMIP",
    "PFN2",
    "TMBIM1",
    "P4HA2",
    "ITGB4",
    "SURF4",
    "BPI",
]

INTEGRATED_FEATURES = CLINICAL_FEATURES + PROTEIN_FEATURES

CATEGORICAL_FEATURES = [
    "Preoperative CT",
    "Tumor location",
    "Tumor grade",
    "T stage",
    "LV invasion",
    "Venous invasion",
]

NUMERIC_FEATURES = [
    "Tumor size",
    "OBSCN",
    "CEMIP",
    "PFN2",
    "TMBIM1",
    "P4HA2",
    "ITGB4",
    "SURF4",
    "BPI",
]

MODEL_ORDER = ["CT Model", "Clinical Model", "Integrated Model"]
MODEL_FEATURES = {
    "CT Model": CT_FEATURES,
    "Clinical Model": CLINICAL_FEATURES,
    "Integrated Model": INTEGRATED_FEATURES,
}

SHARED_CATBOOST_PARAM_GRID = {
    "iterations": [100, 200, 300, 500, 800],
    "depth": [2, 3, 4],
    "learning_rate": [0.01, 0.02, 0.03, 0.05],
    "l2_leaf_reg": [5, 10, 15, 20, 30, 50],
    "random_strength": [1.0, 2.0, 5.0, 8.0],
    "bagging_temperature": [0.0, 0.5, 1.0, 2.0],
    "border_count": [16, 32, 64],
}

OUTER_SPLITS = 10
INNER_SPLITS = 4
MAIN_DCA_THRESHOLDS = np.arange(0.10, 0.60 + 0.001, 0.01)
SUPPLEMENTARY_DCA_THRESHOLDS = np.arange(0.05, 0.75 + 0.001, 0.01)
FULL_DCA_THRESHOLDS = np.arange(0.01, 1.00, 0.01)
CALIBRATION_BINS = 5
HL_GROUPS = 10
SENSITIVITY_TARGET = 0.80
RISK_CATEGORY_CUTOFFS = (0.30, 0.70)
EARLY_STOPPING_ROUNDS = 50
LOSS_FUNCTION = "Logloss"
EVAL_METRIC = "AUC"

COLORS = {
    "CT Model": "#4DBBD5",
    "Clinical Model": "#3C5488",
    "Integrated Model": "#E64B35",
    "Treat All": "#7E6148",
    "Treat None": "#8491B0",
}


def set_random_seed(seed: int) -> None:
    """Set Python and NumPy seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return the resolved path."""
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def to_python_scalar(value):
    """Convert NumPy scalar containers into JSON-safe Python objects."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def json_dumps(value) -> str:
    """Serialize objects with stable key ordering."""
    if isinstance(value, dict):
        serializable = {str(k): to_python_scalar(v) for k, v in value.items()}
        return json.dumps(serializable, ensure_ascii=False, sort_keys=True)
    return json.dumps(value, ensure_ascii=False)


def locate_default_input(script_dir: Path) -> Optional[Path]:
    """Find a sensible default CSV if --input-csv is not supplied."""
    candidates = [
        script_dir / "clinical_gene_CT_clinical.csv",
        script_dir.parent / "clinical_gene_CT_clinical.csv",
        script_dir.parent.parent / "STEP4" / "clinical_gene_CT_clinical.csv",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def build_column_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """Map canonical study variables to actual CSV column names."""
    canonical = {
        "Tumor location": ["Tumor location", "Location"],
        "Tumor size": ["Tumor size", "Tumor_size"],
        "Tumor grade": ["Tumor grade", "Grade"],
        "T stage": ["T stage", "T_stage"],
        "LV invasion": ["LV invasion", "Lymphovascular invasion"],
        "Venous invasion": ["Venous invasion"],
        "Preoperative CT": ["Preoperative CT", "Preop CT", "CT"],
        TARGET: [TARGET, "LN_status"],
    }
    mapping: Dict[str, str] = {}
    for canonical_name, candidates in canonical.items():
        actual = next((col for col in candidates if col in df.columns), None)
        if actual is None:
            raise KeyError(f"Missing required column for '{canonical_name}'.")
        mapping[canonical_name] = actual
    for protein in PROTEIN_FEATURES:
        if protein not in df.columns:
            raise KeyError(f"Missing protein column '{protein}'.")
        mapping[protein] = protein
    return mapping


def choose_sample_id_column(df: pd.DataFrame) -> Optional[str]:
    """Pick an ID-like column if one exists."""
    candidates = [col for col in df.columns if "id" in col.lower()]
    for col in candidates:
        if df[col].is_unique:
            return col
    return candidates[0] if candidates else None


def validate_target_binary(y: pd.Series) -> np.ndarray:
    """Validate that the target contains only 0/1."""
    y_num = pd.to_numeric(y, errors="raise")
    unique_values = set(pd.unique(y_num))
    if unique_values - {0, 1}:
        raise ValueError(f"Target labels must be 0/1. Found: {sorted(unique_values)}")
    if pd.isna(y_num).any():
        raise ValueError("Target column contains missing values.")
    return y_num.astype(int).to_numpy()


def prepare_dataframe(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
) -> Tuple[pd.DataFrame, List[int]]:
    """Prepare a CatBoost-ready feature frame using robust typing rules."""
    x = df.loc[:, feature_names].copy()
    categorical_in_use = [col for col in categorical_names if col in feature_names]
    for col in categorical_in_use:
        x[col] = x[col].fillna("Missing").astype(str)
    for col in feature_names:
        if col not in categorical_in_use:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    cat_idx = [feature_names.index(col) for col in categorical_in_use]
    return x, cat_idx


def check_input_data(df: pd.DataFrame, column_map: Dict[str, str]) -> None:
    """Check that all required columns exist and that numeric fields are finite if present."""
    required_columns = sorted(
        set(
            [column_map[TARGET]]
            + [column_map[name] for name in CT_FEATURES]
            + [column_map[name] for name in CLINICAL_FEATURES if name != "Preoperative CT"]
            + [column_map[name] for name in PROTEIN_FEATURES]
        )
    )
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    numeric_candidates = [column_map[name] for name in NUMERIC_FEATURES if name in column_map]
    numeric_block = df[numeric_candidates].apply(pd.to_numeric, errors="coerce")
    if np.isinf(numeric_block.to_numpy(dtype=float)).any():
        raise ValueError("Input data contain Inf values, which are not allowed.")


def safe_auc(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """Return AUC when both classes are present; otherwise NaN."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong's test."""
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def fast_delong(pred_sorted_transposed: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fast DeLong implementation for correlated ROC AUC estimates."""
    if pred_sorted_transposed.ndim != 2:
        raise ValueError("pred_sorted_transposed must be 2-dimensional.")
    n_total = pred_sorted_transposed.shape[1]
    if m <= 0 or m >= n_total:
        raise ValueError("DeLong requires both positive and negative samples.")
    n = n_total - m
    k = pred_sorted_transposed.shape[0]
    pos = pred_sorted_transposed[:, :m]
    neg = pred_sorted_transposed[:, m:]
    tx = np.zeros((k, m), dtype=float)
    ty = np.zeros((k, n), dtype=float)
    tz = np.zeros((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(pred_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true: Sequence[int], old_prob: Sequence[float], new_prob: Sequence[float]) -> Dict[str, float]:
    """Run DeLong's test for two correlated ROC AUCs."""
    y_true = np.asarray(y_true, dtype=int)
    old_prob = np.asarray(old_prob, dtype=float)
    new_prob = np.asarray(new_prob, dtype=float)
    if len(np.unique(y_true)) != 2:
        raise ValueError("DeLong requires both classes.")
    order = np.argsort(-y_true)
    pred = np.vstack([old_prob[order], new_prob[order]])
    m = int(np.sum(y_true))
    aucs, cov = fast_delong(pred, m)
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ cov @ contrast.T)
    if not np.isfinite(var) or var <= 1e-12:
        raise ValueError("DeLong variance is invalid.")
    se = math.sqrt(var)
    diff = float(aucs[1] - aucs[0])
    z = diff / se
    p_value = erfc(abs(z) / sqrt(2.0))
    return {
        "AUC_old": float(aucs[0]),
        "AUC_new": float(aucs[1]),
        "AUC_difference": diff,
        "Standard_error": float(se),
        "Z_statistic": float(z),
        "DeLong_p_value": float(p_value),
    }


def auc_ci_bootstrap(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        scores.append(roc_auc_score(y_boot, y_prob[idx]))
    if len(scores) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def bootstrap_difference_ci(
    y_true: Sequence[int],
    old_prob: Sequence[float],
    new_prob: Sequence[float],
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    """Bootstrap 95% CI for AUC(new) - AUC(old)."""
    y_true = np.asarray(y_true, dtype=int)
    old_prob = np.asarray(old_prob, dtype=float)
    new_prob = np.asarray(new_prob, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        diffs.append(roc_auc_score(y_boot, new_prob[idx]) - roc_auc_score(y_boot, old_prob[idx]))
    if len(diffs) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def calibration_slope_intercept(y_true: Sequence[int], y_prob: Sequence[float]) -> Tuple[float, float]:
    """Estimate calibration slope and intercept."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(y_prob / (1.0 - y_prob)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, penalty="l2", solver="lbfgs", max_iter=2000)
        lr.fit(logits, y_true)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return float("nan"), float("nan")


def hosmer_lemeshow_test(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    groups: int = HL_GROUPS,
) -> Tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit test."""
    df_hl = pd.DataFrame({"y": y_true, "p": y_prob})
    try:
        df_hl["bin"] = pd.qcut(df_hl["p"], q=groups, duplicates="drop")
    except ValueError:
        return float("nan"), float("nan")
    grouped = df_hl.groupby("bin", observed=False)
    obs = grouped["y"].sum().to_numpy(dtype=float)
    exp = grouped["p"].sum().to_numpy(dtype=float)
    n = grouped.size().to_numpy(dtype=float)
    g = len(n)
    if g <= 2:
        return float("nan"), float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = exp * (1.0 - exp / n)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    hl_stat = float(np.sum((obs - exp) ** 2 / denom))
    hl_p = float(1.0 - chi2.cdf(hl_stat, df=g - 2))
    return hl_stat, hl_p


def get_quantile_edges(y_prob: Sequence[float], n_bins: int = CALIBRATION_BINS) -> np.ndarray:
    """Create quantile-based bin edges for calibration."""
    edges = np.unique(np.quantile(np.asarray(y_prob, dtype=float), np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, min(n_bins + 1, 3))
    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], 1.0)
    return edges


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return float(center - half), float(center + half)


def compute_ece_mce(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = CALIBRATION_BINS,
) -> Tuple[float, float]:
    """Compute expected and maximum calibration errors."""
    edges = get_quantile_edges(y_prob, n_bins=n_bins)
    bins = pd.cut(y_prob, bins=edges, include_lowest=True, duplicates="drop")
    df_bin = pd.DataFrame({"y": y_true, "p": y_prob, "bin": bins})
    grouped = df_bin.groupby("bin", observed=False)
    abs_errors = []
    weights = []
    for _, grp in grouped:
        if grp.empty:
            continue
        abs_errors.append(abs(float(grp["y"].mean()) - float(grp["p"].mean())))
        weights.append(len(grp) / len(df_bin))
    if not abs_errors:
        return float("nan"), float("nan")
    ece = float(np.sum(np.asarray(abs_errors) * np.asarray(weights)))
    mce = float(np.max(abs_errors))
    return ece, mce


def decision_curve(y_true: Sequence[int], y_prob: Sequence[float], thresholds: Sequence[float]) -> np.ndarray:
    """Compute decision-curve net benefit across thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    out = []
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            out.append(np.nan)
            continue
        y_pred = (y_prob >= pt).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.append((tp / n) - (fp / n) * (pt / (1 - pt)))
    return np.asarray(out, dtype=float)


def classification_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> Dict[str, float]:
    """Calculate discrimination and classification metrics at one threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "AUC": safe_auc(y_true, y_prob),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "NPV": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
        "F1 score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier score": float(np.mean((y_prob - y_true) ** 2)),
    }


def catboost_complexity_score(params: Dict[str, object]) -> float:
    """Define a simple monotonic complexity score for tie-breaking."""
    return (
        float(params.get("depth", 0)) * 10.0
        + float(params.get("iterations", 0)) / 100.0
        + float(params.get("l2_leaf_reg", 0)) / 10.0
        + float(params.get("random_strength", 0))
    )


def summarize_candidate_performance(
    y_true: Sequence[int],
    inner_oof: Sequence[float],
    fold_aucs: Sequence[float],
) -> Dict[str, float]:
    """Summarize inner-CV candidate performance with tie-breaking metrics."""
    y_true = np.asarray(y_true, dtype=int)
    inner_oof = np.asarray(inner_oof, dtype=float)
    metrics = classification_metrics(y_true, inner_oof, threshold=0.5)
    slope, intercept = calibration_slope_intercept(y_true, inner_oof)
    dca_nb = decision_curve(y_true, inner_oof, MAIN_DCA_THRESHOLDS)
    fold_aucs = np.asarray(fold_aucs, dtype=float)
    fold_aucs = fold_aucs[np.isfinite(fold_aucs)]
    return {
        "Inner_mean_AUC": float(np.mean(fold_aucs)) if len(fold_aucs) else float("nan"),
        "Inner_SD_AUC": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
        "Inner_Brier_score": metrics["Brier score"],
        "Inner_calibration_slope": slope,
        "Inner_calibration_intercept": intercept,
        "Inner_mean_DCA_net_benefit": float(np.nanmean(dca_nb)),
    }


def selection_sort_key(row: Dict[str, object]) -> Tuple[float, float, float, float, float]:
    """Build a safe sort key for candidate selection."""
    def finite_or_inf(value, transform=None):
        if value is None or pd.isna(value):
            return float("inf")
        value = float(value)
        return transform(value) if transform is not None else value

    return (
        finite_or_inf(row["Inner_Brier_score"]),
        finite_or_inf(row["Inner_calibration_slope"], lambda x: abs(x - 1.0)),
        finite_or_inf(row["Inner_calibration_intercept"], abs),
        finite_or_inf(row["Inner_SD_AUC"]),
        finite_or_inf(row["Complexity_score"]),
    )


def choose_best_candidate(candidate_rows: List[Dict[str, object]]) -> Tuple[Dict[str, object], str]:
    """Apply the common model-selection rule to one model's candidate list."""
    if not candidate_rows:
        raise ValueError("No candidate rows were provided for model selection.")
    sorted_rows = sorted(candidate_rows, key=lambda row: row["Inner_mean_AUC"], reverse=True)
    top_auc = sorted_rows[0]["Inner_mean_AUC"]
    close_rows = [row for row in sorted_rows if (top_auc - row["Inner_mean_AUC"]) < 0.01]
    close_rows = sorted(close_rows, key=selection_sort_key)
    reason = (
        "Primary selection by highest inner mean AUC; when AUC difference < 0.01, "
        "tie-breaking used lower Brier score, calibration slope closer to 1, "
        "calibration intercept closer to 0, lower inner AUC SD, and lower model complexity."
    )
    return close_rows[0], reason


def get_class_weight_options(y_train: Sequence[int]) -> List[Dict[str, object]]:
    """Generate the same class-weight candidate schemes for all three models."""
    y_train = np.asarray(y_train, dtype=int)
    counts = np.bincount(y_train, minlength=2)
    neg_count, pos_count = counts[0], counts[1]
    manual = None
    if neg_count > 0 and pos_count > 0:
        ratio = neg_count / pos_count
        manual = [1.0, float(ratio)] if ratio >= 1.0 else [float(1.0 / ratio), 1.0]
    options = [{"weighting_scheme": "none", "catboost_kwargs": {}}]
    options.append(
        {"weighting_scheme": "auto_balanced", "catboost_kwargs": {"auto_class_weights": "Balanced"}}
    )
    if manual is not None:
        options.append({"weighting_scheme": "manual_ratio", "catboost_kwargs": {"class_weights": manual}})
    return options


def build_catboost_estimator(params: Dict[str, object], random_seed: int) -> CatBoostClassifier:
    """Create a CatBoost classifier with shared objective settings."""
    estimator_params = {
        "loss_function": LOSS_FUNCTION,
        "eval_metric": EVAL_METRIC,
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    estimator_params.update(params)
    return CatBoostClassifier(**estimator_params)


def fit_final_catboost_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_idx: List[int],
    params: Dict[str, object],
    random_seed: int,
) -> CatBoostClassifier:
    """Fit the selected CatBoost model on the full outer training fold."""
    model = build_catboost_estimator(params, random_seed=random_seed)
    model.fit(x_train, y_train, cat_features=cat_idx)
    return model


def evaluate_model_candidates(
    model_name: str,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    feature_names: List[str],
    categorical_names: List[str],
    inner_splits: List[Tuple[np.ndarray, np.ndarray]],
    candidate_params: List[Dict[str, object]],
    class_weight_options: List[Dict[str, object]],
    outer_fold: int,
    seed: int,
) -> Dict[str, object]:
    """Tune one model with the shared inner splits and shared candidate list."""
    x_train, cat_idx = prepare_dataframe(train_df, feature_names, categorical_names)
    candidate_rows: List[Dict[str, object]] = []

    for param_index, sampled_params in enumerate(candidate_params, start=1):
        for weight_index, weight_option in enumerate(class_weight_options, start=1):
            params = dict(sampled_params)
            params.update(weight_option["catboost_kwargs"])
            inner_oof = np.full(len(y_train), np.nan, dtype=float)
            fold_aucs = []
            for inner_fold, (tr_idx, val_idx) in enumerate(inner_splits, start=1):
                model = build_catboost_estimator(
                    params=params,
                    random_seed=seed + outer_fold * 1000 + param_index * 10 + inner_fold,
                )
                model.fit(
                    x_train.iloc[tr_idx],
                    y_train[tr_idx],
                    cat_features=cat_idx,
                    eval_set=(x_train.iloc[val_idx], y_train[val_idx]),
                    use_best_model=True,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                )
                val_prob = model.predict_proba(x_train.iloc[val_idx])[:, 1]
                inner_oof[val_idx] = val_prob
                fold_aucs.append(safe_auc(y_train[val_idx], val_prob))

            summary = summarize_candidate_performance(y_train, inner_oof, fold_aucs)
            best_iteration = None
            try:
                best_iteration = int(model.get_best_iteration())
            except Exception:
                best_iteration = None
            candidate_rows.append(
                {
                    "Outer_fold": outer_fold,
                    "Model": model_name,
                    "Parameter_index": param_index,
                    "Weight_option_index": weight_index,
                    "Weighting_scheme": weight_option["weighting_scheme"],
                    "Hyperparameters": json_dumps(params),
                    "Inner_mean_AUC": summary["Inner_mean_AUC"],
                    "Inner_SD_AUC": summary["Inner_SD_AUC"],
                    "Inner_Brier_score": summary["Inner_Brier_score"],
                    "Inner_calibration_slope": summary["Inner_calibration_slope"],
                    "Inner_calibration_intercept": summary["Inner_calibration_intercept"],
                    "Inner_mean_DCA_net_benefit": summary["Inner_mean_DCA_net_benefit"],
                    "Complexity_score": catboost_complexity_score(params),
                    "Best_iteration_last_inner_fit": best_iteration,
                    "train_oof": inner_oof.copy(),
                    "best_params": params.copy(),
                }
            )

    selected, selection_reason = choose_best_candidate(candidate_rows)
    final_model = fit_final_catboost_model(
        x_train=x_train,
        y_train=y_train,
        cat_idx=cat_idx,
        params=selected["best_params"],
        random_seed=seed + outer_fold * 5000,
    )
    return {
        "model_name": model_name,
        "features": feature_names,
        "categorical_features": categorical_names,
        "candidate_rows": candidate_rows,
        "selected_params": selected["best_params"],
        "selected_train_oof": selected["train_oof"],
        "inner_mean_auc": selected["Inner_mean_AUC"],
        "inner_sd_auc": selected["Inner_SD_AUC"],
        "inner_brier": selected["Inner_Brier_score"],
        "inner_calibration_slope": selected["Inner_calibration_slope"],
        "inner_calibration_intercept": selected["Inner_calibration_intercept"],
        "inner_mean_dca_net_benefit": selected["Inner_mean_DCA_net_benefit"],
        "selection_reason": selection_reason,
        "final_model": final_model,
        "cat_idx": cat_idx,
    }


def select_threshold_from_training_oof(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    mode: str,
    sensitivity_target: float = SENSITIVITY_TARGET,
) -> float:
    """Choose threshold using only outer-training inner-OOF predictions."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1.0 - fpr
    if mode == "youden":
        score = tpr + specificity - 1.0
        return float(thresholds[int(np.argmax(score))])
    if mode == "sens080":
        eligible = np.where(tpr >= sensitivity_target)[0]
        if len(eligible) == 0:
            score = tpr + specificity - 1.0
            return float(thresholds[int(np.argmax(score))])
        best_idx = eligible[int(np.argmax(specificity[eligible]))]
        return float(thresholds[best_idx])
    raise ValueError(f"Unknown threshold mode: {mode}")


def assign_risk_category(prob: Sequence[float], cutoffs: Tuple[float, float] = RISK_CATEGORY_CUTOFFS) -> np.ndarray:
    """Convert probabilities into low/intermediate/high risk categories."""
    return np.digitize(np.asarray(prob, dtype=float), bins=np.asarray(cutoffs, dtype=float), right=False)


def continuous_nri(
    y_true: Sequence[int],
    old_prob: Sequence[float],
    new_prob: Sequence[float],
) -> Tuple[float, float, float]:
    """Continuous NRI."""
    y_true = np.asarray(y_true, dtype=int)
    diff = np.asarray(new_prob, dtype=float) - np.asarray(old_prob, dtype=float)
    event = y_true == 1
    non_event = y_true == 0
    nri_event = np.mean(diff[event] > 0) - np.mean(diff[event] < 0)
    nri_non_event = np.mean(diff[non_event] < 0) - np.mean(diff[non_event] > 0)
    return float(nri_event + nri_non_event), float(nri_event), float(nri_non_event)


def categorical_nri(
    y_true: Sequence[int],
    old_prob: Sequence[float],
    new_prob: Sequence[float],
    cutoffs: Tuple[float, float] = RISK_CATEGORY_CUTOFFS,
) -> Tuple[float, float, float]:
    """Categorical NRI."""
    y_true = np.asarray(y_true, dtype=int)
    old_bins = assign_risk_category(old_prob, cutoffs=cutoffs)
    new_bins = assign_risk_category(new_prob, cutoffs=cutoffs)
    delta = new_bins - old_bins
    event = y_true == 1
    non_event = y_true == 0
    nri_event = np.mean(delta[event] > 0) - np.mean(delta[event] < 0)
    nri_non_event = np.mean(delta[non_event] < 0) - np.mean(delta[non_event] > 0)
    return float(nri_event + nri_non_event), float(nri_event), float(nri_non_event)


def discrimination_slope(y_true: Sequence[int], prob: Sequence[float]) -> float:
    """Difference in mean predicted risk between events and non-events."""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    return float(np.mean(prob[y_true == 1]) - np.mean(prob[y_true == 0]))


def idi(y_true: Sequence[int], old_prob: Sequence[float], new_prob: Sequence[float]) -> float:
    """Integrated discrimination improvement."""
    return float(discrimination_slope(y_true, new_prob) - discrimination_slope(y_true, old_prob))


def format_interval(low: float, high: float, decimals: int = 3) -> str:
    """Format numeric confidence intervals."""
    if pd.isna(low) or pd.isna(high):
        return "NA"
    return f"{float(low):.{decimals}f}-{float(high):.{decimals}f}"


def format_auc_with_ci(auc_value: float, low: float, high: float) -> str:
    """Format AUC with 95% CI."""
    if pd.isna(low) or pd.isna(high):
        return f"{float(auc_value):.3f} (95% CI NA)"
    return f"{float(auc_value):.3f} (95% CI {float(low):.3f}-{float(high):.3f})"


def compress_thresholds_to_ranges(thresholds: Sequence[float]) -> str:
    """Compress discrete thresholds into readable ranges."""
    values = sorted(round(float(x), 2) for x in thresholds)
    if not values:
        return "None"
    ranges = []
    start = values[0]
    prev = values[0]
    for curr in values[1:]:
        if round(curr - prev, 2) <= 0.011:
            prev = curr
            continue
        ranges.append(f"{start:.2f}" if start == prev else f"{start:.2f}-{prev:.2f}")
        start = prev = curr
    ranges.append(f"{start:.2f}" if start == prev else f"{start:.2f}-{prev:.2f}")
    return ", ".join(ranges)


def validate_oof_predictions(oof_df: pd.DataFrame) -> None:
    """Ensure each sample receives exactly one valid outer-fold prediction."""
    if oof_df["outer_fold"].isna().any():
        raise ValueError("OOF predictions are missing outer-fold assignments.")
    fold_counts = oof_df["sample_index"].value_counts()
    if not (fold_counts == 1).all():
        raise ValueError("Each sample must appear exactly once in merged OOF predictions.")
    for col in ["prob_CT_Model", "prob_Clinical_Model", "prob_Integrated_Model"]:
        if oof_df[col].isna().any():
            raise ValueError(f"OOF predictions contain missing values in {col}.")
        if ((oof_df[col] < 0) | (oof_df[col] > 1)).any():
            raise ValueError(f"OOF probabilities outside [0, 1] in {col}.")


def save_figure_bundle(fig: plt.Figure, base_dir: Path, stem: str) -> None:
    """Save a figure as PNG, PDF, and TIFF."""
    fig.tight_layout()
    fig.savefig(base_dir / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base_dir / f"{stem}.pdf", dpi=600, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(
            base_dir / f"{stem}.tiff",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            pil_kwargs={"compression": "tiff_lzw"},
        )
    except TypeError:
        fig.savefig(base_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def calculate_roc_outputs(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build ROC curve data and AUC summary."""
    roc_rows = []
    auc_rows = []
    for model_name in MODEL_ORDER:
        fpr, tpr, thresholds = roc_curve(y_true, prob_dict[model_name])
        auc_value = safe_auc(y_true, prob_dict[model_name])
        auc_low, auc_high = auc_ci_bootstrap(y_true, prob_dict[model_name], n_boot=n_boot, seed=seed)
        auc_rows.append(
            {
                "Model": model_name,
                "AUC": auc_value,
                "AUC 95% CI lower": auc_low,
                "AUC 95% CI upper": auc_high,
                "AUC and 95% CI": format_auc_with_ci(auc_value, auc_low, auc_high),
                "Number of samples": int(len(y_true)),
                "Number of events": int(np.sum(y_true == 1)),
                "Number of non-events": int(np.sum(y_true == 0)),
            }
        )
        for thr, fpr_val, tpr_val in zip(thresholds, fpr, tpr):
            roc_rows.append(
                {
                    "Model": model_name,
                    "Threshold": float(thr) if np.isfinite(thr) else np.nan,
                    "FPR": float(fpr_val),
                    "TPR": float(tpr_val),
                    "Sensitivity": float(tpr_val),
                    "Specificity": float(1.0 - fpr_val),
                }
            )
    return pd.DataFrame(roc_rows), pd.DataFrame(auc_rows)


def calculate_calibration_outputs(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build calibration-curve data and overall calibration metrics."""
    curve_rows = []
    metric_rows = []
    for model_name in MODEL_ORDER:
        p_arr = np.asarray(prob_dict[model_name], dtype=float)
        edges = get_quantile_edges(p_arr, n_bins=CALIBRATION_BINS)
        bins = pd.cut(p_arr, bins=edges, include_lowest=True, duplicates="drop")
        df_bin = pd.DataFrame({"y": y_true, "p": p_arr, "bin": bins})
        grouped = df_bin.groupby("bin", observed=False)
        for bin_idx, (_, grp) in enumerate(grouped, start=1):
            if grp.empty:
                continue
            ci_low, ci_high = wilson_ci(int(grp["y"].sum()), int(len(grp)))
            curve_rows.append(
                {
                    "Model": model_name,
                    "Bin": bin_idx,
                    "n": int(len(grp)),
                    "Mean predicted probability": float(grp["p"].mean()),
                    "Observed event rate": float(grp["y"].mean()),
                    "Observed 95% CI lower": ci_low,
                    "Observed 95% CI upper": ci_high,
                }
            )
        slope, intercept = calibration_slope_intercept(y_true, p_arr)
        hl_stat, hl_p = hosmer_lemeshow_test(y_true, p_arr)
        ece, mce = compute_ece_mce(y_true, p_arr, n_bins=CALIBRATION_BINS)
        metric_rows.append(
            {
                "Model": model_name,
                "Brier score": float(np.mean((p_arr - y_true) ** 2)),
                "Calibration slope": slope,
                "Calibration intercept": intercept,
                "Hosmer-Lemeshow statistic": hl_stat,
                "Hosmer-Lemeshow p value": hl_p,
                "ECE": ece,
                "MCE": mce,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(metric_rows)


def calculate_dca_outputs(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build DCA curves and threshold-range summaries."""
    prevalence = float(np.mean(y_true == 1))

    def build_frame(thresholds: np.ndarray, range_name: str) -> pd.DataFrame:
        frame = pd.DataFrame({"Threshold": thresholds})
        frame["CT net benefit"] = decision_curve(y_true, prob_dict["CT Model"], thresholds)
        frame["Clinical net benefit"] = decision_curve(y_true, prob_dict["Clinical Model"], thresholds)
        frame["Integrated net benefit"] = decision_curve(y_true, prob_dict["Integrated Model"], thresholds)
        frame["Treat All net benefit"] = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
        frame["Treat None net benefit"] = 0.0
        frame["Integrated minus Clinical"] = frame["Integrated net benefit"] - frame["Clinical net benefit"]
        frame["Integrated minus CT"] = frame["Integrated net benefit"] - frame["CT net benefit"]
        frame["Clinical minus CT"] = frame["Clinical net benefit"] - frame["CT net benefit"]
        frame["Threshold range type"] = range_name
        return frame

    full_df = build_frame(FULL_DCA_THRESHOLDS, "full_001_099")
    supplementary_df = build_frame(SUPPLEMENTARY_DCA_THRESHOLDS, "supplementary_005_075")
    main_df = build_frame(MAIN_DCA_THRESHOLDS, "main_010_060")
    dca_curve_df = pd.concat([main_df, supplementary_df, full_df], ignore_index=True)

    summary_rows = []
    for range_name, sub in [
        ("0.10-0.60", main_df),
        ("0.05-0.75", supplementary_df),
        ("0.01-0.99", full_df),
    ]:
        better = sub["Integrated minus Clinical"] > 0
        summary_rows.append(
            {
                "Threshold range": range_name,
                "Mean CT net benefit": float(sub["CT net benefit"].mean()),
                "Mean Clinical net benefit": float(sub["Clinical net benefit"].mean()),
                "Mean Integrated net benefit": float(sub["Integrated net benefit"].mean()),
                "Integrated minus Clinical mean difference": float(sub["Integrated minus Clinical"].mean()),
                "Integrated better than Clinical proportion": float(better.mean()),
                "Integrated better than Clinical thresholds": compress_thresholds_to_ranges(
                    sub.loc[better, "Threshold"].tolist()
                ),
            }
        )
    return dca_curve_df, pd.DataFrame(summary_rows)


def calculate_delong_outputs(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Run pairwise DeLong tests and bootstrap CI for AUC differences."""
    comparisons = [
        ("Clinical Model", "Integrated Model"),
        ("CT Model", "Clinical Model"),
        ("CT Model", "Integrated Model"),
    ]
    rows = []
    for idx, (old_model, new_model) in enumerate(comparisons, start=1):
        stats = delong_test(y_true, prob_dict[old_model], prob_dict[new_model])
        diff_low, diff_high = bootstrap_difference_ci(
            y_true=y_true,
            old_prob=prob_dict[old_model],
            new_prob=prob_dict[new_model],
            n_boot=n_boot,
            seed=seed + idx * 100,
        )
        rows.append(
            {
                "Comparison": f"{old_model} -> {new_model}",
                "Old model": old_model,
                "New model": new_model,
                **stats,
                "AUC difference 95% CI lower": diff_low,
                "AUC difference 95% CI upper": diff_high,
            }
        )
    return pd.DataFrame(rows)


def calculate_nri_idi_outputs(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    delong_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pairwise NRI/IDI with bootstrap CIs and patient-level data."""
    comparisons = [
        ("Clinical Model", "Integrated Model"),
        ("CT Model", "Clinical Model"),
        ("CT Model", "Integrated Model"),
    ]
    summary_rows = []
    patient_frames = []
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delong_lookup = delong_df.set_index("Comparison")

    for old_model, new_model in comparisons:
        old_prob = np.asarray(prob_dict[old_model], dtype=float)
        new_prob = np.asarray(prob_dict[new_model], dtype=float)
        cont_total, nri_event, nri_non_event = continuous_nri(y_true, old_prob, new_prob)
        cat_total, cat_event, cat_non_event = categorical_nri(y_true, old_prob, new_prob)
        idi_value = idi(y_true, old_prob, new_prob)
        auc_old = safe_auc(y_true, old_prob)
        auc_new = safe_auc(y_true, new_prob)
        diff_low, diff_high = bootstrap_difference_ci(y_true, old_prob, new_prob, n_boot=n_boot, seed=seed + 50)

        cont_boot = []
        cat_boot = []
        idi_boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            y_boot = y_true[idx]
            if len(np.unique(y_boot)) < 2:
                continue
            cont_boot.append(continuous_nri(y_boot, old_prob[idx], new_prob[idx])[0])
            cat_boot.append(categorical_nri(y_boot, old_prob[idx], new_prob[idx])[0])
            idi_boot.append(idi(y_boot, old_prob[idx], new_prob[idx]))

        def interval(values: List[float]) -> Tuple[float, float]:
            if len(values) < 20:
                return float("nan"), float("nan")
            return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

        cont_low, cont_high = interval(cont_boot)
        cat_low, cat_high = interval(cat_boot)
        idi_low, idi_high = interval(idi_boot)

        idi_p = float("nan")
        if idi_boot:
            idi_p = float(2 * min(np.mean(np.asarray(idi_boot) <= 0), np.mean(np.asarray(idi_boot) >= 0)))

        comparison_name = f"{old_model} -> {new_model}"
        delong_p = float(delong_lookup.loc[comparison_name, "DeLong_p_value"])
        summary_rows.append(
            {
                "Comparison": comparison_name,
                "AUC old": auc_old,
                "AUC new": auc_new,
                "AUC difference": auc_new - auc_old,
                "AUC difference 95% CI lower": diff_low,
                "AUC difference 95% CI upper": diff_high,
                "DeLong p value": delong_p,
                "Continuous NRI": cont_total,
                "Continuous NRI event": nri_event,
                "Continuous NRI non-event": nri_non_event,
                "Continuous NRI 95% CI lower": cont_low,
                "Continuous NRI 95% CI upper": cont_high,
                "Categorical NRI": cat_total,
                "Categorical NRI event": cat_event,
                "Categorical NRI non-event": cat_non_event,
                "Categorical NRI 95% CI lower": cat_low,
                "Categorical NRI 95% CI upper": cat_high,
                "IDI": idi_value,
                "IDI 95% CI lower": idi_low,
                "IDI 95% CI upper": idi_high,
                "IDI p value": idi_p,
            }
        )

        patient_frames.append(
            pd.DataFrame(
                {
                    "Comparison": comparison_name,
                    "sample_index": np.arange(n),
                    "true_label": y_true,
                    "old_model": old_model,
                    "new_model": new_model,
                    "old_probability": old_prob,
                    "new_probability": new_prob,
                    "probability_change": new_prob - old_prob,
                    "old_risk_category": assign_risk_category(old_prob),
                    "new_risk_category": assign_risk_category(new_prob),
                    "category_change": assign_risk_category(new_prob) - assign_risk_category(old_prob),
                }
            )
        )

    patient_level_df = pd.concat(patient_frames, ignore_index=True)
    return pd.DataFrame(summary_rows), patient_level_df


def build_fold_metric_rows(
    y_test: np.ndarray,
    test_prob: np.ndarray,
    model_name: str,
    outer_fold: int,
    selected_params: Dict[str, object],
    inner_mean_auc: float,
    thresholds: Dict[str, float],
) -> List[Dict[str, object]]:
    """Build per-fold metric rows for one model."""
    rows = []
    threshold_mapping = {
        "threshold_05": 0.5,
        "nested_threshold": thresholds["nested_threshold"],
        "sensitivity_080": thresholds["sensitivity_080"],
    }
    for threshold_type, threshold_value in threshold_mapping.items():
        metrics = classification_metrics(y_test, test_prob, threshold=threshold_value)
        rows.append(
            {
                "Fold": outer_fold,
                "Model": model_name,
                "Threshold type": threshold_type,
                "Threshold": threshold_value,
                "Best parameters": json_dumps(selected_params),
                "Inner CV mean AUC": inner_mean_auc,
                **metrics,
            }
        )
    return rows


def calculate_overall_performance(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    oof_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Calculate overall merged-OOF performance summaries."""
    rows = []
    prediction_col_lookup = {
        "threshold_05": {
            "CT Model": "prediction_CT_threshold_05",
            "Clinical Model": "prediction_Clinical_threshold_05",
            "Integrated Model": "prediction_Integrated_threshold_05",
        },
        "nested_threshold": {
            "CT Model": "prediction_CT_nested_threshold",
            "Clinical Model": "prediction_Clinical_nested_threshold",
            "Integrated Model": "prediction_Integrated_nested_threshold",
        },
        "sensitivity_080": {
            "CT Model": "prediction_CT_sensitivity_080",
            "Clinical Model": "prediction_Clinical_sensitivity_080",
            "Integrated Model": "prediction_Integrated_sensitivity_080",
        },
    }
    for model_name in MODEL_ORDER:
        prob = np.asarray(prob_dict[model_name], dtype=float)
        auc_value = safe_auc(y_true, prob)
        auc_low, auc_high = auc_ci_bootstrap(y_true, prob, n_boot=n_boot, seed=seed)
        slope, intercept = calibration_slope_intercept(y_true, prob)
        brier = float(np.mean((prob - y_true) ** 2))
        for threshold_type, col_lookup in prediction_col_lookup.items():
            pred = oof_df[col_lookup[model_name]].to_numpy(dtype=int)
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            rows.append(
                {
                    "Model": model_name,
                    "Threshold type": threshold_type,
                    "AUC": auc_value,
                    "AUC 95% CI lower": auc_low,
                    "AUC 95% CI upper": auc_high,
                    "AUC and 95% CI": format_auc_with_ci(auc_value, auc_low, auc_high),
                    "Accuracy": float(accuracy_score(y_true, pred)),
                    "Sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                    "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                    "PPV": float(precision_score(y_true, pred, zero_division=0)),
                    "NPV": float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                    "F1 score": float(f1_score(y_true, pred, zero_division=0)),
                    "Brier score": brier,
                    "Calibration slope": slope,
                    "Calibration intercept": intercept,
                }
            )
    return pd.DataFrame(rows)


def make_roc_figure(base_dir: Path, roc_auc_summary_df: pd.DataFrame, roc_curve_df: pd.DataFrame) -> None:
    """Save three-model ROC figure."""
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    summary_lookup = roc_auc_summary_df.set_index("Model")
    for model_name in MODEL_ORDER:
        sub = roc_curve_df[roc_curve_df["Model"] == model_name]
        row = summary_lookup.loc[model_name]
        ax.plot(
            sub["FPR"],
            sub["TPR"],
            linewidth=2.2,
            color=COLORS[model_name],
            label=(
                f"{model_name} (AUC={row['AUC']:.3f}, "
                f"95% CI {row['AUC 95% CI lower']:.3f}-{row['AUC 95% CI upper']:.3f})"
            ),
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    save_figure_bundle(fig, base_dir, "ROC_3models")


def make_calibration_figure(base_dir: Path, calibration_curve_df: pd.DataFrame) -> None:
    """Save three-model calibration figure."""
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    for model_name in MODEL_ORDER:
        sub = calibration_curve_df[calibration_curve_df["Model"] == model_name].sort_values("Bin")
        ax.plot(
            sub["Mean predicted probability"],
            sub["Observed event rate"],
            marker="o",
            linewidth=2.0,
            color=COLORS[model_name],
            label=model_name,
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Event Rate")
    ax.set_title("Calibration Curves")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    save_figure_bundle(fig, base_dir, "Calibration_3models")


def make_dca_figure(base_dir: Path, dca_curve_df: pd.DataFrame) -> None:
    """Save three-model DCA figure for the primary threshold range 0.10-0.60."""
    sub = dca_curve_df[dca_curve_df["Threshold range type"] == "main_010_060"]
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    ax.plot(sub["Threshold"], sub["CT net benefit"], color=COLORS["CT Model"], linewidth=2.0, label="CT Model")
    ax.plot(
        sub["Threshold"],
        sub["Clinical net benefit"],
        color=COLORS["Clinical Model"],
        linewidth=2.0,
        label="Clinical Model",
    )
    ax.plot(
        sub["Threshold"],
        sub["Integrated net benefit"],
        color=COLORS["Integrated Model"],
        linewidth=2.0,
        label="Integrated Model",
    )
    ax.plot(
        sub["Threshold"],
        sub["Treat All net benefit"],
        "--",
        color=COLORS["Treat All"],
        linewidth=1.5,
        label="Treat All",
    )
    ax.plot(
        sub["Threshold"],
        sub["Treat None net benefit"],
        ":",
        color=COLORS["Treat None"],
        linewidth=1.8,
        label="Treat None",
    )
    ax.set_xlim(float(sub["Threshold"].min()), float(sub["Threshold"].max()))
    ax.set_xlabel("Threshold Probability")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis (0.10-0.60)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=8, frameon=True)
    save_figure_bundle(fig, base_dir, "DCA_3models")


def make_performance_figure(base_dir: Path, overall_performance_df: pd.DataFrame) -> None:
    """Save an overall performance comparison figure based on nested thresholds."""
    metrics = ["AUC", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "F1 score", "Brier score"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    sub = overall_performance_df[overall_performance_df["Threshold type"] == "nested_threshold"].copy()
    sub = sub.set_index("Model").loc[MODEL_ORDER].reset_index()
    x = np.arange(len(MODEL_ORDER))
    for ax, metric in zip(axes, metrics):
        values = sub[metric].to_numpy(dtype=float)
        ax.bar(x, values, color=[COLORS[m] for m in MODEL_ORDER], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=18)
        ax.set_title(metric)
        ax.grid(alpha=0.25, linestyle="--", axis="y")
        if metric == "Brier score":
            ymax = min(1.0, max(0.20, float(np.max(values)) * 1.25))
            ax.set_ylim(0, ymax)
        else:
            ax.set_ylim(0, 1)
    fig.suptitle("Overall Performance Comparison (Nested Youden Threshold)", y=1.02)
    save_figure_bundle(fig, base_dir, "model_performance_comparison")


def write_analysis_report(
    output_dir: Path,
    input_csv: Path,
    y_true: np.ndarray,
    selected_params_df: pd.DataFrame,
    roc_auc_summary_df: pd.DataFrame,
    calibration_metrics_df: pd.DataFrame,
    dca_range_summary_df: pd.DataFrame,
    nri_idi_summary_df: pd.DataFrame,
) -> None:
    """Write a concise text report including the acceptance checklist."""
    main_nri = nri_idi_summary_df[nri_idi_summary_df["Comparison"] == "Clinical Model -> Integrated Model"].iloc[0]
    lines = [
        "Fair Three-Model CatBoost Analysis Report",
        f"Input CSV: {input_csv}",
        f"Sample size: {len(y_true)}",
        f"Events: {int(np.sum(y_true == 1))}",
        f"Non-events: {int(np.sum(y_true == 0))}",
        "",
        "Feature sets:",
        f"CT Model: {CT_FEATURES}",
        f"Clinical Model: {CLINICAL_FEATURES}",
        f"Integrated Model: {INTEGRATED_FEATURES}",
        "",
        "Study design checks:",
        "1. CT, Clinical, and Integrated models all use CatBoost: Yes",
        "2. Three models use the same outer folds: Yes",
        "3. Three models use the same inner folds within each outer fold: Yes",
        "4. Three models use the same hyperparameter search space: Yes",
        "5. Three models use the same hyperparameter search budget: Yes",
        "6. Three models compare the same class-weight candidates: Yes",
        "7. CT Model only contains Preoperative CT: Yes",
        "8. Clinical Model contains only the specified 7 variables: Yes",
        "9. Clinical Model excludes Gender and Age: Yes",
        "10. Integrated Model equals Clinical Model plus all 8 proteins: Yes",
        "11. Main analysis removes Logistic, Elastic Net, and fusion-strategy selection: Yes",
        "12. Overall evaluation uses merged outer OOF predictions: Yes",
        "13. No outer test data participate in tuning or threshold selection: Yes",
        "14. Script can run from the command line: Yes",
        "",
        "Selected hyperparameters by fold:",
        selected_params_df.to_string(index=False),
        "",
        "ROC AUC summary:",
        roc_auc_summary_df.to_string(index=False),
        "",
        "Calibration metrics:",
        calibration_metrics_df.to_string(index=False),
        "",
        "DCA range summary:",
        dca_range_summary_df.to_string(index=False),
        "",
        "NRI/IDI summary:",
        nri_idi_summary_df.to_string(index=False),
        "",
        "Primary incremental-value focus:",
        (
            "Clinical vs Integrated: "
            f"AUC difference={main_nri['AUC difference']:.3f}, "
            f"DeLong p={main_nri['DeLong p value']:.5f}, "
            f"Continuous NRI={main_nri['Continuous NRI']:.3f}, "
            f"Categorical NRI={main_nri['Categorical NRI']:.3f}, "
            f"IDI={main_nri['IDI']:.3f}"
        ),
    ]
    (output_dir / "analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    sample_id_col: Optional[str],
    input_csv: Path,
    output_dir: Path,
    param_iter: int,
    bootstrap_n: int,
    seed: int,
) -> None:
    """Run the strict nested-CV CatBoost comparison and write all outputs."""
    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
    outer_splits = list(outer_cv.split(df, y_true))

    n_samples = len(df)
    raw_prob_arrays = {model: np.full(n_samples, np.nan, dtype=float) for model in MODEL_ORDER}
    outer_fold_array = np.full(n_samples, np.nan)
    nested_threshold_arrays = {model: np.full(n_samples, np.nan, dtype=float) for model in MODEL_ORDER}
    sens080_threshold_arrays = {model: np.full(n_samples, np.nan, dtype=float) for model in MODEL_ORDER}
    prediction_arrays = {
        "threshold_05": {model: np.full(n_samples, np.nan, dtype=float) for model in MODEL_ORDER},
        "nested_threshold": {model: np.full(n_samples, np.nan, dtype=float) for model in MODEL_ORDER},
        "sensitivity_080": {model: np.full(n_samples, np.nan, dtype=float) for model in MODEL_ORDER},
    }

    per_fold_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    candidate_rows_for_export: List[Dict[str, object]] = []

    actual_feature_names = {
        model_name: [name for name in MODEL_FEATURES[model_name]]
        for model_name in MODEL_ORDER
    }
    actual_categorical_names = {
        model_name: [name for name in CATEGORICAL_FEATURES if name in MODEL_FEATURES[model_name]]
        for model_name in MODEL_ORDER
    }

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        y_train = y_true[train_idx]
        y_test = y_true[test_idx]

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=2400 + outer_fold,
        )
        inner_splits = list(inner_cv.split(train_df, y_train))
        candidate_params = list(
            ParameterSampler(
                SHARED_CATBOOST_PARAM_GRID,
                n_iter=param_iter,
                random_state=seed + outer_fold,
            )
        )
        class_weight_options = get_class_weight_options(y_train)

        model_results = {}
        for model_name in MODEL_ORDER:
            feature_names = actual_feature_names[model_name]
            categorical_names = actual_categorical_names[model_name]
            result = evaluate_model_candidates(
                model_name=model_name,
                train_df=train_df,
                y_train=y_train,
                feature_names=feature_names,
                categorical_names=categorical_names,
                inner_splits=inner_splits,
                candidate_params=candidate_params,
                class_weight_options=class_weight_options,
                outer_fold=outer_fold,
                seed=seed,
            )
            x_test, _ = prepare_dataframe(test_df, feature_names, categorical_names)
            test_prob = result["final_model"].predict_proba(x_test)[:, 1]
            model_results[model_name] = {**result, "test_prob": test_prob}

        outer_fold_array[test_idx] = outer_fold
        for model_name in MODEL_ORDER:
            result = model_results[model_name]
            raw_prob_arrays[model_name][test_idx] = result["test_prob"]

            nested_threshold = select_threshold_from_training_oof(
                y_true=y_train,
                y_prob=result["selected_train_oof"],
                mode="youden",
            )
            sens080_threshold = select_threshold_from_training_oof(
                y_true=y_train,
                y_prob=result["selected_train_oof"],
                mode="sens080",
            )

            nested_threshold_arrays[model_name][test_idx] = nested_threshold
            sens080_threshold_arrays[model_name][test_idx] = sens080_threshold
            prediction_arrays["threshold_05"][model_name][test_idx] = (result["test_prob"] >= 0.5).astype(int)
            prediction_arrays["nested_threshold"][model_name][test_idx] = (
                result["test_prob"] >= nested_threshold
            ).astype(int)
            prediction_arrays["sensitivity_080"][model_name][test_idx] = (
                result["test_prob"] >= sens080_threshold
            ).astype(int)

            per_fold_rows.extend(
                build_fold_metric_rows(
                    y_test=y_test,
                    test_prob=result["test_prob"],
                    model_name=model_name,
                    outer_fold=outer_fold,
                    selected_params=result["selected_params"],
                    inner_mean_auc=result["inner_mean_auc"],
                    thresholds={
                        "nested_threshold": nested_threshold,
                        "sensitivity_080": sens080_threshold,
                    },
                )
            )

            selected_rows.append(
                {
                    "Outer_fold": outer_fold,
                    "Model": model_name,
                    "Selected_hyperparameters": json_dumps(result["selected_params"]),
                    "Inner_mean_AUC": result["inner_mean_auc"],
                    "Inner_SD_AUC": result["inner_sd_auc"],
                    "Inner_Brier_score": result["inner_brier"],
                    "Inner_calibration_slope": result["inner_calibration_slope"],
                    "Inner_calibration_intercept": result["inner_calibration_intercept"],
                    "Inner_mean_DCA_net_benefit": result["inner_mean_dca_net_benefit"],
                    "Threshold_Youden": nested_threshold,
                    "Threshold_Sensitivity_ge_0.80": sens080_threshold,
                    "Selection_reason": result["selection_reason"],
                }
            )

            for row in result["candidate_rows"]:
                export_row = {k: v for k, v in row.items() if k not in {"train_oof", "best_params"}}
                export_row["Selected"] = export_row["Hyperparameters"] == json_dumps(result["selected_params"])
                candidate_rows_for_export.append(export_row)

    sample_ids = (
        df[sample_id_col].astype(str).to_numpy()
        if sample_id_col is not None
        else np.asarray([str(i) for i in range(n_samples)])
    )
    oof_df = pd.DataFrame(
        {
            "sample_index": np.arange(n_samples),
            "sample_id": sample_ids,
            "outer_fold": outer_fold_array.astype(int),
            "true_label": y_true,
            "prob_CT_Model": raw_prob_arrays["CT Model"],
            "prob_Clinical_Model": raw_prob_arrays["Clinical Model"],
            "prob_Integrated_Model": raw_prob_arrays["Integrated Model"],
            "prediction_CT_threshold_05": prediction_arrays["threshold_05"]["CT Model"].astype(int),
            "prediction_Clinical_threshold_05": prediction_arrays["threshold_05"]["Clinical Model"].astype(int),
            "prediction_Integrated_threshold_05": prediction_arrays["threshold_05"]["Integrated Model"].astype(int),
            "selected_threshold_CT": nested_threshold_arrays["CT Model"],
            "selected_threshold_Clinical": nested_threshold_arrays["Clinical Model"],
            "selected_threshold_Integrated": nested_threshold_arrays["Integrated Model"],
            "prediction_CT_nested_threshold": prediction_arrays["nested_threshold"]["CT Model"].astype(int),
            "prediction_Clinical_nested_threshold": prediction_arrays["nested_threshold"]["Clinical Model"].astype(int),
            "prediction_Integrated_nested_threshold": prediction_arrays["nested_threshold"]["Integrated Model"].astype(int),
            "selected_threshold_sensitivity_080_CT": sens080_threshold_arrays["CT Model"],
            "selected_threshold_sensitivity_080_Clinical": sens080_threshold_arrays["Clinical Model"],
            "selected_threshold_sensitivity_080_Integrated": sens080_threshold_arrays["Integrated Model"],
            "prediction_CT_sensitivity_080": prediction_arrays["sensitivity_080"]["CT Model"].astype(int),
            "prediction_Clinical_sensitivity_080": prediction_arrays["sensitivity_080"]["Clinical Model"].astype(int),
            "prediction_Integrated_sensitivity_080": prediction_arrays["sensitivity_080"]["Integrated Model"].astype(int),
        }
    ).sort_values("sample_index")
    validate_oof_predictions(oof_df)

    prob_dict = raw_prob_arrays
    roc_curve_df, roc_auc_summary_df = calculate_roc_outputs(y_true, prob_dict, bootstrap_n, seed)
    calibration_curve_df, calibration_metrics_df = calculate_calibration_outputs(y_true, prob_dict)
    dca_curve_df, dca_range_summary_df = calculate_dca_outputs(y_true, prob_dict)
    delong_df = calculate_delong_outputs(y_true, prob_dict, bootstrap_n, seed)
    nri_idi_summary_df, nri_patient_level_df = calculate_nri_idi_outputs(
        y_true=y_true,
        prob_dict=prob_dict,
        delong_df=delong_df,
        n_boot=bootstrap_n,
        seed=seed,
    )
    per_fold_df = pd.DataFrame(per_fold_rows)
    selected_params_df = pd.DataFrame(selected_rows)
    candidate_results_df = pd.DataFrame(candidate_rows_for_export)
    overall_performance_df = calculate_overall_performance(y_true, prob_dict, oof_df, bootstrap_n, seed)

    roc_curve_df.to_csv(output_dir / "ROC_curve_data.csv", index=False, encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_predictions_complete.csv", index=False, encoding="utf-8")
    overall_performance_df.to_csv(output_dir / "model_performance_overall.csv", index=False, encoding="utf-8")
    per_fold_df.to_csv(output_dir / "model_performance_by_fold.csv", index=False, encoding="utf-8")
    selected_params_df.to_csv(output_dir / "selected_hyperparameters_by_fold.csv", index=False, encoding="utf-8")
    candidate_results_df.to_csv(output_dir / "inner_cv_candidate_results.csv", index=False, encoding="utf-8")
    roc_auc_summary_df.to_csv(output_dir / "ROC_AUC_summary.csv", index=False, encoding="utf-8")
    delong_df.to_csv(output_dir / "DeLong_comparisons.csv", index=False, encoding="utf-8")
    calibration_metrics_df.to_csv(output_dir / "calibration_metrics.csv", index=False, encoding="utf-8")
    calibration_curve_df.to_csv(output_dir / "calibration_curve_data.csv", index=False, encoding="utf-8")
    dca_curve_df.to_csv(output_dir / "DCA_curve_data.csv", index=False, encoding="utf-8")
    dca_range_summary_df.to_csv(output_dir / "DCA_range_summary.csv", index=False, encoding="utf-8")
    nri_idi_summary_df.to_csv(output_dir / "NRI_IDI_summary.csv", index=False, encoding="utf-8")
    nri_patient_level_df.to_csv(output_dir / "NRI_IDI_patient_level_data.csv", index=False, encoding="utf-8")

    make_roc_figure(output_dir, roc_auc_summary_df, roc_curve_df)
    make_calibration_figure(output_dir, calibration_curve_df)
    make_dca_figure(output_dir, dca_curve_df)
    make_performance_figure(output_dir, overall_performance_df)

    write_analysis_report(
        output_dir=output_dir,
        input_csv=input_csv,
        y_true=y_true,
        selected_params_df=selected_params_df,
        roc_auc_summary_df=roc_auc_summary_df,
        calibration_metrics_df=calibration_metrics_df,
        dca_range_summary_df=dca_range_summary_df,
        nri_idi_summary_df=nri_idi_summary_df,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    default_input = locate_default_input(script_dir)
    parser = argparse.ArgumentParser(
        description="Strict three-model CatBoost comparison with fair nested cross-validation."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(default_input) if default_input is not None else None,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "three_model_catboost_outputs"),
        help="Directory for output tables and figures.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap resamples for AUC and NRI/IDI confidence intervals.",
    )
    parser.add_argument(
        "--param-iter",
        type=int,
        default=80,
        help="Number of sampled hyperparameter combinations per outer fold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for CV splitting and parameter sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    if not args.input_csv:
        raise FileNotFoundError("No input CSV was provided and no default CSV could be located.")

    set_random_seed(args.seed)
    input_csv = Path(args.input_csv).resolve()
    output_dir = ensure_dir(Path(args.output_dir))

    df = pd.read_csv(input_csv)
    column_map = build_column_mapping(df)
    check_input_data(df, column_map)

    rename_map = {actual: canonical for canonical, actual in column_map.items()}
    analysis_df = df.rename(columns=rename_map).copy()
    y_true = validate_target_binary(analysis_df[TARGET])
    sample_id_col = choose_sample_id_column(df)

    run_analysis(
        df=analysis_df,
        y_true=y_true,
        sample_id_col=sample_id_col,
        input_csv=input_csv,
        output_dir=output_dir,
        param_iter=args.param_iter,
        bootstrap_n=args.bootstrap,
        seed=args.seed,
    )
    print(f"Done. Outputs were written to: {output_dir}")


if __name__ == "__main__":
    main()
