

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from boruta import BorutaPy
except ImportError:  # pragma: no cover
    BorutaPy = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
DEFAULT_BASE_DIR = Path("/Users/zhangwy/Documents")
DEFAULT_EXPR_CSV = "preprocess data.csv"
DEFAULT_SPLIT_CSV = "final_60_40_split.csv"
DEFAULT_CANDIDATE_CSV = "candidates_gene.csv"
DEFAULT_OUT_DIR = "Result"
DEFAULT_AUC_TOLERANCE = 0.01
PARTIAL_RFE_NAME = "RFE_AUC_9_Models_RepeatedCV_Development.partial.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank candidate proteins and select a Development-cohort protein panel."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help=f"Directory containing the fixed input files (default: {DEFAULT_BASE_DIR}).",
    )
    parser.add_argument(
        "--expr-csv",
        default=DEFAULT_EXPR_CSV,
        help="Processed expression matrix filename.",
    )
    parser.add_argument(
        "--split-csv",
        default=DEFAULT_SPLIT_CSV,
        help="60/40 split and clinical-label filename.",
    )
    parser.add_argument(
        "--candidate-csv",
        default=DEFAULT_CANDIDATE_CSV,
        help="Candidate-protein filename. Must contain the fixed column 'Protein'.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Output directory, relative to --base-dir unless absolute.",
    )
    parser.add_argument(
        "--auc-tolerance",
        type=float,
        default=DEFAULT_AUC_TOLERANCE,
        help=(
            "RFE plateau rule: choose the smallest panel whose mean AUC is within "
            "this value of the global maximum. Use 0 for exact maximum only."
        ),
    )
    parser.add_argument(
        "--key-genes-output-csv",
        default="",
        help=(
            "Optional CSV filename for exporting the selected panel, relative "
            "to --out-dir. The strict downstream file is always key genes.xlsx."
        ),
    )
    return parser.parse_args()


def make_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    base_dir = args.base_dir.expanduser().resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    return (
        base_dir / args.expr_csv,
        base_dir / args.split_csv,
        base_dir / args.candidate_csv,
        out_dir,
    )


def require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following required files were not found:\n" + "\n".join(missing)
        )


def check_dependencies() -> None:
    missing = []
    if BorutaPy is None:
        missing.append("boruta")
    if LGBMClassifier is None:
        missing.append("lightgbm")
    if XGBClassifier is None:
        missing.append("xgboost")
    if missing:
        raise ImportError(
            "Missing required packages for the first-stage feature-selection pipeline: "
            + ", ".join(missing)
            + "\nInstall them in the Python environment you use to run this script, for example:\n"
            + "pip install boruta lightgbm xgboost"
        )


def normalize_id_series(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )


def ln_status_to_binary(values: pd.Series) -> np.ndarray:
    labels = values.astype(str).str.strip()
    expected = {"LN_positive", "LN_negative"}
    unexpected = sorted(set(labels.dropna().unique()) - expected)
    if unexpected:
        raise ValueError(
            "Unexpected LN_status values detected: "
            + ", ".join(unexpected)
            + ". Expected exactly LN_positive or LN_negative."
        )
    return labels.eq("LN_positive").astype(int).to_numpy()


def get_candidate_features(candidate_csv: Path) -> list[str]:
    candidate_df = pd.read_csv(candidate_csv, encoding="utf-8-sig")
    if "Protein" not in candidate_df.columns:
        raise ValueError(f"{candidate_csv} must contain a column named 'Protein'.")

    candidates = candidate_df["Protein"].dropna().astype(str).str.strip().tolist()
    if not candidates:
        raise ValueError("The candidate-protein list is empty.")

    duplicated = pd.Series(candidates)[pd.Series(candidates).duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Duplicated candidate proteins detected: {duplicated}")
    return candidates


def load_development_data(
    expr_csv: Path,
    split_csv: Path,
    candidate_csv: Path,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    candidates = get_candidate_features(candidate_csv)

    split_df = pd.read_csv(split_csv, dtype={"patient_id": str}, encoding="utf-8-sig")
    required_split_cols = {"patient_id", "cohort_assignment", "LN_status"}
    missing_split_cols = sorted(required_split_cols - set(split_df.columns))
    if missing_split_cols:
        raise ValueError(f"Split file is missing columns: {missing_split_cols}")

    if split_df["patient_id"].isna().any():
        raise ValueError("The split file contains missing patient_id values.")
    split_df["patient_id"] = normalize_id_series(split_df["patient_id"])
    if split_df["patient_id"].duplicated().any():
        duplicated = split_df.loc[split_df["patient_id"].duplicated(), "patient_id"].tolist()
        raise ValueError(f"Duplicate patient_id values in split file: {duplicated}")

    dev_df = split_df[split_df["cohort_assignment"].astype(str).str.strip().eq("Development")].copy()
    if dev_df.empty:
        raise ValueError("No rows with cohort_assignment == 'Development' were found.")

    expr_header = pd.read_csv(expr_csv, nrows=0, encoding="utf-8-sig")
    if "Sample" not in expr_header.columns:
        raise ValueError(f"{expr_csv} must contain a column named 'Sample'.")
    missing_expr_features = [feature for feature in candidates if feature not in expr_header.columns]
    if missing_expr_features:
        raise ValueError(
            "Candidate proteins missing from expression matrix: "
            + ", ".join(missing_expr_features)
        )

    expr_df = pd.read_csv(
        expr_csv,
        usecols=["Sample"] + candidates,
        dtype={"Sample": str},
        encoding="utf-8-sig",
    )
    if expr_df["Sample"].isna().any():
        raise ValueError("The expression matrix contains missing Sample values.")
    expr_df["Sample"] = normalize_id_series(expr_df["Sample"])
    if expr_df["Sample"].duplicated().any():
        duplicated = expr_df.loc[expr_df["Sample"].duplicated(), "Sample"].tolist()
        raise ValueError(f"Duplicate Sample values in expression matrix: {duplicated}")

    df = dev_df.merge(
        expr_df,
        left_on="patient_id",
        right_on="Sample",
        how="left",
        validate="one_to_one",
    )

    missing_patients = sorted(set(dev_df["patient_id"]) - set(expr_df["Sample"]))
    if missing_patients:
        raise ValueError(
            "Development patients missing from expression matrix: "
            + ", ".join(missing_patients)
        )

    if df[candidates].isna().any().any():
        missing_counts = df[candidates].isna().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Candidate expression matrix contains missing values: {missing_counts}")

    y = ln_status_to_binary(df["LN_status"])
    x = df[candidates].astype(float).to_numpy()

    if np.unique(y).size != 2:
        raise ValueError("Development cohort must contain both LN-positive and LN-negative cases.")

    return df, candidates, x, y


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        verbosity=0,
    )


def get_base_models() -> dict[str, object]:
    return {
        "Boruta": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=1
        ),
        "DT": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "GBDT": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "GaussianNB": GaussianNB(),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
        "LR": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "PLSDA": "PLSDA",
        "RF": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": make_xgb(),
    }


def cv_auc(model_obj: object, x_sub: np.ndarray, y: np.ndarray, cv) -> float:
    aucs = []
    for train_idx, test_idx in cv.split(x_sub, y):
        x_train, x_test = x_sub[train_idx], x_sub[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit the scaler on the training fold only.
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        if model_obj == "PLSDA":
            n_components = min(2, x_train_scaled.shape[1])
            model = PLSRegression(n_components=n_components)
            model.fit(x_train_scaled, y_train)
            y_score = model.predict(x_test_scaled).ravel()
        else:
            model = clone(model_obj)
            model.fit(x_train_scaled, y_train)
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(x_test_scaled)[:, 1]
            else:
                y_score = model.decision_function(x_test_scaled)

        aucs.append(roc_auc_score(y_test, y_score))

    return float(np.mean(aucs))


def rank_descending(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="min", ascending=False).to_numpy()


def build_composite_ranking(
    feature_cols: list[str], x: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    rank_df = pd.DataFrame({"Feature": feature_cols})

    boruta_estimator = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=1,
        class_weight="balanced",
    )
    boruta = BorutaPy(
        estimator=boruta_estimator,
        n_estimators="auto",
        random_state=RANDOM_STATE,
        verbose=0,
    )
    boruta.fit(x, y)
    boruta_raw = boruta.ranking_.astype(float)
    rank_df["Boruta_raw"] = boruta_raw
    rank_df["Boruta_rank"] = pd.Series(boruta_raw).rank(method="min").to_numpy()

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(x, y)
    rank_df["DT_raw"] = dt.feature_importances_
    rank_df["DT_rank"] = rank_descending(dt.feature_importances_)

    gbdt = GradientBoostingClassifier(random_state=RANDOM_STATE).fit(x, y)
    rank_df["GBDT_raw"] = gbdt.feature_importances_
    rank_df["GBDT_rank"] = rank_descending(gbdt.feature_importances_)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    gnb = GaussianNB().fit(x_scaled, y)
    gnb_perm = permutation_importance(
        gnb,
        x_scaled,
        y,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
    )
    rank_df["GaussianNB_raw"] = gnb_perm.importances_mean
    rank_df["GaussianNB_rank"] = rank_descending(gnb_perm.importances_mean)

    lgbm = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1).fit(x, y)
    rank_df["LightGBM_raw"] = lgbm.feature_importances_.astype(float)
    rank_df["LightGBM_rank"] = rank_descending(lgbm.feature_importances_)

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    ).fit(x, y)
    lr_raw = np.abs(lr.named_steps["lr"].coef_).ravel()
    rank_df["LR_raw"] = lr_raw
    rank_df["LR_rank"] = rank_descending(lr_raw)

    pls = PLSRegression(n_components=min(2, x_scaled.shape[1])).fit(x_scaled, y)
    pls_raw = np.abs(pls.coef_).ravel()
    rank_df["PLSDA_raw"] = pls_raw
    rank_df["PLSDA_rank"] = rank_descending(pls_raw)

    rf = RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=1
    ).fit(x, y)
    rank_df["RF_raw"] = rf.feature_importances_
    rank_df["RF_rank"] = rank_descending(rf.feature_importances_)

    xgb = make_xgb().fit(x, y)
    rank_df["XGBoost_raw"] = xgb.feature_importances_.astype(float)
    rank_df["XGBoost_rank"] = rank_descending(xgb.feature_importances_)

    rank_cols = [
        "Boruta_rank",
        "DT_rank",
        "GBDT_rank",
        "GaussianNB_rank",
        "LightGBM_rank",
        "LR_rank",
        "PLSDA_rank",
        "RF_rank",
        "XGBoost_rank",
    ]
    rank_df["Average_Rank"] = rank_df[rank_cols].mean(axis=1)
    rank_df = rank_df.sort_values(
        ["Average_Rank", "Feature"], ascending=[True, True]
    ).reset_index(drop=True)
    rank_df["Final_Rank"] = np.arange(1, len(rank_df) + 1)
    return rank_df


def run_rfe(
    feature_cols: list[str],
    x: np.ndarray,
    y: np.ndarray,
    ordered_features: list[str],
    out_dir: Path,
    auc_tolerance: float,
) -> tuple[pd.DataFrame, float, int, list[str]]:
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    feature_index = {feature: index for index, feature in enumerate(feature_cols)}
    rows = []

    for n_features in range(len(ordered_features), 0, -1):
        print(f"RFE_START n_features={n_features}", flush=True)
        selected = ordered_features[:n_features]
        indices = [feature_index[feature] for feature in selected]
        x_sub = x[:, indices]

        row = {
            "n_features": n_features,
            "selected_features": ";".join(selected),
        }
        auc_values = []
        for model_name, model_obj in get_base_models().items():
            print(f"  MODEL_START {model_name}", flush=True)
            auc = cv_auc(model_obj, x_sub, y, cv)
            row[f"{model_name}_AUC"] = auc
            auc_values.append(auc)

        row["Mean_AUC_9Models"] = float(np.mean(auc_values))
        rows.append(row)

        
        pd.DataFrame(rows).sort_values("n_features", ascending=False).to_csv(
            out_dir / PARTIAL_RFE_NAME,
            index=False,
        )
        print(
            f"RFE_DONE n_features={n_features} "
            f"Mean_AUC_9Models={row['Mean_AUC_9Models']:.6f}",
            flush=True,
        )

    rfe_df = (
        pd.DataFrame(rows)
        .sort_values("n_features", ascending=False)
        .reset_index(drop=True)
    )
    max_auc = float(rfe_df["Mean_AUC_9Models"].max())

    
    eligible = rfe_df[rfe_df["Mean_AUC_9Models"] >= max_auc - auc_tolerance]
    best_row = eligible.sort_values(
        ["n_features", "Mean_AUC_9Models"],
        ascending=[True, False],
    ).iloc[0]
    gold_n = int(best_row["n_features"])
    gold_features = str(best_row["selected_features"]).split(";")
    return rfe_df, max_auc, gold_n, gold_features


def write_outputs(
    out_dir: Path,
    dev_df: pd.DataFrame,
    feature_cols: list[str],
    rank_df: pd.DataFrame,
    rfe_df: pd.DataFrame,
    max_auc: float,
    gold_n: int,
    gold_features: list[str],
    auc_tolerance: float,
    expr_csv: Path,
    split_csv: Path,
    candidate_csv: Path,
    key_genes_output_csv: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    table1_path = out_dir / "Table1_Composite_Ranking_9_Models_Development.csv"
    table2_path = out_dir / "Table2_RFE_AUC_9_Models_RepeatedCV_Development.csv"
    plot_path = out_dir / "FeatureCount_vs_MeanAUC_RepeatedCV_Development.png"
    panel_path = out_dir / "Final_Panel_Development.csv"
    summary_path = out_dir / "Final_Panel_Summary_Development.txt"
    audit_path = out_dir / "Development_Merged_Candidate_Expression.csv"

    rank_df.to_csv(table1_path, index=False)
    rfe_df.to_csv(table2_path, index=False)
    panel_df = pd.DataFrame(
        {
            "Feature": gold_features,
            "Panel_Rank": np.arange(1, gold_n + 1),
        }
    )
    panel_df.to_csv(panel_path, index=False)

    key_genes_xlsx_path = out_dir / "key genes.xlsx"
    pd.DataFrame({"key genes": gold_features}).to_excel(key_genes_xlsx_path, index=False)
    if key_genes_output_csv:
        key_genes_csv_path = out_dir / key_genes_output_csv
        pd.DataFrame({"key genes": gold_features}).to_csv(key_genes_csv_path, index=False)
    else:
        key_genes_csv_path = None

    dev_df[["patient_id", "LN_status", "cohort_assignment", *feature_cols]].to_csv(
        audit_path,
        index=False,
    )

    plot_df = rfe_df.sort_values("n_features")
    plt.figure(figsize=(8, 5))
    plt.plot(
        plot_df["n_features"],
        plot_df["Mean_AUC_9Models"],
        marker="o",
        linewidth=1.2,
    )
    plt.axvline(gold_n, color="tab:red", linestyle="--", linewidth=1)
    plt.xlabel("Number of Features")
    plt.ylabel("Mean AUC (9 Models, Repeated 5-Fold CV x10)")
    plt.title("Feature Count vs Mean AUC, Development")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    status_counts = dev_df["LN_status"].value_counts().to_dict()
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Protein feature selection using Development cohort\n")
        handle.write(f"BASE_DIR={expr_csv.parent}\n")
        handle.write(f"EXPR_CSV={expr_csv.name}\n")
        handle.write(f"SPLIT_CSV={split_csv.name}\n")
        handle.write(f"CANDIDATE_CSV={candidate_csv.name}\n")
        handle.write("FEATURE_RANKING=9 models\n")
        handle.write("RFE=backward removal following composite ranking\n")
        handle.write(f"RFE_AUC_TOLERANCE={auc_tolerance}\n")
        handle.write(f"N_DEVELOPMENT={len(dev_df)}\n")
        handle.write(f"LN_STATUS_COUNTS={status_counts}\n")
        handle.write(f"N_CANDIDATES={len(feature_cols)}\n")
        handle.write(
            f"CV=RepeatedStratifiedKFold(n_splits={N_SPLITS}, "
            f"n_repeats={N_REPEATS})\n"
        )
        handle.write("SCALING=StandardScaler fitted on each CV training fold only\n")
        handle.write(f"MAX_MEAN_AUC={max_auc:.6f}\n")
        handle.write(f"GOLD_N_FEATURES={gold_n}\n")
        handle.write("GOLD_FEATURES=" + ",".join(gold_features) + "\n")

    run_config = {
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "auc_tolerance": auc_tolerance,
        "base_dir": str(expr_csv.parent),
        "expr_csv": str(expr_csv),
        "split_csv": str(split_csv),
        "candidate_csv": str(candidate_csv),
        "out_dir": str(out_dir),
        "n_development": int(len(dev_df)),
        "ln_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "n_candidates": int(len(feature_cols)),
        "max_mean_auc": max_auc,
        "gold_n_features": gold_n,
        "gold_features": gold_features,
        "output_files": {
            "composite_ranking": str(table1_path),
            "rfe_auc": str(table2_path),
            "feature_count_plot": str(plot_path),
            "final_panel": str(panel_path),
            "key_genes_xlsx": str(key_genes_xlsx_path),
            "key_genes_csv": str(key_genes_csv_path) if key_genes_csv_path else None,
            "development_audit_table": str(audit_path),
            "summary": str(summary_path),
        },
    }
    (out_dir / "feature_selection_run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"TABLE1_PATH={table1_path}")
    print(f"TABLE2_PATH={table2_path}")
    print(f"PLOT_PATH={plot_path}")
    print(f"PANEL_PATH={panel_path}")
    print(f"KEY_GENES_XLSX={key_genes_xlsx_path}")
    if key_genes_csv_path:
        print(f"KEY_GENES_CSV={key_genes_csv_path}")
    print(f"SUMMARY_PATH={summary_path}")
    print(f"N_DEVELOPMENT={len(dev_df)}")
    print(f"N_CANDIDATES={len(feature_cols)}")
    print(f"MAX_MEAN_AUC={max_auc:.6f}")
    print(f"GOLD_N_FEATURES={gold_n}")
    print("GOLD_FEATURES=" + ",".join(gold_features))


def main() -> None:
    args = parse_args()
    expr_csv, split_csv, candidate_csv, out_dir = make_paths(args)
    require_files(expr_csv, split_csv, candidate_csv)
    check_dependencies()
    out_dir.mkdir(parents=True, exist_ok=True)

   
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir.resolve()))

    print(f"BASE_DIR={args.base_dir.expanduser().resolve()}")
    print(f"EXPR_CSV={expr_csv}")
    print(f"SPLIT_CSV={split_csv}")
    print(f"CANDIDATE_CSV={candidate_csv}")
    print(f"OUT_DIR={out_dir}")

    dev_df, feature_cols, x, y = load_development_data(
        expr_csv=expr_csv,
        split_csv=split_csv,
        candidate_csv=candidate_csv,
    )

    print(f"LOADED_DEVELOPMENT={len(dev_df)}")
    print(f"LN_STATUS_COUNTS={dev_df['LN_status'].value_counts().to_dict()}")
    print(f"N_CANDIDATES={len(feature_cols)}")

    rank_df = build_composite_ranking(feature_cols, x, y)
    ordered_features = rank_df["Feature"].tolist()
    rfe_df, max_auc, gold_n, gold_features = run_rfe(
        feature_cols=feature_cols,
        x=x,
        y=y,
        ordered_features=ordered_features,
        out_dir=out_dir,
        auc_tolerance=args.auc_tolerance,
    )
    write_outputs(
        out_dir=out_dir,
        dev_df=dev_df,
        feature_cols=feature_cols,
        rank_df=rank_df,
        rfe_df=rfe_df,
        max_auc=max_auc,
        gold_n=gold_n,
        gold_features=gold_features,
        auc_tolerance=args.auc_tolerance,
        expr_csv=expr_csv,
        split_csv=split_csv,
        candidate_csv=candidate_csv,
        key_genes_output_csv=args.key_genes_output_csv,
    )


if __name__ == "__main__":
    main()
