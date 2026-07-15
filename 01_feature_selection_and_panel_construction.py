import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from boruta import BorutaPy
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

RANDOM_STATE = 42
INPUT_CSV = "/Users/zhangwy/Documents/约约爱科研/汉卫/转移预测模型/数据预处理/21_gene_clinical.csv"
OUT_DIR = "/Users/zhangwy/Documents/约约爱科研/汉卫/转移预测模型/数据预处理"


def get_base_models(random_state: int):
    return {
        "Boruta": RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "GBDT": GradientBoostingClassifier(random_state=random_state),
        "GaussianNB": GaussianNB(),
        "LightGBM": LGBMClassifier(random_state=random_state, verbosity=-1),
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=5000, random_state=random_state)),
            ]
        ),
        "PLSDA": "PLSDA",
        "RF": RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1),
        "XGBoost": XGBClassifier(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        ),
    }


def cv_auc(model_obj, x_sub, y_sub, cv):
    aucs = []
    for tr_idx, te_idx in cv.split(x_sub, y_sub):
        x_tr, x_te = x_sub[tr_idx], x_sub[te_idx]
        y_tr, y_te = y_sub[tr_idx], y_sub[te_idx]
        if model_obj == "PLSDA":
            scaler = StandardScaler()
            x_tr_s = scaler.fit_transform(x_tr)
            x_te_s = scaler.transform(x_te)
            n_comp = min(2, x_tr_s.shape[1])
            pls = PLSRegression(n_components=n_comp)
            pls.fit(x_tr_s, y_tr)
            y_score = pls.predict(x_te_s).ravel()
        else:
            model = model_obj
            model.fit(x_tr, y_tr)
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(x_te)[:, 1]
            else:
                y_score = model.decision_function(x_te)
        aucs.append(roc_auc_score(y_te, y_score))
    return float(np.mean(aucs))


def main():
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c != "LN_status"]
    x = df[feature_cols].values
    y = df["LN_status"].values

    rank_df = pd.DataFrame({"Feature": feature_cols})

    boruta_est = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    boruta = BorutaPy(
        estimator=boruta_est,
        n_estimators="auto",
        random_state=RANDOM_STATE,
        verbose=0,
    )
    boruta.fit(x, y)
    boruta_raw = boruta.ranking_.astype(float)
    rank_df["Boruta_raw"] = boruta_raw
    rank_df["Boruta_rank"] = pd.Series(boruta_raw).rank(method="min", ascending=True).values

    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(x, y)
    dt_raw = dt.feature_importances_
    rank_df["DT_raw"] = dt_raw
    rank_df["DT_rank"] = pd.Series(dt_raw).rank(method="min", ascending=False).values

    gbdt = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gbdt.fit(x, y)
    gbdt_raw = gbdt.feature_importances_
    rank_df["GBDT_raw"] = gbdt_raw
    rank_df["GBDT_rank"] = pd.Series(gbdt_raw).rank(method="min", ascending=False).values

    gnb = GaussianNB()
    gnb.fit(x, y)
    gnb_perm = permutation_importance(
        gnb,
        x,
        y,
        scoring="roc_auc",
        n_repeats=30,
        random_state=RANDOM_STATE,
    )
    gnb_raw = gnb_perm.importances_mean
    rank_df["GaussianNB_raw"] = gnb_raw
    rank_df["GaussianNB_rank"] = pd.Series(gnb_raw).rank(method="min", ascending=False).values

    lgbm = LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
    lgbm.fit(x, y)
    lgbm_raw = lgbm.feature_importances_.astype(float)
    rank_df["LightGBM_raw"] = lgbm_raw
    rank_df["LightGBM_rank"] = pd.Series(lgbm_raw).rank(method="min", ascending=False).values

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]
    )
    lr.fit(x, y)
    lr_raw = np.abs(lr.named_steps["lr"].coef_).ravel()
    rank_df["LR_raw"] = lr_raw
    rank_df["LR_rank"] = pd.Series(lr_raw).rank(method="min", ascending=False).values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pls = PLSRegression(n_components=min(2, x_scaled.shape[1]))
    pls.fit(x_scaled, y)
    pls_raw = np.abs(pls.coef_).ravel()
    rank_df["PLSDA_raw"] = pls_raw
    rank_df["PLSDA_rank"] = pd.Series(pls_raw).rank(method="min", ascending=False).values

    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(x, y)
    rf_raw = rf.feature_importances_
    rank_df["RF_raw"] = rf_raw
    rank_df["RF_rank"] = pd.Series(rf_raw).rank(method="min", ascending=False).values

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )
    xgb.fit(x, y)
    xgb_raw = xgb.feature_importances_
    rank_df["XGBoost_raw"] = xgb_raw
    rank_df["XGBoost_rank"] = pd.Series(xgb_raw).rank(method="min", ascending=False).values

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
    rank_df = rank_df.sort_values("Average_Rank", ascending=True).reset_index(drop=True)
    rank_df["Final_Rank"] = np.arange(1, len(rank_df) + 1)

    ordered_features = rank_df["Feature"].tolist()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rfe_rows = []
    for n_feat in range(len(ordered_features), 0, -1):
        selected = ordered_features[:n_feat]
        idx = [feature_cols.index(c) for c in selected]
        x_sub = x[:, idx]
        models = get_base_models(RANDOM_STATE)
        row = {"n_features": n_feat, "selected_features": ";".join(selected)}
        auc_values = []
        for model_name, model_obj in models.items():
            auc = cv_auc(model_obj, x_sub, y, cv)
            row[f"{model_name}_AUC"] = auc
            auc_values.append(auc)
        row["Mean_AUC_9Models"] = float(np.mean(auc_values))
        rfe_rows.append(row)

    rfe_df = pd.DataFrame(rfe_rows).sort_values("n_features", ascending=False).reset_index(drop=True)
    max_auc = rfe_df["Mean_AUC_9Models"].max()
    tie_df = rfe_df[np.isclose(rfe_df["Mean_AUC_9Models"], max_auc)]
    best_row = tie_df.sort_values("n_features", ascending=True).iloc[0]
    gold_n = int(best_row["n_features"])
    gold_features = best_row["selected_features"].split(";")

    table1_path = f"{OUT_DIR}/Step2_Table1_Composite_Ranking_9_Models.csv"
    table2_path = f"{OUT_DIR}/Step2_Table2_RFE_AUC_9_Models.csv"
    plot_path = f"{OUT_DIR}/Step2_FeatureCount_vs_MeanAUC.png"
    rank_df.to_csv(table1_path, index=False)
    rfe_df.to_csv(table2_path, index=False)

    plot_df = rfe_df.sort_values("n_features")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df["n_features"], plot_df["Mean_AUC_9Models"], marker="o")
    plt.xlabel("Number of Features")
    plt.ylabel("Mean AUC (9 Models, 5-Fold CV)")
    plt.title("Feature Count vs Mean AUC")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    print(f"TABLE1_PATH={table1_path}")
    print(f"TABLE2_PATH={table2_path}")
    print(f"PLOT_PATH={plot_path}")
    print(f"MAX_MEAN_AUC={max_auc:.6f}")
    print(f"GOLD_N_FEATURES={gold_n}")
    print("GOLD_FEATURES=" + ",".join(gold_features))


if __name__ == "__main__":
    main()
