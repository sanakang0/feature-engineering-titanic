import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)

DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "assignment_outputs"
FIG_DIR = OUT_DIR / "figures"

RANDOM_STATE = 42
TOP_N_FEATURES = 12


EXPERIMENTS = [
    {
        "name": "Base",
        "imputation": "none",
        "encoding": "none",
        "scaling": "none",
        "feature_selection": False,
    },
    {
        "name": "Exp-1",
        "imputation": "mean",
        "encoding": "onehot",
        "scaling": "standard",
        "feature_selection": False,
    },
    {
        "name": "Exp-2",
        "imputation": "median",
        "encoding": "label",
        "scaling": "minmax",
        "feature_selection": True,
    },
    {
        "name": "Exp-3",
        "imputation": "most_frequent",
        "encoding": "onehot",
        "scaling": "robust",
        "feature_selection": True,
    },
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    out["Title"] = (
        out["Name"]
        .fillna("Unknown, Unknown.")
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .fillna("Unknown")
        .str.strip()
    )

    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)
    out["FarePerPerson"] = out["FarePerPerson"].replace([np.inf, -np.inf], np.nan)

    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[0, 16, 32, 48, 64, np.inf],
        labels=["Child", "YoungAdult", "Adult", "MiddleAge", "Senior"],
        include_lowest=True,
    )

    return out


def save_dataset_intro(df: pd.DataFrame) -> None:
    column_desc = [
        ("Survived", "Target (0: no, 1: yes)"),
        ("Pclass", "Ticket class"),
        ("Sex", "Passenger sex"),
        ("Age", "Passenger age"),
        ("SibSp", "# siblings/spouses aboard"),
        ("Parch", "# parents/children aboard"),
        ("Fare", "Passenger fare"),
        ("Embarked", "Port of embarkation"),
        ("FamilySize", "Derived: SibSp + Parch + 1"),
        ("IsAlone", "Derived: 1 if FamilySize == 1"),
        ("Title", "Derived from Name"),
        ("FarePerPerson", "Derived: Fare / FamilySize"),
        ("AgeGroup", "Derived age bucket"),
    ]
    desc_df = pd.DataFrame(column_desc, columns=["column", "description"])
    desc_df.to_csv(OUT_DIR / "column_description.csv", index=False)

    with open(OUT_DIR / "dataset_shape.txt", "w", encoding="utf-8") as f:
        f.write(f"shape: {df.shape}\n")


def perform_eda(df: pd.DataFrame) -> None:
    missing_ratio = (
        (df.isna().mean() * 100).sort_values(ascending=False).rename("missing_pct")
    )
    missing_ratio.to_csv(OUT_DIR / "eda_missing_ratio.csv", header=True)

    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "FarePerPerson"]
    outlier_records = []
    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_cnt = int(((series < lower) | (series > upper)).sum())
        outlier_records.append(
            {
                "feature": col,
                "outlier_count": outlier_cnt,
                "outlier_ratio": outlier_cnt / max(len(series), 1),
            }
        )
    pd.DataFrame(outlier_records).to_csv(
        OUT_DIR / "eda_outlier_summary.csv", index=False
    )

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], kde=True, bins=30)
    plt.title("Histogram of Age")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hist_age.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df[["Age", "Fare", "FamilySize"]])
    plt.title("Boxplot of Key Numeric Features")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "boxplot_numeric.png", dpi=150)
    plt.close()

    corr_cols = [
        "Survived",
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "FarePerPerson",
        "IsAlone",
    ]
    corr = df[corr_cols].corr(numeric_only=True)
    corr.to_csv(OUT_DIR / "eda_correlation_matrix.csv")

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "heatmap_corr.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Survived")
    plt.title("Target Distribution (Survived)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "countplot_target.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Pclass", y="Survived", estimator=np.mean, errorbar=None)
    plt.title("Mean Survival by Ticket Class")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "barplot_survival_by_pclass.png", dpi=150)
    plt.close()


def get_feature_columns() -> tuple[list[str], list[str]]:
    numeric_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "FarePerPerson",
    ]
    categorical_features = ["Sex", "Embarked", "Title", "AgeGroup"]
    return numeric_features, categorical_features


def build_transformer(
    config: dict, numeric_features: list[str], categorical_features: list[str]
):
    imputation = config["imputation"]
    encoding = config["encoding"]
    scaling = config["scaling"]

    if config["name"] == "Base":
        # Base는 전처리 비교 기준을 위해 수치형 원본만 사용하고 결측 행을 제거한다.
        return None

    if imputation in {"mean", "median"}:
        num_imputer = SimpleImputer(strategy=imputation)
        cat_imputer = SimpleImputer(strategy="most_frequent")
    elif imputation == "most_frequent":
        num_imputer = SimpleImputer(strategy="most_frequent")
        cat_imputer = SimpleImputer(strategy="most_frequent")
    else:
        num_imputer = "passthrough"
        cat_imputer = "passthrough"

    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "none": "passthrough",
    }
    num_scaler = scaler_map[scaling]

    if encoding == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding == "label":
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        cat_encoder = "passthrough"

    numeric_pipeline = Pipeline(
        steps=[("imputer", num_imputer), ("scaler", num_scaler)]
    )
    categorical_pipeline = Pipeline(
        steps=[("imputer", cat_imputer), ("encoder", cat_encoder)]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return transformer


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def run_experiments(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features, categorical_features = get_feature_columns()
    y = df["Survived"].astype(int).values

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE
        ),
    }

    all_rows = []
    fs_rows = []

    for config in EXPERIMENTS:
        if config["name"] == "Base":
            base_features = [
                "Pclass",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "FamilySize",
                "IsAlone",
                "FarePerPerson",
            ]
            temp = df[base_features + ["Survived"]].dropna().copy()
            X = temp[base_features]
            y_local = temp["Survived"].astype(int).values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_local, test_size=0.2, random_state=RANDOM_STATE, stratify=y_local
            )
            X_train_t = X_train.values
            X_test_t = X_test.values
            feature_names = base_features
        else:
            X = df[numeric_features + categorical_features].copy()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            transformer = build_transformer(
                config, numeric_features, categorical_features
            )
            X_train_t = transformer.fit_transform(X_train)
            X_test_t = transformer.transform(X_test)

            fn_num = numeric_features
            fn_cat = list(
                transformer.named_transformers_["cat"]
                .named_steps["encoder"]
                .get_feature_names_out(categorical_features)
            )
            feature_names = fn_num + fn_cat

        selected_indices = list(range(X_train_t.shape[1]))
        selected_features = feature_names

        if config["feature_selection"]:
            selector_model = RandomForestClassifier(
                n_estimators=400, random_state=RANDOM_STATE
            )
            selector_model.fit(X_train_t, y_train)
            importances = selector_model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            selected_indices = sorted_idx[: min(TOP_N_FEATURES, len(sorted_idx))]
            selected_features = [feature_names[i] for i in selected_indices]

            fs_df = pd.DataFrame(
                {
                    "experiment": config["name"],
                    "feature": feature_names,
                    "importance": importances,
                    "selected": [
                        i in set(selected_indices) for i in range(len(feature_names))
                    ],
                }
            ).sort_values("importance", ascending=False)
            fs_rows.append(fs_df)

        for model_name, base_model in models.items():
            model_before = clone(base_model)
            model_before.fit(X_train_t, y_train)
            pred_before = model_before.predict(X_test_t)
            prob_before = model_before.predict_proba(X_test_t)[:, 1]
            m_before = metric_dict(y_test, pred_before, prob_before)

            all_rows.append(
                {
                    "experiment": config["name"],
                    "model": model_name,
                    "stage": "before_fs",
                    **m_before,
                    "imputation": config["imputation"],
                    "encoding": config["encoding"],
                    "scaling": config["scaling"],
                    "feature_selection": config["feature_selection"],
                }
            )

            if config["feature_selection"]:
                model_after = clone(base_model)
                X_train_sel = X_train_t[:, selected_indices]
                X_test_sel = X_test_t[:, selected_indices]
                model_after.fit(X_train_sel, y_train)
                pred_after = model_after.predict(X_test_sel)
                prob_after = model_after.predict_proba(X_test_sel)[:, 1]
                m_after = metric_dict(y_test, pred_after, prob_after)

                all_rows.append(
                    {
                        "experiment": config["name"],
                        "model": model_name,
                        "stage": "after_fs",
                        **m_after,
                        "imputation": config["imputation"],
                        "encoding": config["encoding"],
                        "scaling": config["scaling"],
                        "feature_selection": config["feature_selection"],
                    }
                )

        if config["feature_selection"]:
            with open(
                OUT_DIR / f"selected_features_{config['name']}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(selected_features, f, ensure_ascii=False, indent=2)

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUT_DIR / "experiment_results.csv", index=False)

    if fs_rows:
        pd.concat(fs_rows, ignore_index=True).to_csv(
            OUT_DIR / "feature_importance_details.csv", index=False
        )

    return result_df


def summarize_results(result_df: pd.DataFrame) -> None:
    pivot = result_df.pivot_table(
        index=["experiment", "model", "stage"],
        values=["accuracy", "precision", "recall", "f1", "roc_auc"],
        aggfunc="mean",
    ).reset_index()
    pivot.to_csv(OUT_DIR / "result_summary_table.csv", index=False)

    best_row = result_df.sort_values("f1", ascending=False).iloc[0]

    exp_table = pd.DataFrame(EXPERIMENTS)
    exp_table["feature_selection"] = exp_table["feature_selection"].map(
        {True: "O", False: "X"}
    )
    exp_table.to_csv(OUT_DIR / "required_experiment_design_table.csv", index=False)

    report_lines = [
        "# Feature Engineering Assignment Report (Draft)",
        "",
        "## 1. Dataset Introduction",
        f"- Dataset: Titanic (URL: {DATA_URL})",
        "- Task: Binary classification for passenger survival (Survived)",
        "- Samples: 891 (>= 500 requirement satisfied)",
        "- Data types: numeric + categorical mixed",
        "",
        "### Data Shape",
        (OUT_DIR / "dataset_shape.txt").read_text(encoding="utf-8").strip(),
        "",
        "### Column Description",
        pd.read_csv(OUT_DIR / "column_description.csv").to_markdown(index=False),
        "",
        "## 2. EDA Results",
        "- Missing value ratios saved to assignment_outputs/eda_missing_ratio.csv",
        "- Outlier summary saved to assignment_outputs/eda_outlier_summary.csv",
        "- Required plots created:",
        "  - Histogram: assignment_outputs/figures/hist_age.png",
        "  - Boxplot: assignment_outputs/figures/boxplot_numeric.png",
        "  - Heatmap: assignment_outputs/figures/heatmap_corr.png",
        "  - Countplot: assignment_outputs/figures/countplot_target.png",
        "  - Barplot: assignment_outputs/figures/barplot_survival_by_pclass.png",
        "",
        "## 3. Feature Engineering",
        "- Derived features: FamilySize, IsAlone, Title, FarePerPerson, AgeGroup",
        "- Imputation comparison: none(base) / mean / median / most_frequent",
        "- Encoding comparison: none(base) / onehot / label(ordinal)",
        "- Scaling comparison: none(base) / standard / minmax / robust",
        "",
        "## 4. Feature Selection",
        "- Method: RandomForest feature importance top-N selection",
        f"- Top N: {TOP_N_FEATURES}",
        "- Before/after comparison included for Exp-2 and Exp-3",
        "",
        "## 5. Model Training & Evaluation",
        "- Models: LogisticRegression, RandomForest",
        "- Metrics: accuracy, precision, recall, f1, roc_auc",
        "",
        "### Required Experiment Design Table",
        exp_table.to_markdown(index=False),
        "",
        "### Performance Summary",
        pivot.to_markdown(index=False),
        "",
        "## 6. Final Conclusion (Auto-generated)",
        (
            f"Best setting was {best_row['experiment']} / {best_row['model']} / {best_row['stage']} "
            f"with F1={best_row['f1']:.4f}, ROC-AUC={best_row['roc_auc']:.4f}."
        ),
        "- Check feature_importance_details.csv and selected_features_Exp-2/3.json for selected variable details.",
    ]

    with open(OUT_DIR / "report_draft.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def main() -> None:
    ensure_dirs()

    df_raw = load_data()
    df = add_derived_features(df_raw)

    save_dataset_intro(df)
    perform_eda(df)
    results = run_experiments(df)
    summarize_results(results)

    print("Assignment pipeline completed.")
    print(f"Outputs saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
