"""
Model Training & Evaluation Pipeline.
Trains the primary CatBoost classifier and baseline models (Decision Tree, Logistic Regression),
computes comprehensive evaluation metrics, generates evaluation curves & SHAP XAI plots.
"""
import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering

from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import shap


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_models():
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Load data
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    cat_cols = config["data"]["features"]["categorical"]
    num_cols = config["data"]["features"]["numerical"]
    all_features = cat_cols + num_cols
    target = config["data"]["target"]

    X_train, y_train = train_df[all_features], train_df[target]
    X_val, y_val = val_df[all_features], val_df[target]
    X_test, y_test = test_df[all_features], test_df[target]

    print("=" * 60)
    print("TRAINING CHAMPION MODEL: CatBoost (Oblivious Trees & Ordered Boosting)")
    print("=" * 60)

    # CatBoost native Pools
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)
    test_pool = Pool(X_test, y_test, cat_features=cat_cols)

    cb_params = config["model"]["params"]
    champion_model = CatBoostClassifier(**cb_params, cat_features=cat_cols)
    champion_model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=config["model"]["early_stopping_rounds"],
        verbose=100
    )

    # 2. Train Standard Lecture Baseline Models
    print("\n" + "=" * 60)
    print("TRAINING BASELINE 1: Logistic Regression")
    print("TRAINING BASELINE 2: Single Decision Tree")
    print("=" * 60)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    log_reg = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    log_reg.fit(X_train, y_train)

    decision_tree = Pipeline([
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
    ])
    decision_tree.fit(X_train, y_train)

    # 3. Comprehensive Evaluation
    models = {
        "CatBoost (Champion)": champion_model,
        "Decision Tree (Baseline)": decision_tree,
        "Logistic Regression (Baseline)": log_reg
    }

    results = {}
    plt.figure(figsize=(8, 6))

    for name, m in models.items():
        if name == "CatBoost (Champion)":
            y_pred = m.predict(test_pool)
            y_prob = m.predict_proba(test_pool)[:, 1]
        else:
            y_pred = m.predict(X_test)
            y_prob = m.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4),
            "ROC_AUC": round(float(auc), 4)
        }

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess")
    plt.title("Comparative ROC Curves: Champion CatBoost vs Baseline Lecture Models", fontsize=12)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=300)
    plt.close()

    print("\nEVALUATION RESULTS COMPARISON:")
    print(pd.DataFrame(results).T.to_string())

    # Export metrics.json for Quality Gate
    champion_metrics = {
        "test_accuracy": results["CatBoost (Champion)"]["Accuracy"],
        "test_precision": results["CatBoost (Champion)"]["Precision"],
        "test_recall": results["CatBoost (Champion)"]["Recall"],
        "test_f1": results["CatBoost (Champion)"]["F1_Score"],
        "test_roc_auc": results["CatBoost (Champion)"]["ROC_AUC"]
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(champion_metrics, f, indent=2)

    with open("reports/metrics_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # 4. Confusion Matrix Heatmap
    y_pred_cb = champion_model.predict(test_pool)
    cm = confusion_matrix(y_test, y_pred_cb)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Ready (0)", "Harvest Ready (1)"])
    disp.plot(cmap="Greens", ax=ax, values_format="d")
    plt.title("CatBoost Confusion Matrix (Test Set: 2022-2024)", fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=300)
    plt.close()

    # 5. Explainable AI: SHAP TreeExplainer
    print("\n" + "=" * 60)
    print("COMPUTING EXPLAINABLE AI (XAI): SHAP TreeExplainer")
    print("=" * 60)
    explainer = shap.TreeExplainer(champion_model)
    shap_values = explainer(test_pool)

    # SHAP Global Summary Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Global Feature Importance & Impact Direction (SHAP Beeswarm)", fontsize=12)
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=300)
    plt.close()

    # 6. Save Champion Model & Explainer
    joblib.dump(champion_model, "models/champion_model.joblib")
    joblib.dump(explainer, "models/shap_explainer.joblib")
    print("\nSaved trained model to 'models/champion_model.joblib'")
    print("Saved SHAP explainer to 'models/shap_explainer.joblib'")
    print("All evaluation plots generated in 'reports/' directory.")

    return results


if __name__ == "__main__":
    train_models()
