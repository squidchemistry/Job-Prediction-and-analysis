"""Model training pipeline for job role prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data.generate_dataset import generate_dataset
from utils.skill_analysis import ROLE_KNOWLEDGE
from utils.text_processing import normalize_skill_text


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "job_skills_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
METRICS_PATH = BASE_DIR / "models" / "model_comparison.csv"


def load_or_create_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)

    df = generate_dataset(samples_per_role=90, seed=42)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    return df


def train_and_save_models() -> tuple[str, pd.DataFrame]:
    df = load_or_create_dataset().copy()
    df["clean_skills"] = df["skills"].astype(str).apply(normalize_skill_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_skills"],
        df["job_role"],
        test_size=0.2,
        random_state=42,
        stratify=df["job_role"],
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42),
        "Naive Bayes": MultinomialNB(),
    }

    comparison_rows = []
    fitted_models = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        }
        comparison_rows.append(metrics)
        fitted_models[model_name] = pipeline

    metrics_df = pd.DataFrame(comparison_rows).sort_values(
        by=["accuracy", "precision", "recall"], ascending=False
    )
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    metrics_df.to_csv(METRICS_PATH, index=False)

    metadata = {
        "best_model_name": best_model_name,
        "roles": sorted(df["job_role"].unique().tolist()),
        "role_knowledge": ROLE_KNOWLEDGE,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return best_model_name, metrics_df


if __name__ == "__main__":
    best_model_name, metrics_df = train_and_save_models()
    print(f"Best model: {best_model_name}")
    print(metrics_df.to_string(index=False))
