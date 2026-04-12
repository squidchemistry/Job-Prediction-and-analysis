"""Prediction helpers used by the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from utils.skill_analysis import analyze_skill_gap
from utils.text_processing import extract_keywords, normalize_skill_text


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def predict_job_roles(skill_text: str, top_k: int = 3) -> tuple[pd.DataFrame, list[str]]:
    model, _ = load_artifacts()
    cleaned = normalize_skill_text(skill_text)
    keywords = extract_keywords(cleaned)

    probabilities = model.predict_proba([cleaned])[0]
    labels = model.classes_

    results = (
        pd.DataFrame({"job_role": labels, "confidence": probabilities})
        .sort_values("confidence", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    results["confidence_percent"] = (results["confidence"] * 100).round(2)
    return results, keywords


def build_role_insights(skill_text: str, predictions: pd.DataFrame) -> list[dict]:
    insights = []
    for _, row in predictions.iterrows():
        role = row["job_role"]
        gap_info = analyze_skill_gap(skill_text, role)
        insights.append(
            {
                "job_role": role,
                "confidence_percent": row["confidence_percent"],
                **gap_info,
            }
        )
    return insights
