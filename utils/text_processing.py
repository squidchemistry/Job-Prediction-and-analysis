"""Utilities for text cleaning, synonym handling, and skill extraction."""

from __future__ import annotations

import re
from typing import Iterable


SYNONYM_MAP = {
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "dl": "deep learning",
    "powerbi": "power bi",
    "nlp": "natural language processing",
    "rest api": "api development",
    "apis": "api development",
    "biz analytics": "analytics",
    "business requirements": "requirements gathering",
}

SKILL_KEYWORDS = sorted(
    {
        "python",
        "sql",
        "tableau",
        "excel",
        "statistics",
        "power bi",
        "pandas",
        "numpy",
        "data visualization",
        "reporting",
        "dashboards",
        "machine learning",
        "deep learning",
        "scikit-learn",
        "data mining",
        "matplotlib",
        "tensorflow",
        "pytorch",
        "mlops",
        "docker",
        "api development",
        "deployment",
        "aws",
        "communication",
        "requirements gathering",
        "stakeholder management",
        "presentation",
        "process mapping",
        "analytics",
        "etl",
        "spark",
        "airflow",
        "data warehousing",
        "hadoop",
        "azure",
        "pipelines",
        "big data",
        "data modeling",
        "dax",
        "visualization",
        "natural language processing",
        "llms",
        "transformers",
        "java",
        "git",
        "data structures",
        "algorithms",
        "system design",
        "testing",
        "javascript",
        "cloud",
        "linux",
        "problem solving",
        "teamwork",
        "documentation",
        "agile",
        "jira",
        "github",
        "feature engineering",
    },
    key=len,
    reverse=True,
)


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_skill_text(text: str) -> str:
    normalized = f" {clean_text(text)} "
    for synonym, canonical in SYNONYM_MAP.items():
        normalized = normalized.replace(f" {synonym} ", f" {canonical} ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_keywords(text: str, skill_catalog: Iterable[str] | None = None) -> list[str]:
    catalog = list(skill_catalog) if skill_catalog is not None else SKILL_KEYWORDS
    normalized = normalize_skill_text(text)
    found: list[str] = []

    for skill in catalog:
        if re.search(rf"\b{re.escape(skill)}\b", normalized):
            found.append(skill)

    if found:
        return found

    # Fallback: treat tokens as rough keywords if nothing matches the catalog.
    return [token for token in normalized.split() if len(token) > 2]
