"""Synthetic dataset generator for job role prediction."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


ROLE_SKILLS = {
    "Data Analyst": {
        "core": ["python", "sql", "tableau", "excel", "statistics", "power bi"],
        "secondary": ["pandas", "numpy", "data visualization", "reporting", "dashboards"],
    },
    "Data Scientist": {
        "core": ["python", "sql", "machine learning", "statistics", "pandas", "numpy"],
        "secondary": ["deep learning", "scikit-learn", "feature engineering", "data mining", "matplotlib"],
    },
    "Machine Learning Engineer": {
        "core": ["python", "machine learning", "scikit-learn", "tensorflow", "pytorch", "mlops"],
        "secondary": ["docker", "api development", "deployment", "feature engineering", "aws"],
    },
    "Business Analyst": {
        "core": ["sql", "excel", "power bi", "communication", "requirements gathering", "reporting"],
        "secondary": ["tableau", "stakeholder management", "presentation", "process mapping", "analytics"],
    },
    "Data Engineer": {
        "core": ["python", "sql", "etl", "spark", "airflow", "data warehousing"],
        "secondary": ["hadoop", "aws", "azure", "pipelines", "big data"],
    },
    "BI Developer": {
        "core": ["sql", "power bi", "tableau", "data modeling", "dashboards", "reporting"],
        "secondary": ["excel", "dax", "etl", "analytics", "visualization"],
    },
    "AI Engineer": {
        "core": ["python", "machine learning", "deep learning", "nlp", "tensorflow", "pytorch"],
        "secondary": ["llms", "transformers", "deployment", "mlops", "api development"],
    },
    "Software Engineer": {
        "core": ["python", "java", "git", "data structures", "algorithms", "api development"],
        "secondary": ["docker", "system design", "testing", "javascript", "cloud"],
    },
}

NOISE_SKILLS = [
    "linux",
    "problem solving",
    "teamwork",
    "communication",
    "documentation",
    "agile",
    "jira",
    "github",
    "rest api",
    "cloud",
]

SYNONYM_VARIANTS = {
    "machine learning": ["ml", "machine-learning"],
    "power bi": ["powerbi"],
    "data visualization": ["visualization"],
    "api development": ["apis", "rest api"],
    "deep learning": ["dl"],
    "requirements gathering": ["business requirements"],
}


def _maybe_variant(skill: str) -> str:
    variants = SYNONYM_VARIANTS.get(skill, [])
    return random.choice([skill] + variants) if variants else skill


def generate_dataset(samples_per_role: int = 90, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows: list[dict[str, str]] = []

    for role, skill_groups in ROLE_SKILLS.items():
        core = skill_groups["core"]
        secondary = skill_groups["secondary"]

        for _ in range(samples_per_role):
            selected = random.sample(core, k=random.randint(4, min(6, len(core))))
            selected += random.sample(secondary, k=random.randint(2, min(4, len(secondary))))

            if random.random() < 0.65:
                selected += random.sample(NOISE_SKILLS, k=random.randint(1, 3))

            selected = [_maybe_variant(skill) for skill in selected]
            random.shuffle(selected)
            rows.append({"skills": " ".join(dict.fromkeys(selected)), "job_role": role})

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    output_path = Path(__file__).with_name("job_skills_dataset.csv")
    df = generate_dataset()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")


if __name__ == "__main__":
    main()
