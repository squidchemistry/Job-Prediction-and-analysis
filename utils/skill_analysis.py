"""Skill gap, salary, and career-path helpers."""

from __future__ import annotations

from typing import Any

from utils.text_processing import extract_keywords, normalize_skill_text


ROLE_KNOWLEDGE = {
    "Data Analyst": {
        "required_skills": ["sql", "excel", "tableau", "power bi", "statistics", "python"],
        "recommended_skills": ["pandas", "numpy", "data visualization", "reporting"],
        "salary_range_lpa": "5 - 10",
        "career_path": "Data Analyst -> Senior Data Analyst -> Analytics Manager",
    },
    "Data Scientist": {
        "required_skills": ["python", "sql", "machine learning", "statistics", "pandas", "numpy"],
        "recommended_skills": ["deep learning", "scikit-learn", "data mining", "feature engineering"],
        "salary_range_lpa": "8 - 18",
        "career_path": "Data Scientist -> Senior Data Scientist -> Lead Data Scientist",
    },
    "Machine Learning Engineer": {
        "required_skills": ["python", "machine learning", "scikit-learn", "tensorflow", "pytorch", "mlops"],
        "recommended_skills": ["docker", "deployment", "aws", "feature engineering"],
        "salary_range_lpa": "10 - 22",
        "career_path": "ML Engineer -> Senior ML Engineer -> AI Platform Lead",
    },
    "Business Analyst": {
        "required_skills": ["sql", "excel", "power bi", "communication", "requirements gathering"],
        "recommended_skills": ["tableau", "stakeholder management", "presentation", "analytics"],
        "salary_range_lpa": "6 - 12",
        "career_path": "Business Analyst -> Senior BA -> Product or Strategy Manager",
    },
    "Data Engineer": {
        "required_skills": ["python", "sql", "etl", "spark", "airflow", "data warehousing"],
        "recommended_skills": ["aws", "azure", "pipelines", "big data"],
        "salary_range_lpa": "9 - 20",
        "career_path": "Data Engineer -> Senior Data Engineer -> Data Architect",
    },
    "BI Developer": {
        "required_skills": ["sql", "power bi", "tableau", "data modeling", "dashboards"],
        "recommended_skills": ["dax", "etl", "excel", "reporting"],
        "salary_range_lpa": "6 - 14",
        "career_path": "BI Developer -> Senior BI Developer -> BI Manager",
    },
    "AI Engineer": {
        "required_skills": ["python", "machine learning", "deep learning", "natural language processing", "tensorflow", "pytorch"],
        "recommended_skills": ["llms", "transformers", "deployment", "mlops"],
        "salary_range_lpa": "12 - 25",
        "career_path": "AI Engineer -> Senior AI Engineer -> Applied AI Lead",
    },
    "Software Engineer": {
        "required_skills": ["python", "java", "git", "data structures", "algorithms", "api development"],
        "recommended_skills": ["docker", "system design", "testing", "cloud"],
        "salary_range_lpa": "6 - 18",
        "career_path": "Software Engineer -> Senior Engineer -> Engineering Manager",
    },
}


def analyze_skill_gap(user_text: str, role: str, top_n: int = 4) -> dict[str, Any]:
    normalized = normalize_skill_text(user_text)
    user_skills = set(extract_keywords(normalized))
    role_info = ROLE_KNOWLEDGE[role]
    required = set(role_info["required_skills"])
    recommended = set(role_info["recommended_skills"])

    missing_required = sorted(required - user_skills)
    recommended_next = sorted((recommended | set(missing_required)) - user_skills)[:top_n]

    return {
        "user_skills": sorted(user_skills),
        "missing_skills": missing_required,
        "suggested_skills": recommended_next,
        "salary_range_lpa": role_info["salary_range_lpa"],
        "career_path": role_info["career_path"],
    }
