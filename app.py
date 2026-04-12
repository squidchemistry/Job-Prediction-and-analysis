"""Streamlit dashboard for job role prediction using skills."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from models.predict import build_role_insights, predict_job_roles
from models.train_model import MODEL_PATH, METADATA_PATH, METRICS_PATH, train_and_save_models
from utils.resume_parser import extract_skills_from_resume


st.set_page_config(page_title="Job Role Prediction using Skills", page_icon="AI", layout="wide")


def ensure_artifacts() -> None:
    if MODEL_PATH.exists() and METADATA_PATH.exists() and METRICS_PATH.exists():
        return
    train_and_save_models()


ensure_artifacts()

st.title("Job Role Prediction using Skills")
st.caption("Data Mining & Machine Learning project with model comparison, skill-gap analysis, and career recommendations.")

with st.sidebar:
    st.header("About")
    st.write(
        "Enter skills manually or upload a PDF resume. The app predicts the top matching roles, "
        "shows confidence scores, missing skills, salary estimates, and possible career paths."
    )
    if st.button("Retrain Models"):
        best_model_name, metrics_df = train_and_save_models()
        st.success(f"Training complete. Best model: {best_model_name}")
        st.dataframe(metrics_df, use_container_width=True)

col1, col2 = st.columns([2, 1])

with col1:
    manual_input = st.text_area(
        "Enter your skills",
        placeholder="Example: Python SQL Tableau Power BI Statistics Machine Learning",
        height=150,
    )

with col2:
    uploaded_resume = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    extracted_skills = []
    if uploaded_resume is not None:
        extracted_skills = extract_skills_from_resume(uploaded_resume.read())
        if extracted_skills:
            st.success("Skills extracted from resume")
            st.write(", ".join(extracted_skills))
        else:
            st.warning("No known skills were detected from the uploaded PDF.")

combined_input = manual_input.strip()
if extracted_skills:
    combined_input = f"{combined_input} {' '.join(extracted_skills)}".strip()

if st.button("Predict Job Roles", type="primary", use_container_width=True):
    if not combined_input:
        st.error("Please enter skills or upload a resume before predicting.")
    else:
        predictions, keywords = predict_job_roles(combined_input)
        insights = build_role_insights(combined_input, predictions)
        metrics_df = pd.read_csv(METRICS_PATH)

        st.subheader("Detected Keywords")
        st.write(", ".join(keywords) if keywords else "No keywords detected")

        a, b = st.columns([1, 1])

        with a:
            st.subheader("Top 3 Predicted Roles")
            st.dataframe(
                predictions[["job_role", "confidence_percent"]].rename(
                    columns={"confidence_percent": "confidence (%)"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with b:
            st.subheader("Prediction Confidence")
            chart_df = predictions.set_index("job_role")["confidence_percent"]
            st.bar_chart(chart_df)

        st.subheader("Skill Gap Analysis and Recommendations")
        for insight in insights:
            with st.container(border=True):
                st.markdown(
                    f"### {insight['job_role']} ({insight['confidence_percent']}% match)"
                )
                st.write(f"Known matching skills: {', '.join(insight['user_skills']) or 'None detected'}")
                st.write(f"Missing skills: {', '.join(insight['missing_skills']) or 'No major gaps'}")
                st.write(f"Suggested next skills: {', '.join(insight['suggested_skills']) or 'Keep building projects'}")
                st.write(f"Estimated salary range (LPA): {insight['salary_range_lpa']}")
                st.write(f"Career path: {insight['career_path']}")

        st.subheader("Model Performance Comparison")
        st.dataframe(metrics_df.round(4), use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    """
    **Expected workflow**

    1. Enter skills such as `Python SQL Tableau`.
    2. Click **Predict Job Roles**.
    3. Review predicted roles, confidence chart, missing skills, and the suggested learning path.
    """
)
