# Job Role Prediction using Skills

This project predicts the most suitable job roles from a user's skills and highlights missing skills, recommended next skills, salary estimates, and a possible career path. It is designed as a beginner-friendly full-stack AI mini-product using Python, scikit-learn, and Streamlit.

## Features

- Skill text cleaning and normalization
- Synonym handling such as `ML -> Machine Learning`
- TF-IDF feature extraction
- Three trained models:
  - Logistic Regression
  - Random Forest
  - Naive Bayes
- Model comparison using accuracy, precision, and recall
- Top 3 role predictions with confidence scores
- Skill-gap analysis and learning recommendations
- PDF resume upload and skill extraction
- Salary range and career path suggestions

## Project Structure

```text
Job Prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── generate_dataset.py
│   └── job_skills_dataset.csv
├── models/
│   ├── train_model.py
│   ├── predict.py
│   ├── best_model.joblib
│   ├── model_comparison.csv
│   └── model_metadata.json
└── utils/
    ├── resume_parser.py
    ├── skill_analysis.py
    └── text_processing.py
```

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the models and generate artifacts:

```bash
python models/train_model.py
```

4. Launch the Streamlit app:

```bash
streamlit run app.py
```

## Dataset

- The project includes a synthetic dataset generator with 700+ samples.
- Sample format:

```text
"Python SQL Tableau Power BI Statistics" -> "Data Analyst"
```

## Expected Output

After entering skills such as `Python SQL Tableau Statistics Power BI`, the dashboard will show:

- Top 3 matching roles
- Confidence scores in a bar chart
- Missing skills for each predicted role
- Suggested skills to learn next
- Estimated salary range
- Career path recommendation
- Model comparison table

## Notes

- The synthetic dataset is suitable for academic demos, portfolio projects, and final-year CSE presentations.
- You can replace the synthetic dataset with a real CSV that contains `skills` and `job_role` columns.
