# Job Role Prediction using Skills: Project Documentation

## 1. Project Overview
This project is a **Data Mining & Analytics** application designed to help individuals understand their career potential based on their technical skill sets. It uses Machine Learning to classify users into the most suitable tech roles and provides actionable insights for professional growth.

### Key Features
- **Predictive Modeling**: Suggests the top 3 most relevant job roles.
- **Skill-Gap Analysis**: Identifies missing essential skills for those roles.
- **Career Path Mapping**: Visualizes the typical progression for each role.
- **Salary Estimation**: Provides average market salary ranges (LPA).
- **Resume Integration**: Supports PDF resume parsing for automatic skill extraction.

---

## 2. Dataset Information
The project uses a structured dataset specifically curated for tech roles.

- **Data Source**: Synthetic/Curated CSV containing skills mapped to roles.
- **Size**: ~720 samples across 8 distinct categories.
- **Format**: CSV (Comma Separated Values).
- **Features**:
  - `skills`: A space-separated string of technical keywords (Features/Independent variables).
  - `job_role`: The Target label (Class/Dependent variable).

### Categories (Classes)
1. Data Analyst
2. Data Scientist
3. Machine Learning Engineer
4. Business Analyst
5. Data Engineer
6. BI Developer
7. AI Engineer
8. Software Engineer

---

## 3. Data Mining Pipeline

### Step 1: Data Preprocessing
Raw skill text is often messy (special characters, case sensitivity). 
- **Normalization**: Converting to lowercase and removing punctuation.
- **Stopwords**: Standard tech connectors like "and" or "with" are filtered.
- **Keyword Extraction**: Breaking the text into discrete tokens representing specific technologies (e.g., "python", "scikit-learn").

### Step 2: Feature Engineering (TF-IDF)
We use **TF-IDF (Term Frequency-Inverse Document Frequency)** Vectorization. 
- It converts text into a numerical matrix.
- **N-grams**: We use (1, 2) n-grams, meaning it recognizes single words like "Python" and phrases like "Machine Learning" as distinct features.

### Step 3: Model Training & Evaluation
We compare three major classification algorithms:
1. **Multinomial Naive Bayes**: Efficient for text classification using probability.
2. **Logistic Regression**: A robust linear model (with 'balanced' class weights to handle distribution).
3. **Random Forest**: An ensemble method using multiple decision trees for higher accuracy and stability.

### Step 4: Model Deployment
The best-performing model (usually Logistic Regression or Random Forest in this context) is serialized using `joblib` and served via a **Streamlit** dashboard.

---

## 4. Analytics & Algorithm Depth (Q&A)

### Q1: Why use TF-IDF instead of simple counting (Bag of Words)?
**A**: TF-IDF reduces the weight of terms that appear very frequently across all roles (like "communication" or "teamwork") and increases the weight of role-specific technical terms (like "spark" or "pytorch"), leading to more accurate classification.

### Q2: What is the role of the 'Random Forest' algorithm in this project?
**A**: Random Forest is an **ensemble algorithm** that creates multiple decision trees during training. It is excellent at capturing non-linear relationships between skills. For example, the combination of "Python" + "SQL" might point to Data Analyst, but "Python" + "TensorFlow" points to AI Engineer.

### Q3: How do we handle different career paths in Analytics?
**A**: The project uses a **Rule-Based Insight Engine** (`utils/skill_analysis.py`). Once the Machine Learning model predicts the role, the engine performs a set-difference operation between the "Required Skills" for that role and the "Detected Skills" from the user to generate the "Skill Gap".

### Q4: How is accuracy measured during the 'Data Mining' phase?
**A**: We use three primary metrics:
- **Accuracy**: Overall correct predictions.
- **Precision**: How many predicted "Data Scientists" were actually Data Scientists.
- **Recall**: Proportion of actual "Data Scientists" the model successfully found.
We use a **Stratified Train-Test Split** (80/20) to ensure every role is represented fairly in both training and testing.

### Q5: What makes this a 'Data Mining' project?
**A**: Data Mining involves discovering patterns in large datasets. This project identifies the "pattern of skills" that define a professional role. It transforms unstructured text (resumes/skill lists) into structured predictions and career analytics.

---

## 5. Technology Stack
- **Languages**: Python
- **ML Libraries**: Scikit-Learn, Pandas, NumPy, Joblib
- **Frontend**: Streamlit
- **Parsing**: PyPDF (for resumes)
- **Deployment**: Localhost (Port 8501)
