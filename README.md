# 🎓 Student Dropout Risk Prediction in Indian Government Schools

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-dropout-risk-prediction.streamlit.app/)

## 🎯 Overview & Problem Statement

Student dropout is a significant challenge in the Indian education system, particularly within government schools. Early identification of students at risk of discontinuing their education is crucial for enabling timely and effective interventions. This project aims to develop a robust machine learning system to predict student dropout risk, providing actionable insights for educators, administrators, and policymakers. By leveraging student-specific data, academic performance, socio-economic factors, and school infrastructure details, we can proactively support vulnerable students and work towards improving overall student retention.

This Proof of Concept (PoC) encompasses the entire lifecycle from data creation and preprocessing, exploratory data analysis (EDA), model development, to an interactive Streamlit dashboard for prediction and analysis.

## ✨ Key Features

- **End-to-End ML Pipeline:** Covers synthetic data generation, comprehensive preprocessing, EDA, model training, and evaluation.
- **Advanced Feature Engineering:** Creates insightful features like a socio-economic index, school support score, accessibility score, and flags for critical risk factors.
- **Multiple Predictive Models:** Implements and compares Logistic Regression, Random Forest, XGBoost, and an Ensemble model.
- **Optimized XGBoost Model:** The primary model, tuned for performance and using an optimal threshold for dropout classification.
- **Interactive Streamlit Dashboard (`app.py`):**
  - **Individual Prediction:** Assess dropout risk for a single student with detailed explanations.
  - **Batch Analysis:** Upload CSV data for bulk predictions.
  - **Model Performance:** View detailed metrics (Accuracy, Precision, Recall, F1-score) and confusion matrices from test set evaluations.
  - **Risk Tier Analytics:** Analyze student distribution across risk tiers and actual dropout rates per tier.
  - **Model Testing:** Explore model sensitivity with extreme case testing and feature impact analysis.
- **Actionable Insights:** Categorizes students into risk tiers (Very Low to Very High) and provides context for intervention.

## 🖼️ Dashboard Preview

**Individual Prediction Page:**


<img width="1773" alt="Screenshot 2025-05-28 at 2 12 22 PM" src="https://github.com/user-attachments/assets/6eb89530-07cc-4f3e-84bb-f83a3898abb2" />


<img width="1752" alt="Screenshot 2025-05-28 at 2 12 31 PM" src="https://github.com/user-attachments/assets/7ddba50d-33df-49bf-94ae-a3e49713de58" />


**Model Performance & Risk Analytics:**


<img width="1768" alt="Screenshot 2025-05-28 at 2 11 42 PM" src="https://github.com/user-attachments/assets/2b25667f-f4f0-4b85-a0cd-c4a84a933ea3" />


<img width="1774" alt="Screenshot 2025-05-28 at 2 11 52 PM" src="https://github.com/user-attachments/assets/a0a4d998-edcf-4814-8fc5-d4f00c8c935f" />


<img width="1766" alt="Screenshot 2025-05-28 at 2 12 07 PM" src="https://github.com/user-attachments/assets/1d866983-cd0f-45da-ad7e-4ddccee3e39b" />


## 🗂️ Data Journey

This project utilizes a synthetically generated dataset designed to mimic real-world complexities found in Indian government school data. The data journey is primarily covered in two notebooks: `Data_creation_and_preprocessing.ipynb` and `EDA.ipynb`.

### 1. Data Source & Generation (`Data_creation_and_preprocessing.ipynb`)

- **Inspiration:** The dataset structure is inspired by data typically available from portals like UDISE+ (Unified District Information System for Education).
- **Synthetic Data:** A synthetic dataset of **3 million student records** (`synthetic_student_data_3M.csv`) is generated to provide a large and diverse base for model training. This process simulates various student attributes.
- **Initial Raw Features Simulated:**
  - Demographics: `student_id`, `age`, `gender`, `caste_category` (renamed to `caste` later), `father_education`, `family_income`.
  - Academic: `grade`, `attendance_rate`, `grade_performance`.
  - School/Access: `midday_meal`, `free_uniforms`, `free_textbooks` (renamed with `_access` suffix), `internet_access` (renamed to `internet_access_home`), `distance_to_school`.
  - Target Variable: `dropout`.

### 2. Exploratory Data Analysis & Preprocessing (`EDA.ipynb`)

This notebook loads the synthetic raw data and performs extensive EDA and preprocessing to prepare the final dataset for modeling (`preprocessed_student_data.csv`).

- **Key Observations from Visualizations:**

* Average dropout rate is 17.6%, with higher rates among students with higher grades and ages.

* Attendance rate and grade performance shows strong negative correlation with dropout (r = X)

* Students with access to internet at home, having access to free uniforms, mid-day meals and short distances from school have lower dropout rates, hence the inverse relationship with dropout rate.

* Students with family income below 3.5 lakhs have more than 5% higher dropout rates compared to higher income groups

* Father's education has sizeable impact only after he has done higher secondary education and above

* Difference between dropout rate of students belonging to ST and SC categories is 4% higher than those belonging to OBC and General

* Key risk factors appear to be: Attendance rate, grade performance , age, grade ,caste, family income, infrastructure score, along with shorter distances from school, free uniforms and other such school facilities among others

* Distribution of key variables (e.g., dropout rates across different castes, income levels, attendance brackets).
    ![image](https://github.com/user-attachments/assets/5d42d266-ec42-4555-9c9e-b088220d27ea)

    ![image](https://github.com/user-attachments/assets/dd306282-49e0-4ef4-8dad-476f008bef14)

    ![image](https://github.com/user-attachments/assets/46684762-066f-4b55-9bbe-3c7085d135d0)

    ![image](https://github.com/user-attachments/assets/131dd155-0cdc-480a-89ea-88d46d644822)



- **Feature Engineering Highlights:**
  - **Age-Grade Mismatch:** `age_grade_diff`, `age_grade_mismatch` (binary flag).
  - **Performance Ratios:** `performance_attendance_ratio`.
  - **Risk Flags:** `low_attendance`, `low_grades`, `high_distance`.
  - **Socio-Economic Index (`socioeconomic_index`):** Calculated based on `family_income`, `father_education`, and `caste`.
  - **School Support Score (`school_support_score`):** Derived from `midday_meal_access`, `free_text_books_access`, `free_uniform_access`.
  - **Accessibility Score (`accessibility_score`):** Calculated from `distance_to_school` using exponential decay.
  - **Expected Age (`expected_age`):** Calculated based on `grade`.
  - School characteristics like `rural_urban_y`, `girl_ratio`, `female_teacher_ratio`.
- **Preprocessing Steps:**
  - Handling of missing values (imputation strategies employed).
  - Renaming columns for clarity and consistency.
  - Conversion of binary features (e.g., Yes/No to 1/0).
  - Scaling: `RobustScaler` applied to `infrastructure_score` during EDA/preprocessing stages before feeding to models.
  - The `EDA.ipynb` saves the `preprocessed_student_data.csv` which is then used by `model.ipynb`.

### 3. Final Modeling Dataset (`model.ipynb`)

- Loads `preprocessed_student_data.csv`.
- Performs One-Hot Encoding for categorical features like `gender`, `caste`, `father_education`, `family_income` using a saved `OneHotEncoder`.
- The final feature set for the XGBoost model comprises approximately 36 features.

## 🛠️ Modeling Workflow (`model.ipynb`)

1.  **Data Splitting:** The preprocessed data is split into training (80%), validation (10%), and test (10%) sets. Stratification is used to maintain class balance for the `dropout` target.
2.  **Model Training & Tuning:**
    - **Logistic Regression:** Baseline model.
    - **Random Forest:** Tuned using `RandomizedSearchCV` and threshold adjustment for optimal F1-score.
    - **XGBoost Classifier:** Primary model, optimized parameters, and class weight handling (`scale_pos_weight`) for imbalanced data.
    - **Ensemble:** A custom `DropoutPredictionPipeline` class is used to train, evaluate, and make predictions using individual models and a weighted average ensemble.
3.  **Evaluation:**
    - Metrics: Accuracy, Precision (dropout class), Recall (dropout class), F1-score (dropout class), ROC AUC.
    - Confusion matrices.
    - Feature importance analysis (gain-based for XGBoost, default for RF).
    - Risk tier analysis: Students are segmented into five risk tiers based on predicted probabilities to assess model calibration and actual dropout rates within tiers.

## 📈 Performance Highlights (Test Set)

The models were evaluated on a held-out test set. The **XGBoost (Optimized)** model is the primary focus for the Streamlit application.

| Model                     | Accuracy   | Precision (Dropout) | Recall (Dropout) | F1-score (Dropout) |
| :------------------------ | :--------- | :------------------ | :--------------- | :----------------- |
| Logistic Regression       | ~71.1%     | ~33.9%              | ~56.1%           | ~42.3%             |
| Random Forest (Optimized) | ~76.8%     | ~39.6%              | ~64.0%           | ~48.9%             |
| **XGBoost (Optimized)**   | **~78.2%** | **~41.0%**          | **~61.0%**       | **~49.1%**         |
| Ensemble (Weighted Avg)   | ~72.0%     | ~34.8%              | ~75.3%           | ~47.6%             |

_(Note: XGBoost optimal threshold for these metrics was found to be ~0.25 on the validation set)._

## 💡 Key Insights from Risk Tier Analysis (XGBoost - Test Set)

The XGBoost model effectively segments students into risk categories, demonstrating practical utility:

- **Very Low Risk (<20% prob.):** Comprises ~66% of students, with an actual test set dropout rate of ~7.5%.
- **High & Very High Risk (≥60% prob.):** Represents a smaller fraction of students (~3.4%) but shows a significantly higher average actual dropout rate (~76.9%).
  _(Detailed breakdown available on the "Risk Analytics" page of the Streamlit app)._

## ⚙️ Technology Stack

- **Python 3.9+**
- **Data Handling & Numerics:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (for preprocessing, Logistic Regression, Random Forest, metrics, model selection), XGBoost
- **Serialization:** Joblib (for saving/loading models and encoders)
- **Visualization:** Matplotlib, Seaborn (in notebooks), Plotly, Plotly Express (in Streamlit app)
- **Dashboard:** Streamlit
- **Development Environment:** Jupyter Notebook/Lab

## 🏁 Getting Started

### Prerequisites

- Python 3.9 or higher.
- Pip (Python package installer).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourGitHubUsername]/[YourRepoName].git
    cd [YourRepoName]
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```
3.  **Install dependencies:**
    _(Create `requirements.txt` using `pip freeze > requirements.txt` in your environment after installing all libraries listed in the notebooks)_
    ```bash
    pip install -r requirements.txt
    ```

### Data and Model Setup

1.  **Run the notebooks sequentially:**
    - Execute `notebooks/Data_creation_and_preprocessing.ipynb` to generate `synthetic_student_data_combined.csv` in `data/processed/`.
    - Then, run `notebooks/EDA.ipynb` to perform EDA and generate `preprocessed_student_data.csv` in `data/processed/`.
    - Finally, run `notebooks/model.ipynb` to train models and save the necessary artifacts (e.g., `xgboost_model.json`, `one_hot_encoder.joblib`, etc.) into the `models/` directory.
      _(Important: Ensure the file paths for saving/loading data and models in the notebooks are correctly set up to use these relative paths if you want portability. Your current `app.py` uses an absolute path for `BASE_MODEL_PATH` which should be changed to `models/` for a GitHub project.)_

## 🚀 Running the Project

### Jupyter Notebooks

1.  Activate your virtual environment.
2.  Launch Jupyter Lab or Notebook: `jupyter lab` or `jupyter notebook`.
3.  Navigate to the `notebooks/` directory and run the notebooks as described in "Data and Model Setup".

### Streamlit Application

1.  Ensure your virtual environment is activated and you are in the project's root directory.
2.  Confirm that the trained model files and the `one_hot_encoder.joblib` are present in the `./models/` directory. **Update `BASE_MODEL_PATH` in `app.py` to `"models/"` for relative pathing.**
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4.  Access the dashboard in your browser (usually `http://localhost:8501`).


## 🔮 Future Enhancements

- **Temporal Data Analysis:** Track students over multiple years.
- **Explainable AI (XAI):** Integrate SHAP/LIME for prediction explanations.
- **Causal Inference:** Explore causal factors beyond correlations.
- **Real-world Data Integration:** Connect with actual school databases (with permissions).
- **Automated Retraining Pipelines:** Keep the model updated.
- **Counselor Recommendation Integration:** Incorporate qualitative inputs.

## ✍️ Author / Contributors

- **Nihal Choutapelly**
- GitHub: `https://github.com/Nihal018`
- LinkedIn: `https://www.linkedin.com/in/nihal-choutapelly-9515b6229/`

---
