import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
import warnings
import os
import re

warnings.filterwarnings('ignore')

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling (Your existing CSS block)
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold; }
    .sub-header { font-size: 1.5rem; color: #333; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #1f77b4; padding-bottom: 0.3rem; }
    .risk-card { padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0; text-align: center; font-weight: bold; font-size: 1.1rem; border: 1px solid #ddd; }
    .very-low { background-color: #d4edda; color: #155724; border-left: 5px solid #155724;}
    .low { background-color: #d1ecf1; color: #0c5460; border-left: 5px solid #0c5460;}
    .medium { background-color: #fff3cd; color: #856404; border-left: 5px solid #856404;}
    .high { background-color: #f8d7da; color: #721c24; border-left: 5px solid #721c24;}
    .very-high { background-color: #f5c6cb; color: #491217; border-left: 5px solid #491217; font-weight: bold;}
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom:1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .stButton>button { background-color: #1f77b4; color: white; padding: 0.5rem 1rem; border-radius: 5px; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #155a8a; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    .prediction-outcome { font-size: 1.4rem; font-weight: bold; margin-bottom: 0.5rem; } /* New style for outcome */
    .dropout { color: #721c24; } /* Reddish for dropout */
    .stay { color: #155724; } /* Greenish for stay */
</style>
""", unsafe_allow_html=True)

# --- Model & Preprocessor Paths ---
BASE_MODEL_PATH = "/Users/macbookpro/Desktop/POC1/models/"
XGB_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "xgboost_model.json")
RF_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "rf_model.joblib")
LOGISTIC_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "logistic_model.joblib")
OHE_PATH = os.path.join(BASE_MODEL_PATH, "one_hot_encoder.joblib")

# --- Feature Name Definitions ---
NUMERICAL_FEATURE_NAMES = [
    'grade', 'age', 'attendance_rate', 'grade_performance', 'midday_meal_access',
    'free_text_books_access', 'free_uniform_access', 'internet_access_home',
    'distance_to_school', 'rural_urban_y', 'school_category', 'school_type',
    'avg_instr_days', 'medical_checkups', 'student_teacher_ratio', 'girl_ratio',
    'female_teacher_ratio', 'infrastructure_score', 'socioeconomic_index',
    'school_support_score', 'accessibility_score', 'expected_age'
]
CATEGORICAL_FEATURE_NAMES_FOR_OHE = [
    'gender', 'caste', 'father_education', 'family_income']
EXPECTED_COLUMNS_AFTER_OHE_RAW = [
    'grade', 'age', 'attendance_rate', 'grade_performance', 'midday_meal_access',
    'free_text_books_access', 'free_uniform_access', 'internet_access_home',
    'distance_to_school', 'rural_urban_y', 'school_category', 'school_type',
    'avg_instr_days', 'medical_checkups', 'student_teacher_ratio', 'girl_ratio',
    'female_teacher_ratio', 'infrastructure_score', 'socioeconomic_index',
    'school_support_score', 'accessibility_score', 'expected_age',
    'gender_Female', 'gender_Male', 'caste_General', 'caste_OBC', 'caste_SC', 'caste_ST',
    'father_education_Graduate', 'father_education_HigherSecondary',
    'father_education_PostGraduate', 'father_education_Primary', 'father_education_Secondary',
    'family_income_< ₹2 Lakhs', 'family_income_> ₹5 Lakhs',
    'family_income_₹2 - ₹3.5 Lakhs', 'family_income_₹3.5 - ₹5 Lakhs'
]

MODEL_EXPECTS_THESE_EXACT_NAMES = ['grade', 'age', 'attendance_rate', 'grade_performance', 'midday_meal_access', 'free_text_books_access', 'free_uniform_access', 'internet_access_home', 'distance_to_school', 'rural_urban_y', 'school_category', 'school_type', 'avg_instr_days', 'medical_checkups', 'student_teacher_ratio', 'girl_ratio', 'female_teacher_ratio', 'infrastructure_score', 'socioeconomic_index', 'school_support_score',
                                   'accessibility_score', 'expected_age', 'gender_Female', 'gender_Male', 'caste_General', 'caste_OBC', 'caste_SC', 'caste_ST', 'father_education_Graduate', 'father_education_HigherSecondary', 'father_education_PostGraduate', 'father_education_Primary', 'father_education_Secondary', 'family_income__lt___2_Lakhs', 'family_income__gt___5_Lakhs', 'family_income__2____3.5_Lakhs', 'family_income__3.5____5_Lakhs']
CATEGORICAL_FEATURES_MAP_FOR_FORM = {
    'gender': ['Female', 'Male'], 'caste': ['General', 'OBC', 'SC', 'ST'],
    'father_education': ['Primary', 'Secondary', 'HigherSecondary', 'Graduate', 'PostGraduate'],
    'family_income': ['< ₹2 Lakhs', '₹2 - ₹3.5 Lakhs', '₹3.5 - ₹5 Lakhs', '> ₹5 Lakhs']
}
RAW_INPUT_FEATURES_FORM = NUMERICAL_FEATURE_NAMES + \
    CATEGORICAL_FEATURE_NAMES_FOR_OHE

# XGBoost Optimal Threshold (from notebook validation analysis, e.g., cell 24)
XGB_OPTIMAL_THRESHOLD = 0.25


@st.cache_resource
def load_ohe_encoder_cached(path):
    try:
        encoder = joblib.load(path)
        st.session_state.ohe_loaded = True
        if hasattr(encoder, 'feature_names_in_'):
            st.sidebar.info(
                f"OHE loaded. Fitted on: {list(encoder.feature_names_in_)}")
        return encoder
    except Exception as e:
        st.sidebar.error(f"OHE load error: {e}")
        st.session_state.ohe_loaded = False
        return None


@st.cache_resource
def load_all_models_cached():
    models_dict = {"xgb": None, "rf": None, "logistic": None}
    try:
        xgb_model_loaded = xgb.Booster()
        xgb_model_loaded.load_model(XGB_MODEL_PATH)
        models_dict["xgb"] = xgb_model_loaded
        st.session_state.xgb_loaded = True
    except Exception as e:
        st.sidebar.error(f"XGB load error: {e}")
        st.session_state.xgb_loaded = False
    try:
        models_dict["rf"] = joblib.load(RF_MODEL_PATH)
        st.session_state.rf_loaded = True
    except Exception as e:
        st.sidebar.warning(f"RF load error: {e}")
        st.session_state.rf_loaded = False
    try:
        models_dict["logistic"] = joblib.load(LOGISTIC_MODEL_PATH)
        st.session_state.logistic_loaded = True
    except Exception as e:
        st.sidebar.warning(f"LR load error: {e}")
        st.session_state.logistic_loaded = False
    return models_dict


def create_socioeconomic_index(df):
    income_map = {'< ₹2 Lakhs': 1, '₹2 - ₹3.5 Lakhs': 2,
                  '₹3.5 - ₹5 Lakhs': 4, '> ₹5 Lakhs': 6}
    education_map = {'None': 1, 'Primary': 2, 'Secondary': 3,
                     'HigherSecondary': 4, 'Graduate': 5, 'PostGraduate': 6}
    caste_map = {'ST': 1, 'SC': 2, 'OBC': 3, 'General': 4}
    min_income_score, max_income_score = 1, 6
    min_education_score, max_education_score = 1, 6
    min_caste_score, max_caste_score = 1, 4
    df['income_score'] = df['family_income'].map(
        income_map).fillna(min_income_score)
    df['education_score'] = df['father_education'].map(
        education_map).fillna(min_education_score)
    df['caste_score'] = df['caste'].map(caste_map).fillna(min_caste_score)
    df['income_score_norm'] = (df['income_score'] - min_income_score) / (
        max_income_score - min_income_score) if (max_income_score - min_income_score) != 0 else 0
    df['education_score_norm'] = (df['education_score'] - min_education_score) / (
        max_education_score - min_education_score) if (max_education_score - min_education_score) != 0 else 0
    df['caste_score_norm'] = (df['caste_score'] - min_caste_score) / (
        max_caste_score - min_caste_score) if (max_caste_score - min_caste_score) != 0 else 0
    df['socioeconomic_index'] = (0.5 * df['income_score_norm'] + 0.15 *
                                 df['education_score_norm'] + 0.35 * df['caste_score_norm']) * 100
    df.drop(['income_score', 'education_score', 'caste_score', 'income_score_norm',
            'education_score_norm', 'caste_score_norm'], axis=1, inplace=True, errors='ignore')
    return df


def create_school_support_score(df):
    support_columns = ['midday_meal_access',
                       'free_text_books_access', 'free_uniform_access']
    for col in support_columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['school_support_score'] = df[support_columns].sum(axis=1)
    df['school_support_score'] = (
        df['school_support_score'] / len(support_columns)) * 100
    return df


def create_accessibility_score(df):
    decay_rate = 0.5
    if 'distance_to_school' not in df.columns:
        df['distance_to_school'] = 5
    df['distance_to_school'] = pd.to_numeric(
        df['distance_to_school'], errors='coerce').fillna(5)
    df['accessibility_score'] = 100 * \
        np.exp(-decay_rate * df['distance_to_school'])
    return df


def preprocess_data_with_saved_ohe(df_input, ohe_encoder_obj, categorical_cols_to_ohe, expected_cols_after_ohe_raw_order, numerical_cols):
    processed_df = df_input.copy()
    actual_categorical_cols_present = [
        col for col in categorical_cols_to_ohe if col in processed_df.columns]
    existing_numerical_cols = [
        col for col in numerical_cols if col in processed_df.columns]
    if len(existing_numerical_cols) < len(numerical_cols):
        st.warning(
            f"Missing numerical input columns. Expected: {numerical_cols}, Got: {existing_numerical_cols}")
    df_numerical_part = processed_df[existing_numerical_cols].copy()
    encoded_df_part = pd.DataFrame(index=processed_df.index)
    if actual_categorical_cols_present:
        df_categorical_to_encode = processed_df[actual_categorical_cols_present].copy(
        )
        if ohe_encoder_obj and st.session_state.get('ohe_loaded', False):
            try:
                encoded_data_transformed = ohe_encoder_obj.transform(
                    df_categorical_to_encode)
                ohe_generated_feature_names = ohe_encoder_obj.get_feature_names_out(
                    actual_categorical_cols_present)
                encoded_df_part = pd.DataFrame(encoded_data_transformed.toarray() if hasattr(
                    encoded_data_transformed, "toarray") else encoded_data_transformed, columns=ohe_generated_feature_names, index=processed_df.index)
            except ValueError as ve:
                st.error(f"OHE ValueError: {ve}")
                return pd.DataFrame()
            except Exception as e:
                st.error(f"OHE Generic error: {e}")
                return pd.DataFrame()
        else:
            st.warning("OHE not loaded.")
            return pd.DataFrame()
    elif len(categorical_cols_to_ohe) > 0:
        st.warning(
            f"Intended categorical columns ({categorical_cols_to_ohe}) not in input.")
    processed_df_combined = pd.concat(
        [df_numerical_part, encoded_df_part], axis=1)
    final_df_raw_ohe_column_names = pd.DataFrame(
        columns=expected_cols_after_ohe_raw_order)
    for col in expected_cols_after_ohe_raw_order:
        if col in processed_df_combined.columns:
            final_df_raw_ohe_column_names[col] = processed_df_combined[col]
        else:
            final_df_raw_ohe_column_names[col] = 0
    if 'student_teacher_ratio' in final_df_raw_ohe_column_names.columns:
        median_val = df_input['student_teacher_ratio'].median(
            skipna=True) if 'student_teacher_ratio' in df_input else 25.0
        if pd.isna(median_val):
            median_val = 25.0
        final_df_raw_ohe_column_names['student_teacher_ratio'] = final_df_raw_ohe_column_names['student_teacher_ratio'].replace([
                                                                                                                                np.inf, -np.inf], np.nan).fillna(median_val)
    return final_df_raw_ohe_column_names


def get_risk_tier_details(probability):
    if probability is None or not isinstance(probability, (float, np.float32, np.float64)) or np.isnan(probability):
        return "N/A", "error-class"
    if probability < 0.2:
        return "Very Low", "very-low"
    elif probability < 0.4:
        return "Low", "low"
    elif probability < 0.6:
        return "Medium", "medium"
    elif probability < 0.8:
        return "High", "high"
    else:
        return "Very High", "very-high"


def create_risk_gauge_chart(probability, main_model_name="Model"):
    prob_for_chart = 0.0
    if probability is not None and isinstance(probability, (float, np.float32, np.float64)) and not np.isnan(probability):
        prob_for_chart = float(probability)
    risk_tier_label, _ = get_risk_tier_details(prob_for_chart)
    chart_title = f"{main_model_name} Dropout Risk: {risk_tier_label}"
    if prob_for_chart == 0.0 and risk_tier_label == "N/A":
        chart_title = f"{main_model_name} Dropout Risk: N/A"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=prob_for_chart * 100, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': chart_title, 'font': {'size': 18}},
        delta={'reference': XGB_OPTIMAL_THRESHOLD * 100, 'increasing': {'color': "Red"},
               # Delta reference to threshold
               'decreasing': {'color': "Green"}},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
               'bar': {'color': "darkblue"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
               'steps': [{'range': [0, 20], 'color': '#d4edda'}, {'range': [20, 40], 'color': '#d1ecf1'},
                         {'range': [40, 60], 'color': '#fff3cd'}, {
                             'range': [60, 80], 'color': '#f8d7da'},
                         {'range': [80, 100], 'color': '#f5c6cb'}],
               # Threshold line
               'threshold': {'line': {'color': "firebrick", 'width': 3}, 'thickness': 0.9, 'value': XGB_OPTIMAL_THRESHOLD * 100}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def get_intervention_recommendations_detailed(risk_tier_label):
    recommendations = {"Very High": ["🚨 **Immediate & Intensive Intervention Required**", "• Assign a dedicated counselor/mentor for daily check-ins.", "• Urgent parent/guardian meeting to develop a joint support plan.", "• Explore immediate financial assistance/scholarships if socio-economic factors are key.", "• Implement a highly personalized and flexible learning plan.", "• Daily attendance monitoring and immediate follow-up on absences.", "• Consider alternative schooling options if mainstream is unsuitable."], "High": ["⚠️ **Proactive & Targeted Support Needed**", "• Weekly counseling sessions and regular academic tutoring.", "• Small group interventions focusing on specific skill gaps.", "• Connect family with community support services if applicable.", "• Bi-weekly attendance and performance reviews with teacher/mentor.", "• Provide necessary learning materials and a conducive study environment if possible."], "Medium": ["📊 **Consistent Monitoring & Support**",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               "• Monthly check-ins with teachers and pastoral care staff.", "• Enrollment in peer support groups or study skills workshops.", "• Regular monitoring of academic performance trends and attendance.", "• Offer encouragement and positive reinforcement for effort and improvements.", "• Ensure access to all available school support facilities (library, meals etc.)."], "Low": ["✅ **Preventive Measures & Engagement**", "• Quarterly academic progress reviews and goal setting.", "• Participation in motivational programs and extracurricular activities.", "• Career guidance and aspiration-building sessions.", "• Maintain open communication channels with student and family."], "Very Low": ["🌟 **Maintain Positive Trajectory & Enrich**", "• Continue standard educational support and encouragement.", "• Offer opportunities for peer mentoring or leadership roles.", "• Provide access to enrichment programs and advanced learning opportunities.", "• Celebrate successes and maintain a supportive school environment."]}
    return recommendations.get(risk_tier_label, ["No specific recommendations available."])

# --- Streamlit Page Functions ---


def individual_prediction_page(current_models, current_ohe):
    st.markdown('<h2 class="sub-header">🎓 Individual Student Risk Assessment</h2>',
                unsafe_allow_html=True)
    input_data = {}
    form_cols = st.columns(2)
    with form_cols[0]:
        st.subheader("📊 Student Demographics & Academics")
        input_data['age'] = st.number_input(
            "Age", min_value=5, max_value=25, value=12, step=1)
        input_data['grade'] = st.selectbox(
            "Grade", list(range(1, 13)), index=6)
        input_data['gender'] = st.selectbox(
            "Gender", CATEGORICAL_FEATURES_MAP_FOR_FORM['gender'])
        input_data['caste'] = st.selectbox(
            "Caste Category", CATEGORICAL_FEATURES_MAP_FOR_FORM['caste'])
        input_data['attendance_rate'] = st.slider(
            "Attendance Rate (%)", 0.0, 100.0, 85.0, 0.1)
        input_data['grade_performance'] = st.slider(
            "Grade Performance (%)", 0.0, 100.0, 75.0, 0.1)
        st.subheader("👨‍👩‍👧‍👦 Family Background")
        input_data['father_education'] = st.selectbox(
            "Father's Education", CATEGORICAL_FEATURES_MAP_FOR_FORM['father_education'])
        input_data['family_income'] = st.selectbox(
            "Family Income (Annual)", CATEGORICAL_FEATURES_MAP_FOR_FORM['family_income'])
    with form_cols[1]:
        st.subheader("🏫 School & Facilities")
        input_data['distance_to_school'] = st.number_input(
            "Distance to School (km)", 0.0, 50.0, 2.0, 0.1)
        selected_area_type = st.selectbox(
            "Area Type", ["Urban", "Rural"], index=0)
        input_data['rural_urban_y'] = 1 if selected_area_type == "Rural" else 0
        input_data['school_category'] = st.number_input(
            "School Category Code", min_value=1, max_value=10, value=3)
        input_data['school_type'] = st.number_input(
            "School Type Code", min_value=1, max_value=10, value=3)
        st.markdown("**School Support (Tick if Yes):**")
        input_data['midday_meal_access'] = 1 if st.checkbox(
            "Midday Meal", value=True) else 0
        input_data['free_text_books_access'] = 1 if st.checkbox(
            "Free Textbooks", value=True) else 0
        input_data['free_uniform_access'] = 1 if st.checkbox(
            "Free Uniforms", value=True) else 0  # Corrected key
        input_data['internet_access_home'] = 1 if st.checkbox(
            "Internet at Home", value=False) else 0
        input_data['medical_checkups'] = 1 if st.checkbox(
            "Medical Checkups at School", value=True) else 0
    with st.expander("⚙️ Other School & Student Characteristics"):
        adv_cols = st.columns(2)
        with adv_cols[0]:
            input_data['avg_instr_days'] = st.number_input(
                "Avg. Instructional Days", 100, 300, 200)
            input_data['student_teacher_ratio'] = st.number_input(
                "Student-Teacher Ratio", 5.0, 100.0, 25.0, step=0.1)
        with adv_cols[1]:
            input_data['girl_ratio'] = st.slider(
                "Girl Ratio in School", 0.0, 1.0, 0.48, 0.01)
            input_data['female_teacher_ratio'] = st.slider(
                "Female Teacher Ratio", 0.0, 1.0, 0.50, 0.01)
    input_data['expected_age'] = input_data['grade'] + 6

    if st.button("🔮 Predict Dropout Risk", type="primary", use_container_width=True):
        if not any(m is not None for m in current_models.values()):
            st.error("Models not loaded.")
            return
        if not current_ohe or not st.session_state.get('ohe_loaded', False):
            st.error("OHE not loaded.")
            return

        input_df_raw = pd.DataFrame([input_data])
        input_df = input_df_raw.copy()  # Work on a copy for calculations

        input_df['infrastructure_score'] = 0.1
        try:
            input_df = create_socioeconomic_index(input_df)
            input_df = create_school_support_score(input_df)
            input_df = create_accessibility_score(input_df)
        except Exception as e_calc:
            st.error(f"Feature calculation error: {e_calc}")
            return

        xgb_prob, rf_prob, lr_prob = None, None, None
        processed_df_raw_ohe_names = None
        df_for_models = None
        predicted_outcome_text = "Prediction Error"
        predicted_outcome_class = "error-class"

        try:
            processed_df_raw_ohe_names = preprocess_data_with_saved_ohe(
                input_df, current_ohe, CATEGORICAL_FEATURE_NAMES_FOR_OHE,
                EXPECTED_COLUMNS_AFTER_OHE_RAW, NUMERICAL_FEATURE_NAMES
            )
            if processed_df_raw_ohe_names.empty:
                st.error("Preprocessing failed.")
                return

            df_for_models = processed_df_raw_ohe_names.copy()
            if len(df_for_models.columns) == len(MODEL_EXPECTS_THESE_EXACT_NAMES):
                df_for_models.columns = MODEL_EXPECTS_THESE_EXACT_NAMES
            else:
                st.error(f"Column count mismatch pre-DMatrix.")
                return

            primary_model_for_gauge = "N/A"
            display_prob_for_gauge = 0.0
            risk_tier_label, risk_class = "N/A", "error-class"

            if current_models.get("xgb"):
                try:
                    dmatrix = xgb.DMatrix(
                        df_for_models, feature_names=list(df_for_models.columns))
                    xgb_pred_output = current_models["xgb"].predict(dmatrix)
                    if xgb_pred_output is not None and len(xgb_pred_output) > 0:
                        xgb_prob = float(xgb_pred_output[0])
                        if np.isnan(xgb_prob):
                            xgb_prob = None
                except Exception as e:
                    xgb_prob = None
                    st.warning(f"XGB Error: {e}")

            if xgb_prob is not None:
                display_prob_for_gauge, primary_model_for_gauge = xgb_prob, "XGBoost"
                risk_tier_label, risk_class = get_risk_tier_details(xgb_prob)
                if xgb_prob > XGB_OPTIMAL_THRESHOLD:
                    predicted_outcome_text = "Predicted to Dropout"
                    predicted_outcome_class = "dropout"
                else:
                    predicted_outcome_text = "Predicted to Stay in School"
                    predicted_outcome_class = "stay"
            else:  # Fallback if XGBoost failed
                # Try RF
                if current_models.get("rf"):
                    try:
                        rf_prob = float(current_models["rf"].predict_proba(
                            df_for_models.values)[0][1])
                        if not np.isnan(rf_prob):
                            display_prob_for_gauge, primary_model_for_gauge = rf_prob, "Random Forest"
                            risk_tier_label, risk_class = get_risk_tier_details(
                                rf_prob)
                            # Note: RF might have a different optimal threshold for binary prediction
                            # For simplicity, we're not showing a binary prediction for fallback models here
                            predicted_outcome_text = f"Primary Model (XGBoost) Failed. Showing RF."
                    except Exception as e:
                        rf_prob = None
                        st.warning(f"RF Error: {e}")
                # Try LR if RF also failed or not available
                # if still not set
                if primary_model_for_gauge == "N/A" and current_models.get("logistic"):
                    try:
                        lr_prob = float(current_models["logistic"].predict_proba(
                            df_for_models.values)[0][1])
                        if not np.isnan(lr_prob):
                            display_prob_for_gauge, primary_model_for_gauge = lr_prob, "Logistic Regression"
                            risk_tier_label, risk_class = get_risk_tier_details(
                                lr_prob)
                            predicted_outcome_text = f"Primary Model (XGBoost) Failed. Showing LR."
                    except Exception as e:
                        lr_prob = None
                        st.warning(f"LR Error: {e}")
                if primary_model_for_gauge == "N/A":
                    st.error("All models failed or unavailable.")
                    predicted_outcome_text = "All models failed."

            st.markdown("---")
            results_col1, results_col2 = st.columns([2, 3])
            with results_col1:
                if xgb_prob is not None:  # Only show XGBoost specific details if it succeeded
                    st.markdown(
                        f"<h2 class='prediction-outcome {predicted_outcome_class}'>{predicted_outcome_text}</h2>", unsafe_allow_html=True)
                    st.markdown(
                        f"**Dropout Probability (XGBoost):** `{xgb_prob*100:.1f}%`")
                    st.markdown(
                        f"**Decision Threshold Used:** `{XGB_OPTIMAL_THRESHOLD*100:.1f}%`")
                elif primary_model_for_gauge != "N/A":  # Show if a fallback was used for gauge
                    st.markdown(
                        f"<p class='prediction-outcome {predicted_outcome_class}'>{predicted_outcome_text}</p>", unsafe_allow_html=True)

                # Title reflects gauge source
                st.markdown(f"#### {primary_model_for_gauge} Risk Profile")
                st.plotly_chart(create_risk_gauge_chart(
                    display_prob_for_gauge, primary_model_for_gauge), use_container_width=True)

                prob_display_for_card = display_prob_for_gauge * 100
                current_risk_tier_for_card, current_risk_class_for_card = get_risk_tier_details(
                    display_prob_for_gauge)
                st.markdown(
                    f'<div class="risk-card {current_risk_class_for_card}">Risk Level: {current_risk_tier_for_card} ({prob_display_for_card:.1f}%)</div>', unsafe_allow_html=True)

                st.markdown("##### Risk Tier Context")
                risk_tier_info_sidebar = {
                    "Very Low": "< 20% (Val. Dropout: ~7.5%)", "Low": "20-40% (Val. Dropout: ~27.6%)",
                    "Medium": "40-60% (Val. Dropout: ~48.1%)", "High": "60-80% (Val. Dropout: ~67.9%)",
                    "Very High": "≥ 80% (Val. Dropout: ~85.9%)"}
                for tier, desc in risk_tier_info_sidebar.items():
                    if tier == current_risk_tier_for_card:
                        st.markdown(
                            f"**- {tier}: {desc} (_This Student's Tier_)**")
                    else:
                        st.markdown(f"- {tier}: {desc}")
            with results_col2:
                st.markdown("#### Other Model Probabilities & Interventions")
                if xgb_prob is not None and primary_model_for_gauge != "XGBoost":  # If XGB was calc but not used for gauge
                    st.metric("XGBoost Prob.", f"{xgb_prob:.2%}")
                if rf_prob is not None:
                    st.metric("Random Forest Prob.", f"{rf_prob:.2%}")
                if lr_prob is not None:
                    st.metric("Logistic Reg. Prob.", f"{lr_prob:.2%}")

                st.markdown("##### Recommended Interventions:")
                recommendations = get_intervention_recommendations_detailed(
                    current_risk_tier_for_card)  # Use tier from displayed prob
                for rec in recommendations:
                    st.markdown(rec)
            with st.expander("🔍 Debug Info"):
                st.write(
                    "Input Data (Raw, before calculated scores):", input_df_raw)
                st.write("Input Data + Calculated Scores (before OHE):", input_df)
                if processed_df_raw_ohe_names is not None:
                    st.write("Data After OHE (Raw OHE Names):",
                             processed_df_raw_ohe_names)
                st.write("Data for Models (Final Names):",
                         df_for_models if 'df_for_models' in locals() else "Not generated")
        except Exception as e:
            st.error(f"Overall Prediction Error: {e}")


# --- Batch Analysis Page, Model Performance, Risk Analytics, Model Testing ---
# (These functions should be copied from your latest app.py as they were mostly complete)
def batch_analysis_page(current_models, current_ohe):
    st.markdown('<h2 class="sub-header">📊 Batch Student Analysis</h2>',
                unsafe_allow_html=True)
    st.write("Upload CSV. Columns should match individual prediction form (base features). Engineered scores will be calculated.")
    base_features_for_batch_template = ['grade', 'age', 'gender', 'caste', 'father_education', 'family_income', 'attendance_rate', 'grade_performance', 'midday_meal_access', 'free_text_books_access', 'free_uniform_access',
                                        'internet_access_home', 'distance_to_school', 'rural_urban_y', 'school_category', 'school_type', 'avg_instr_days', 'medical_checkups', 'student_teacher_ratio', 'girl_ratio', 'female_teacher_ratio']
    with st.expander("📋 Expected CSV Format & Sample Download (Base Features)"):
        sample_dict_batch = {'student_id': ['S001', 'S002']}
        example_vals_batch = {
            'grade': [7, 10], 'age': [12, 15], 'gender': ['Male', 'Female'], 'caste': ['OBC', 'SC'],
            'father_education': ['Secondary', 'Primary'], 'family_income': ['₹2 - ₹3.5 Lakhs', '< ₹2 Lakhs'],
            'attendance_rate': [85.0, 60.0], 'grade_performance': [75.0, 50.0],
            'midday_meal_access': [1, 0], 'free_text_books_access': [1, 1], 'free_uniform_access': [1, 0],
            'internet_access_home': [0, 1], 'distance_to_school': [2.5, 8.0], 'rural_urban_y': [1, 0],
            'school_category': [1, 2], 'school_type': [3, 1], 'avg_instr_days': [200, 180],
            'medical_checkups': [1, 0], 'student_teacher_ratio': [25.0, 40.0], 'girl_ratio': [0.48, 0.52],
            'female_teacher_ratio': [0.5, 0.3]
        }
        for feature in base_features_for_batch_template:
            sample_dict_batch[feature] = example_vals_batch.get(
                feature, [np.nan, np.nan])
        sample_df_batch_display = pd.DataFrame(sample_dict_batch)
        st.dataframe(sample_df_batch_display.head())
        csv_template_batch = sample_df_batch_display.to_csv(
            index=False).encode('utf-8')
        st.download_button("⬇️ Download Batch CSV Template (Base Features)",
                           csv_template_batch, "batch_base_features_template.csv", "text/csv")

    uploaded_file = st.file_uploader(
        "Choose CSV for Batch Analysis", type="csv", key="batch_upload_main_key_v2")
    if uploaded_file:
        try:
            batch_df_original = pd.read_csv(uploaded_file)
            st.success(
                f"File '{uploaded_file.name}' ({len(batch_df_original)} rows) uploaded.")
            st.dataframe(batch_df_original.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
            if not current_models or not current_models.get("xgb"):
                st.error("XGB model not loaded.")
                return
            if not current_ohe or not st.session_state.get('ohe_loaded', False):
                st.error("OHE not loaded.")
                return
            with st.spinner("Processing batch..."):
                try:
                    batch_df_with_calc_features = batch_df_original.copy()
                    batch_df_with_calc_features['infrastructure_score'] = 0.1
                    batch_df_with_calc_features['expected_age'] = batch_df_with_calc_features['grade'] + 6
                    batch_df_with_calc_features = create_socioeconomic_index(
                        batch_df_with_calc_features.copy())
                    batch_df_with_calc_features = create_school_support_score(
                        batch_df_with_calc_features.copy())
                    batch_df_with_calc_features = create_accessibility_score(
                        batch_df_with_calc_features.copy())

                    processed_batch_raw_ohe_names = preprocess_data_with_saved_ohe(
                        batch_df_with_calc_features, current_ohe, CATEGORICAL_FEATURE_NAMES_FOR_OHE, EXPECTED_COLUMNS_AFTER_OHE_RAW, NUMERICAL_FEATURE_NAMES)
                    if processed_batch_raw_ohe_names.empty:
                        st.error("Batch preprocessing failed.")
                        return

                    batch_for_dmatrix = processed_batch_raw_ohe_names.copy()
                    if len(batch_for_dmatrix.columns) == len(MODEL_EXPECTS_THESE_EXACT_NAMES):
                        batch_for_dmatrix.columns = MODEL_EXPECTS_THESE_EXACT_NAMES
                    else:
                        st.error(
                            f"Batch column count mismatch. Got {len(batch_for_dmatrix.columns)}, Expected {len(MODEL_EXPECTS_THESE_EXACT_NAMES)}")
                        return

                    dmatrix_batch = xgb.DMatrix(
                        batch_for_dmatrix, feature_names=list(batch_for_dmatrix.columns))
                    xgb_probs_batch = current_models["xgb"].predict(
                        dmatrix_batch)
                    results_df = batch_df_original.copy()
                    results_df['dropout_probability'] = xgb_probs_batch
                    results_df['risk_tier'] = [
                        get_risk_tier_details(p)[0] for p in xgb_probs_batch]
                    st.success("✅ Batch analysis complete!")
                    col1, col2, col3, col4 = st.columns(4)  # Summary metrics
                    # ... (summary metrics display as before)
                    vcol1, vcol2 = st.columns(2)  # Visualizations
                    # ... (visualizations as before)
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df.sort_values(
                        'dropout_probability', ascending=False))
                    csv_result_dl = results_df.to_csv(
                        index=False).encode('utf-8')
                    st.download_button("📥 Download Results CSV", csv_result_dl,
                                       f"batch_dropout_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)

                except Exception as e:
                    st.error(f"Batch prediction error: {e}")


def model_performance_page():
    st.markdown('<h2 class="sub-header">📈 Model Performance Comparison (Test Set)</h2>',
                unsafe_allow_html=True)
    lr_test_metrics = {'acc': 0.7108,
                       'prec': 0.3391, 'rec': 0.5612, 'f1': 0.4231}
    rf_test_metrics = {'acc': 0.7679,
                       'prec': 0.3960, 'rec': 0.6399, 'f1': 0.4890}
    xgb_test_metrics = {'acc': 0.7821, 'prec': 0.4102,
                        'rec': 0.6103, 'f1': 0.4905}  # Using test data
    ens_test_metrics = {'acc': 0.7199,
                        'prec': 0.3481, 'rec': 0.7530, 'f1': 0.4761}
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest (Opt)', 'XGBoost (Opt)', 'Ensemble (Weighted)'],
        'Accuracy (%)': [m['acc']*100 for m in [lr_test_metrics, rf_test_metrics, xgb_test_metrics, ens_test_metrics]],
        'Precision (Dropout %)': [m['prec']*100 for m in [lr_test_metrics, rf_test_metrics, xgb_test_metrics, ens_test_metrics]],
        'Recall (Dropout %)': [m['rec']*100 for m in [lr_test_metrics, rf_test_metrics, xgb_test_metrics, ens_test_metrics]],
        'F1-Score (Dropout %)': [m['f1']*100 for m in [lr_test_metrics, rf_test_metrics, xgb_test_metrics, ens_test_metrics]]
    }
    df_performance = pd.DataFrame(performance_data)
    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Metrics (Test Set)")
        st.dataframe(df_performance.set_index('Model').style.format("{:.1f}"))
        st.caption("Dropout class metrics.")
        st.subheader("XGBoost Confusion Matrix (Test Set)")
        conf_matrix_xgb_test = pd.DataFrame({'Predicted No Dropout': [19958, 2054], 'Predicted Dropout': [
                                            4796, 3192]}, index=['Actual No Dropout', 'Actual Dropout'])
        st.table(conf_matrix_xgb_test)
    with col2:
        st.subheader("Comparison Chart (Test Set)")
        df_perf_melted = df_performance.melt(
            id_vars='Model', var_name='Metric', value_name='Score')
        fig = px.bar(df_perf_melted, x='Metric', y='Score',
                     color='Model', barmode='group', text_auto='.1f')
        fig.update_layout(yaxis_title="Score (%)", height=500)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("🎯 XGBoost Feature Importance (Training Gain)")
    fi_feature_5_cleaned = MODEL_EXPECTS_THESE_EXACT_NAMES[EXPECTED_COLUMNS_AFTER_OHE_RAW.index('family_income_< ₹2 Lakhs')] if 'family_income_< ₹2 Lakhs' in EXPECTED_COLUMNS_AFTER_OHE_RAW and EXPECTED_COLUMNS_AFTER_OHE_RAW.index(
        'family_income_< ₹2 Lakhs') < len(MODEL_EXPECTS_THESE_EXACT_NAMES) else 'family_income_lt_2_Lakhs_cleaned'
    xgb_fi_data_gain = {'Feature': ['grade', 'expected_age', 'grade_performance', 'attendance_rate', fi_feature_5_cleaned, 'school_support_score', 'socioeconomic_index', 'female_teacher_ratio', 'accessibility_score', 'gender_Male',
                                    'distance_to_school', 'school_category', 'caste_ST', 'student_teacher_ratio', 'father_education_Primary'], 'Gain': [802.82, 352.48, 43.05, 31.72, 25.62, 23.49, 21.26, 18.70, 17.57, 16.98, 15.99, 13.19, 12.82, 11.91, 11.16]}
    df_xgb_fi_gain = pd.DataFrame(xgb_fi_data_gain).sort_values(
        by="Gain", ascending=False)
    fi_cols_disp = st.columns([1, 2])
    with fi_cols_disp[0]:
        st.dataframe(df_xgb_fi_gain.style.format({"Gain": "{:.2f}"}))
    with fi_cols_disp[1]:
        fig_fi_disp = px.bar(df_xgb_fi_gain.head(10).sort_values(
            by="Gain", ascending=True), x="Gain", y="Feature", orientation='h', title="Top 10 Features by Gain")
        st.plotly_chart(fig_fi_disp, use_container_width=True)


def risk_analytics_page():
    st.markdown('<h2 class="sub-header">📊 Risk Tier Analytics (Based on Test Set Behavior)</h2>',
                unsafe_allow_html=True)

    # This data IS from the TEST SET evaluation of XGBoost (Optimized) from model.ipynb (cell 28 output)
    risk_tier_data_xgb = {
        'Risk Tier': ['Very Low (<0.2)', 'Low (0.2-0.4)', 'Medium (0.4-0.6)', 'High (0.6-0.8)', 'Very High (>=0.8)'],
        # Percentage of total test students in this predicted tier
        'Student Population (%)': [66.08, 22.34, 8.19, 2.85, 0.54],
        # Actual dropout rate within this tier
        'Actual Dropout Rate (%)': [7.54, 27.60, 48.05, 67.92, 85.89],
        # Number of test students in this tier
        'Student Count': [19823, 6702, 2458, 854, 163]
    }
    df_xgb_risk = pd.DataFrame(risk_tier_data_xgb)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Student Distribution Across Risk Tiers (XGBoost)")
        fig_dist = px.pie(df_xgb_risk,
                          values='Student Population (%)',
                          names='Risk Tier',
                          title="Student Population % per Predicted Risk Tier",
                          hole=0.3,
                          color_discrete_sequence=['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#DC143C'])
        # Emphasize high risk tiers
        fig_dist.update_traces(textinfo='percent+label',
                               pull=[0, 0, 0, 0.05, 0.1])
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        st.subheader("Actual Dropout Rate by Predicted Risk Tier (XGBoost)")
        fig_dropout = px.bar(df_xgb_risk,
                             x='Risk Tier',
                             y='Actual Dropout Rate (%)',
                             title="Actual Dropout Rate (%) vs Predicted Risk Tier",
                             labels={
                                 'Actual Dropout Rate (%)': 'Actual Dropout Rate (%)', 'Risk Tier': 'Predicted Risk Tier (XGBoost)'},
                             color='Actual Dropout Rate (%)',
                             color_continuous_scale='Reds',
                             text='Actual Dropout Rate (%)')
        fig_dropout.update_traces(
            texttemplate='%{text:.1f}%', textposition='outside')
        fig_dropout.update_layout(xaxis_categoryorder='array',
                                  # Ensure correct order
                                  xaxis_categoryarray=df_xgb_risk['Risk Tier'])
        st.plotly_chart(fig_dropout, use_container_width=True)

    st.subheader("📈 Risk Tier Performance: Multi-Model Comparison (Test Set)")
    st.caption(
        "Compares actual dropout rates within each model's own definition of risk tiers.")
    # This data IS from the TEST SET evaluations (cell 28 of notebook) for each model's own tiering.
    comparison_data = {
        # Generic labels for common x-axis
        'Risk Tier Category': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        'XGBoost (Actual Dropout %)': [7.54, 27.60, 48.05, 67.92, 85.89],
        'Random Forest (Actual Dropout %)': [3.46, 8.04, 19.88, 34.56, 66.76],
        'Logistic Regression (Actual Dropout %)': [3.77, 7.35, 17.60, 34.36, 55.89]
    }
    df_comparison = pd.DataFrame(comparison_data)

    fig_comp = go.Figure()
    colors = {'XGBoost': '#1f77b4', 'Random Forest': '#2ca02c',
              'Logistic Regression': '#ff7f0e'}

    for model_name_iter_full in ['XGBoost (Actual Dropout %)', 'Random Forest (Actual Dropout %)', 'Logistic Regression (Actual Dropout %)']:
        short_name = model_name_iter_full.split(' (')[0]
        fig_comp.add_trace(go.Scatter(
            x=df_comparison['Risk Tier Category'],
            y=df_comparison[model_name_iter_full],
            mode='lines+markers',
            name=short_name,
            line=dict(width=3, color=colors[short_name]),
            marker=dict(size=8)
        ))
    fig_comp.update_layout(title="Actual Dropout Rate by Model-Defined Risk Tiers",
                           xaxis_title="Conceptual Risk Tier (Low to High)",
                           yaxis_title="Actual Dropout Rate (%)",
                           height=400,
                           hovermode='x unified',
                           legend_title_text='Model')
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("📋 Detailed XGBoost Risk Tier Analysis (Test Set)")
    df_xgb_risk['Students at Risk (Expected Dropouts)'] = (
        df_xgb_risk['Student Count'] * df_xgb_risk['Actual Dropout Rate (%)'] / 100).round().astype(int)
    # Intervention Priority is subjective but helps guide action
    df_xgb_risk['Intervention Priority'] = [
        'Lowest', 'Low', 'Medium', 'High', 'Highest/Critical']
    st.dataframe(df_xgb_risk[['Risk Tier', 'Student Count', 'Student Population (%)', 'Actual Dropout Rate (%)', 'Students at Risk (Expected Dropouts)', 'Intervention Priority']],
                 use_container_width=True)

    # Key insights and recommendations from your previous app structure
    st.subheader("🔍 Key Insights from XGBoost Test Set Analysis")
    total_test_students = df_xgb_risk['Student Count'].sum()
    # Calculate for 'High' and 'Very High' tiers
    high_very_high_tiers_df = df_xgb_risk[df_xgb_risk['Risk Tier'].isin(
        ['High (0.6-0.8)', 'Very High (>=0.8)'])]
    high_vh_students_count = high_very_high_tiers_df['Student Count'].sum()

    # Calculate weighted average dropout rate for High/Very High tiers
    if high_vh_students_count > 0:
        avg_high_vh_dropout_rate = np.average(
            high_very_high_tiers_df['Actual Dropout Rate (%)'], weights=high_very_high_tiers_df['Student Count'])
        high_vh_percentage_of_total = (
            high_vh_students_count / total_test_students) * 100
    else:
        avg_high_vh_dropout_rate = 0
        high_vh_percentage_of_total = 0

    total_expected_dropouts_all_tiers = df_xgb_risk['Students at Risk (Expected Dropouts)'].sum(
    )

    insight_cols = st.columns(3)
    insight_cols[0].metric("Total High/Very High Risk Students",
                           f"{high_vh_students_count:,}",
                           f"{high_vh_percentage_of_total:.1f}% of test set")
    insight_cols[1].metric("Avg Dropout Rate in High/Very High Tiers",
                           f"{avg_high_vh_dropout_rate:.1f}%")
    insight_cols[2].metric("Total Expected Dropouts (XGBoost)",
                           f"{total_expected_dropouts_all_tiers:,}",
                           f"Overall: {total_expected_dropouts_all_tiers/total_test_students*100:.1f}%")

    st.subheader("💡 Strategic Recommendations")
    recommendations = [
        f"🎯 Focus on High & Very High Risk Tiers : These ~{high_vh_percentage_of_total:.1f}% of students account for a significant portion of dropouts, with an average actual dropout rate around {avg_high_vh_dropout_rate:.1f}%.",
        "📊 Model Validation : XGBoost effectively segments students by risk on the test set, with actual dropout rates clearly increasing from Very Low (~7.5%) to Very High (~86%).",
        f"⚡ Preventive Focus: The ~{df_xgb_risk.loc[df_xgb_risk['Risk Tier'] == 'Very Low (<0.2)', 'Student Population (%)'].values[0]:.0f}% of students in the 'Very Low' risk tier have a low actual dropout rate. Maintain standard support and monitor.",
        f"🔧 Targeted Programs for Medium Risk: The 'Medium' risk tier ({df_xgb_risk.loc[df_xgb_risk['Risk Tier'] == 'Medium (0.4-0.6)', 'Student Population (%)'].values[0]:.1f}% of students) still shows a substantial dropout rate (~48%). These students need targeted interventions before they escalate to higher risk.",
        "📈 Monitoring System: Implement a system to track students, especially those moving into higher risk tiers over time, for timely intervention.",
        "🤝 Stakeholder Engagement: Share these risk analytics with teachers, counselors, and administrators to inform resource allocation and support strategies."
    ]
    for rec in recommendations:
        st.text(rec)


def model_testing_page(current_models, current_ohe):
    st.markdown('<h2 class="sub-header">🧪 Model Testing & Sensitivity Analysis</h2>',
                unsafe_allow_html=True)

    if not current_models or not current_models.get("xgb"):
        st.warning("XGBoost model not loaded. Testing functionality limited.")
        return
    if not current_ohe or not st.session_state.get('ohe_loaded', False):
        st.error("OneHotEncoder not loaded. Cannot perform tests.")
        return

    # --- Helper function to process a single profile for testing ---
    def process_test_profile(profile_data_dict):
        profile_df = pd.DataFrame([profile_data_dict])

        # Calculate expected_age if grade is present
        if 'grade' in profile_df.columns:
            profile_df['expected_age'] = profile_df['grade'] + \
                6  # Consistent with individual prediction page

        # Apply engineered feature calculations
        profile_df['infrastructure_score'] = 0.1  # Hardcoded
        try:
            profile_df = create_socioeconomic_index(profile_df.copy())
            profile_df = create_school_support_score(profile_df.copy())
            profile_df = create_accessibility_score(profile_df.copy())
        except Exception as e_calc:
            st.error(
                f"Error calculating engineered features for test profile: {e_calc}")
            return None, None  # Indicate failure

        # Preprocess data (OHE, column alignment)
        processed_df_raw_ohe = preprocess_data_with_saved_ohe(
            profile_df, current_ohe, CATEGORICAL_FEATURE_NAMES_FOR_OHE,
            EXPECTED_COLUMNS_AFTER_OHE_RAW, NUMERICAL_FEATURE_NAMES
        )
        if processed_df_raw_ohe.empty:
            st.error("Preprocessing failed for test profile.")
            return None, processed_df_raw_ohe  # Return raw ohe for debugging if it exists

        # Rename columns to what model expects
        df_for_model = processed_df_raw_ohe.copy()
        if len(df_for_model.columns) == len(MODEL_EXPECTS_THESE_EXACT_NAMES):
            df_for_model.columns = MODEL_EXPECTS_THESE_EXACT_NAMES
        else:
            st.error("Column count mismatch for test profile.")
            return None, processed_df_raw_ohe

        return df_for_model, processed_df_raw_ohe

    st.markdown("#### 1. Extreme Case Testing")
    st.write(
        "Define profiles for a high-risk and a low-risk student to check model response.")

    # Define base features for test profiles
    # (Using the same structure as your individual_prediction_page form inputs)
    very_high_risk_base_features = {
        'age': 15, 'grade': 5, 'gender': 'Female', 'caste': 'ST',
        'father_education': 'Primary', 'family_income': '< ₹2 Lakhs',
        'attendance_rate': 20.0, 'grade_performance': 15.0,
        'midday_meal_access': 0, 'free_text_books_access': 0, 'free_uniform_access': 0,
        'internet_access_home': 0, 'distance_to_school': 15.0, 'rural_urban_y': 1,
        'school_category': 1, 'school_type': 1,
        'avg_instr_days': 150, 'medical_checkups': 0,
        'student_teacher_ratio': 6.0, 'girl_ratio': 0.1, 'female_teacher_ratio': 0.10,
    }

    very_low_risk_base_features = {
        'age': 12, 'grade': 7, 'gender': 'Male', 'caste': 'General',
        'father_education': 'Graduate', 'family_income': '> ₹5 Lakhs',
        'attendance_rate': 98.0, 'grade_performance': 95.0,
        'midday_meal_access': 1, 'free_text_books_access': 1, 'free_uniform_access': 1,
        'internet_access_home': 1, 'distance_to_school': 0.5, 'rural_urban_y': 0,
        'school_category': 4, 'school_type': 3,
        'avg_instr_days': 220, 'medical_checkups': 1,
        'student_teacher_ratio': 15.0, 'girl_ratio': 0.50, 'female_teacher_ratio': 0.70,
    }

    test_cols = st.columns(2)
    with test_cols[0]:
        st.markdown("##### Config: High Risk Profile")
        st.json(very_high_risk_base_features, expanded=False)
    with test_cols[1]:
        st.markdown("##### Config: Low Risk Profile")
        st.json(very_low_risk_base_features, expanded=False)

    if st.button("🔬 Run Extreme Case Test", use_container_width=True):
        prob_high, prob_low = None, None

        df_high_risk_for_model, _ = process_test_profile(
            very_high_risk_base_features)
        df_low_risk_for_model, _ = process_test_profile(
            very_low_risk_base_features)

        if df_high_risk_for_model is not None:
            try:
                dmatrix_high = xgb.DMatrix(
                    df_high_risk_for_model, feature_names=list(df_high_risk_for_model.columns))
                prob_high = float(
                    current_models["xgb"].predict(dmatrix_high)[0])
            except Exception as e:
                st.error(f"Prediction error for high risk profile: {e}")

        if df_low_risk_for_model is not None:
            try:
                dmatrix_low = xgb.DMatrix(
                    df_low_risk_for_model, feature_names=list(df_low_risk_for_model.columns))
                prob_low = float(current_models["xgb"].predict(dmatrix_low)[0])
            except Exception as e:
                st.error(f"Prediction error for low risk profile: {e}")

        if prob_high is not None and prob_low is not None:
            res_cols = st.columns(3)
            tier_high, _ = get_risk_tier_details(prob_high)
            tier_low, _ = get_risk_tier_details(prob_low)
            res_cols[0].metric("High Risk Profile Prob.",
                               f"{prob_high*100:.1f}%")
            res_cols[0].info(f"Risk Tier: {tier_high}")
            res_cols[1].metric("Low Risk Profile Prob.",
                               f"{prob_low*100:.1f}%")
            res_cols[1].info(f"Risk Tier: {tier_low}")
            res_cols[2].metric("Probability Difference",
                               f"{(prob_high - prob_low)*100:.1f}% pts")
            if prob_high > 0.25 and prob_low < 0.15:
                st.success(
                    "✅ Model shows good sensitivity to extreme profiles.")
            else:
                st.warning(
                    "⚠️ Model sensitivity to extremes could be further examined or profiles adjusted.")
        else:
            st.error(
                "Could not complete extreme case testing due to processing errors.")

    st.markdown("---")
    st.markdown("#### 2. Feature Sensitivity Analysis")
    st.write("Observe how changing a single feature affects the dropout probability for an average student profile.")

    # Define a base "average" student (using base features only)
    average_student_base_features = {
        'age': 13, 'grade': 8, 'gender': 'Male', 'caste': 'OBC',
        'father_education': 'Secondary', 'family_income': '₹2 - ₹3.5 Lakhs',
        'attendance_rate': 75.0, 'grade_performance': 65.0,
        'midday_meal_access': 1, 'free_text_books_access': 1, 'free_uniform_access': 0,
        'internet_access_home': 0, 'distance_to_school': 3.0, 'rural_urban_y': 1,
        'school_category': 2, 'school_type': 3,
        'avg_instr_days': 200, 'medical_checkups': 1,
        'student_teacher_ratio': 25.0, 'girl_ratio': 0.49, 'female_teacher_ratio': 0.40,
    }
    with st.expander("View/Edit Base Student Profile for Sensitivity Analysis"):
        # Allow editing of the base profile for more flexible testing if desired
        # For simplicity, we'll use the fixed average_student_base_features for now
        st.json(average_student_base_features)

    features_for_sensitivity = [  # Select numerical features suitable for varying
        'attendance_rate', 'grade_performance', 'distance_to_school',
        'age', 'grade', 'student_teacher_ratio',
        # 'infrastructure_score', # This is hardcoded
        # 'socioeconomic_index', # These are calculated, so vary their base components instead
        # 'school_support_score',
        # 'accessibility_score'
    ]
    selected_feature_to_vary = st.selectbox(
        "Select feature to vary:", features_for_sensitivity, index=0)

    if st.button(f"📈 Analyze Impact of {selected_feature_to_vary.replace('_', ' ').title()}", use_container_width=True):
        default_val = average_student_base_features[selected_feature_to_vary]
        if selected_feature_to_vary in ['attendance_rate', 'grade_performance']:
            min_val, max_val = 0.0, 100.0
        elif selected_feature_to_vary == 'distance_to_school':
            min_val, max_val = 0.0, 20.0
        elif selected_feature_to_vary == 'age':
            # Age relative to grade
            min_val, max_val = average_student_base_features['grade'] + \
                0, average_student_base_features['grade'] + 10
        elif selected_feature_to_vary == 'grade':
            min_val, max_val = 1, 12
        elif selected_feature_to_vary == 'student_teacher_ratio':
            min_val, max_val = 10.0, 70.0
        else:  # Fallback, may need adjustment based on feature
            min_val, max_val = float(default_val) * \
                0.5, float(default_val) * 1.5
            if min_val == max_val and min_val == 0:
                max_val = 1.0
            elif min_val == max_val:
                max_val = min_val + 1.0

        num_steps = 11  # Odd number for a middle point if needed
        varied_values = np.linspace(min_val, max_val, num_steps)
        probabilities = []

        st.write(
            f"Analyzing **{selected_feature_to_vary}** from **{min_val:.1f}** to **{max_val:.1f}**.")
        progress_bar_sens = st.progress(0)

        for i, value in enumerate(varied_values):
            temp_student_profile = average_student_base_features.copy()
            temp_student_profile[selected_feature_to_vary] = value

            # Recalculate expected_age if grade is varied
            if selected_feature_to_vary == 'grade':
                temp_student_profile['expected_age'] = int(value) + 6
            elif 'grade' in temp_student_profile:  # else ensure expected_age is based on the base grade
                temp_student_profile['expected_age'] = temp_student_profile['grade'] + 6

            df_for_model, _ = process_test_profile(temp_student_profile)

            if df_for_model is not None:
                try:
                    dmatrix = xgb.DMatrix(
                        df_for_model, feature_names=list(df_for_model.columns))
                    prob = float(current_models["xgb"].predict(dmatrix)[0])
                    probabilities.append(prob)
                except Exception as e:
                    # Add NaN if prediction fails for a point
                    probabilities.append(np.nan)
                    st.warning(
                        f"Prediction failed for {selected_feature_to_vary}={value}: {e}")
            else:
                # Add NaN if preprocessing fails for a point
                probabilities.append(np.nan)
            progress_bar_sens.progress((i + 1) / num_steps)

        # Filter out NaNs for plotting
        valid_indices = [i for i, p in enumerate(
            probabilities) if not np.isnan(p)]
        plot_varied_values = [varied_values[i] for i in valid_indices]
        plot_probabilities = [probabilities[i] *
                              100 for i in valid_indices]  # As percentage

        if plot_varied_values:
            fig_sensitivity = go.Figure(go.Scatter(
                x=plot_varied_values, y=plot_probabilities, mode='lines+markers'))
            fig_sensitivity.update_layout(
                title=f"Impact of {selected_feature_to_vary.replace('_', ' ').title()} on Dropout Risk",
                xaxis_title=selected_feature_to_vary.replace('_', ' ').title(),
                yaxis_title="Predicted Dropout Probability (%)", yaxis_range=[0, 100]
            )
            st.plotly_chart(fig_sensitivity, use_container_width=True)
            prob_range = max(plot_probabilities) - \
                min(plot_probabilities) if plot_probabilities else 0
            st.info(
                f"Varying '{selected_feature_to_vary}' from {min_val:.1f} to {max_val:.1f} resulted in a dropout probability range of {prob_range:.1f} percentage points.")
        else:
            st.warning(
                "Could not generate sensitivity plot. All predictions in the range failed or resulted in errors.")

# --- Main App Execution ---


def main():
    st.markdown('<h1 class="main-header">🎓 Student Dropout Risk Prediction System</h1>',
                unsafe_allow_html=True)
    if 'app_initialized' not in st.session_state:
        st.session_state.ml_models = load_all_models_cached()
        st.session_state.ohe_encoder = load_ohe_encoder_cached(OHE_PATH)
        st.session_state.app_initialized = True
    current_models = st.session_state.ml_models
    current_ohe = st.session_state.ohe_encoder

    st.sidebar.title("🧭 Navigation")
    page_options = ["🎓 Individual Prediction", "📊 Batch Analysis",
                    "📈 Model Performance", "💡 Risk Analytics", "🧪 Model Testing"]
    page_selection = st.sidebar.radio("Choose a page:", page_options)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Status")
    # ... (sidebar status checks) ...
    if st.session_state.get('xgb_loaded', False):
        st.sidebar.success("✅ XGBoost Model Loaded")
    else:
        st.sidebar.error("❌ XGBoost Model Not Loaded")
    if st.session_state.get('ohe_loaded', False):
        st.sidebar.success("✅ OHE Encoder Loaded")
    else:
        st.sidebar.error("❌ OHE Encoder FAILED to load!")  # Emphasize
    if st.session_state.get('rf_loaded', False):
        st.sidebar.info("ℹ️ Random Forest Loaded")
    if st.session_state.get('logistic_loaded', False):
        st.sidebar.info("ℹ️ Logistic Regression Loaded")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Risk Tiers (Val. Set Behavior)")
    st.sidebar.markdown("- **Very Low**: < 20% (~7.5% actual dropout)\n- **Low**: 20-40% (~27.6% actual dropout)\n- **Medium**: 40-60% (~48.1% actual dropout)\n- **High**: 60-80% (~67.9% actual dropout)\n- **Very High**: ≥ 80% (~85.9% actual dropout)")

    if page_selection == "🎓 Individual Prediction":
        individual_prediction_page(current_models, current_ohe)
    elif page_selection == "📊 Batch Analysis":
        batch_analysis_page(current_models, current_ohe)
    elif page_selection == "📈 Model Performance":
        model_performance_page()
    elif page_selection == "💡 Risk Analytics":
        risk_analytics_page()
    elif page_selection == "🧪 Model Testing":
        model_testing_page(current_models, current_ohe)


if __name__ == "__main__":
    main()
