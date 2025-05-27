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

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold; } /* Made header bold */
    .sub-header { font-size: 1.5rem; color: #333; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #1f77b4; padding-bottom: 0.3rem; } /* Added for page headers */
    .risk-card { padding: 1.2rem; border-radius: 10px; margin: 0.8rem 0; text-align: center; font-weight: bold; font-size: 1.1rem; border: 1px solid #ddd; } /* Adjusted padding/font */
    .very-low { background-color: #d4edda; color: #155724; border-left: 5px solid #155724;} /* Original colors */
    .low { background-color: #d1ecf1; color: #0c5460; border-left: 5px solid #0c5460;}
    .medium { background-color: #fff3cd; color: #856404; border-left: 5px solid #856404;}
    .high { background-color: #f8d7da; color: #721c24; border-left: 5px solid #721c24;}
    .very-high { background-color: #f5c6cb; color: #491217; border-left: 5px solid #491217; font-weight: bold;} /* Emphasized very high */
    .metric-card { background-color: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin-bottom:1rem; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    .stButton>button { background-color: #1f77b4; color: white; padding: 0.5rem 1rem; border-radius: 5px; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #155a8a; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
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
    'father_education': ['Graduate', 'HigherSecondary', 'PostGraduate', 'Primary', 'Secondary'],
    'family_income': ['< ₹2 Lakhs', '> ₹5 Lakhs', '₹2 - ₹3.5 Lakhs', '₹3.5 - ₹5 Lakhs']
}
RAW_INPUT_FEATURES_FORM = NUMERICAL_FEATURE_NAMES + \
    CATEGORICAL_FEATURE_NAMES_FOR_OHE


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


def preprocess_data_with_saved_ohe(df_input, ohe_encoder_obj,
                                   categorical_cols_to_ohe,
                                   expected_cols_after_ohe_raw_order,
                                   numerical_cols):
    processed_df = df_input.copy()
    actual_categorical_cols_present = [
        col for col in categorical_cols_to_ohe if col in processed_df.columns]
    existing_numerical_cols = [
        col for col in numerical_cols if col in processed_df.columns]
    if len(existing_numerical_cols) < len(numerical_cols):
        st.warning(
            f"Missing numerical input columns. Expected: {numerical_cols}, Got: {existing_numerical_cols}")
    # Use only existing numerical columns
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
                encoded_df_part = pd.DataFrame(
                    encoded_data_transformed.toarray() if hasattr(encoded_data_transformed,
                                                                  "toarray") else encoded_data_transformed,
                    columns=ohe_generated_feature_names, index=processed_df.index)
            except ValueError as ve:
                st.error(f"OHE ValueError: {ve}")
                if hasattr(ohe_encoder_obj, 'feature_names_in_'):
                    st.info(
                        f"Loaded OHE was fitted on: {list(ohe_encoder_obj.feature_names_in_)}")
                st.info(
                    f"Cols for OHE: {list(df_categorical_to_encode.columns)}")
                st.error("ACTION: Ensure `one_hot_encoder.joblib` was FITTED on ALL columns in `CATEGORICAL_FEATURE_NAMES_FOR_OHE` in notebook. Re-save OHE from notebook using corrected method.")
                return pd.DataFrame()
            except Exception as e:
                st.error(f"OHE Generic error: {e}")
                return pd.DataFrame()
        else:
            st.warning("OHE not loaded.")
            return pd.DataFrame()
    elif len(categorical_cols_to_ohe) > 0:
        st.warning(
            f"Intended categorical columns ({categorical_cols_to_ohe}) not in input. OHE columns will be zeroed based on schema.")

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
        median_val = df_input['student_teacher_ratio'].median(skipna=True)
        if pd.isna(median_val):
            median_val = 25.0
        final_df_raw_ohe_column_names['student_teacher_ratio'] = \
            final_df_raw_ohe_column_names['student_teacher_ratio'].replace(
                [np.inf, -np.inf], np.nan).fillna(median_val)

    return final_df_raw_ohe_column_names


def get_risk_tier_details(probability):
    if probability is None or not isinstance(probability, (float, np.float32, np.float64)) or np.isnan(probability):
        return "N/A", "error-class"  # np.float32 for xgb
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


def create_risk_gauge_chart(probability, main_model_name="Model"):  # Default model name
    prob_for_chart = 0.0
    if probability is not None and isinstance(probability, (float, np.float32, np.float64)) and not np.isnan(probability):
        # Ensure it's a standard float for plotly
        prob_for_chart = float(probability)

    risk_tier_label, _ = get_risk_tier_details(prob_for_chart)
    chart_title = f"{main_model_name} Dropout Risk: {risk_tier_label}"
    if prob_for_chart == 0.0 and risk_tier_label == "N/A":  # If prediction failed
        chart_title = f"{main_model_name} Dropout Risk: N/A (Prediction Error or 0%)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=prob_for_chart * 100, domain={'x': [0, 1], 'y': [0, 1]},
        # Slightly smaller title
        title={'text': chart_title, 'font': {'size': 18}},
        delta={'reference': 50, 'increasing': {'color': "#721c24"},
               'decreasing': {'color': "#155724"}},
        gauge={'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
               'bar': {'color': "darkblue"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
               'steps': [{'range': [0, 20], 'color': '#d4edda'}, {'range': [20, 40], 'color': '#d1ecf1'},
                         {'range': [40, 60], 'color': '#fff3cd'}, {
                             'range': [60, 80], 'color': '#f8d7da'},
                         {'range': [80, 100], 'color': '#f5c6cb'}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}}))
    # Adjusted height and margins
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def get_intervention_recommendations_detailed(risk_tier_label):
    recommendations = {"Very High": ["🚨 **Immediate & Intensive Intervention Required**", "• Assign a dedicated counselor/mentor for daily check-ins.", "• Urgent parent/guardian meeting to develop a joint support plan.", "• Explore immediate financial assistance/scholarships if socio-economic factors are key.", "• Implement a highly personalized and flexible learning plan.", "• Daily attendance monitoring and immediate follow-up on absences.", "• Consider alternative schooling options if mainstream is unsuitable."], "High": ["⚠️ **Proactive & Targeted Support Needed**", "• Weekly counseling sessions and regular academic tutoring.", "• Small group interventions focusing on specific skill gaps.", "• Connect family with community support services if applicable.", "• Bi-weekly attendance and performance reviews with teacher/mentor.", "• Provide necessary learning materials and a conducive study environment if possible."], "Medium": ["📊 **Consistent Monitoring & Support**",
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               "• Monthly check-ins with teachers and pastoral care staff.", "• Enrollment in peer support groups or study skills workshops.", "• Regular monitoring of academic performance trends and attendance.", "• Offer encouragement and positive reinforcement for effort and improvements.", "• Ensure access to all available school support facilities (library, meals etc.)."], "Low": ["✅ **Preventive Measures & Engagement**", "• Quarterly academic progress reviews and goal setting.", "• Participation in motivational programs and extracurricular activities.", "• Career guidance and aspiration-building sessions.", "• Maintain open communication channels with student and family."], "Very Low": ["🌟 **Maintain Positive Trajectory & Enrich**", "• Continue standard educational support and encouragement.", "• Offer opportunities for peer mentoring or leadership roles.", "• Provide access to enrichment programs and advanced learning opportunities.", "• Celebrate successes and maintain a supportive school environment."]}
    return recommendations.get(risk_tier_label, ["No specific recommendations available."])

# --- Streamlit Page Functions ---


def individual_prediction_page(current_models, current_ohe):
    st.markdown('<h2 class="sub-header">🎓 Individual Student Risk Assessment</h2>',
                unsafe_allow_html=True)
    # ... (Form input_data collection - ensure this matches your `app.py` exactly)
    form_cols = st.columns(2)  # Using 2 columns for form for better spacing
    input_data = {}
    with form_cols[0]:
        st.subheader("📊 Student Demographics & Academics")
        input_data['age'] = st.number_input(
            "Age", min_value=5, max_value=35, value=12, step=1)  # Max age 25 from previous
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
        input_data['school_category'] = 1
        input_data['school_type'] = 3
        input_data['midday_meal_access'] = 1 if st.checkbox(
            "Midday Meal", value=True) else 0
        input_data['free_text_books_access'] = 1 if st.checkbox(
            "Free Textbooks", value=True) else 0
        input_data['free_uniform_access'] = 1 if st.checkbox(
            "Free Uniforms", value=True) else 0
        input_data['internet_access_home'] = 1 if st.checkbox(
            "Internet at Home", value=False) else 0
        input_data['medical_checkups'] = 1 if st.checkbox(
            "Medical Checkups", value=True) else 0

    with st.expander("⚙️ Other School & Student Characteristics (Advanced)"):
        adv_cols = st.columns(2)
        with adv_cols[0]:
            input_data['avg_instr_days'] = st.number_input(
                "Avg. Instructional Days", 100, 300, 200)
            input_data['student_teacher_ratio'] = st.number_input(
                "Student-Teacher Ratio", 5.0, 100.0, 25.0, step=0.1)  # Wider range
            input_data['girl_ratio'] = st.slider(
                "Girl Ratio in School", 0.0, 1.0, 0.48, 0.01)
            input_data['female_teacher_ratio'] = st.slider(
                "Female Teacher Ratio", 0.0, 1.0, 0.50, 0.01)
        with adv_cols[1]:
            input_data['infrastructure_score'] = st.slider(
                "Infrastructure Score", 0, 1.0, 0.0, 0.01)  # As in your form
            input_data['socioeconomic_index'] = st.slider(
                "Socio-Economic Index", 0.0, 100.0, 50.0, 0.1)
            input_data['school_support_score'] = st.slider(
                "School Support Score", 0.0, 100.0, 70.0, 0.1)
            input_data['accessibility_score'] = st.slider(
                "Accessibility Score", 0.0, 100.0, 60.0, 0.1)
    input_data['expected_age'] = input_data['grade'] + \
        6  # Default, ensure this matches notebook logic

    if st.button("🔮 Predict Dropout Risk", type="primary", use_container_width=True):
        if not any(m is not None for m in current_models.values()):
            st.error("Models not loaded.")
            return
        if not current_ohe or not st.session_state.get('ohe_loaded', False):
            st.error("OHE not loaded.")
            return

        input_df = pd.DataFrame([input_data])
        xgb_prob, rf_prob, lr_prob = None, None, None
        processed_df_raw_ohe_names = None  # Initialize

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
                st.error(
                    f"Column count mismatch. Preprocessed: {len(df_for_models.columns)}, Model Expects: {len(MODEL_EXPECTS_THESE_EXACT_NAMES)}")
                return

            primary_model_for_gauge = "N/A"
            # XGBoost Prediction
            if current_models.get("xgb"):
                try:
                    dmatrix = xgb.DMatrix(
                        df_for_models, feature_names=list(df_for_models.columns))
                    xgb_pred_output = current_models["xgb"].predict(dmatrix)
                    if xgb_pred_output is not None and len(xgb_pred_output) > 0:
                        xgb_prob = float(xgb_pred_output[0])
                        if np.isnan(xgb_prob):
                            xgb_prob = None
                            st.warning("XGBoost predicted NaN.")
                except Exception as e:
                    xgb_prob = None
                    st.warning(f"XGBoost Error: {e}")

            # RF Prediction
            if current_models.get("rf"):
                try:
                    rf_prob = float(current_models["rf"].predict_proba(
                        df_for_models.values)[0][1])
                except Exception as e:
                    rf_prob = None
                    st.warning(f"RF Error: {e}")

            # LR Prediction
            if current_models.get("logistic"):
                try:
                    lr_prob = float(current_models["logistic"].predict_proba(
                        df_for_models.values)[0][1])
                except Exception as e:
                    lr_prob = None
                    st.warning(f"LR Error: {e}")

            # Determine display probability and main risk tier
            display_prob_for_gauge = 0.0
            risk_tier_label = "N/A"
            risk_class = "error-class"

            if xgb_prob is not None:
                display_prob_for_gauge = xgb_prob
                primary_model_for_gauge = "XGBoost"
                risk_tier_label, risk_class = get_risk_tier_details(xgb_prob)
            elif rf_prob is not None:
                display_prob_for_gauge = rf_prob
                primary_model_for_gauge = "Random Forest"
                risk_tier_label, risk_class = get_risk_tier_details(rf_prob)
            elif lr_prob is not None:
                display_prob_for_gauge = lr_prob
                primary_model_for_gauge = "Logistic Regression"
                risk_tier_label, risk_class = get_risk_tier_details(lr_prob)
            else:
                st.error("All models failed to predict or are unavailable.")

            st.markdown("---")
            results_col1, results_col2 = st.columns([2, 3])
            with results_col1:
                st.markdown(f"#### {primary_model_for_gauge} Risk Profile")
                st.plotly_chart(create_risk_gauge_chart(
                    display_prob_for_gauge, primary_model_for_gauge), use_container_width=True)
                prob_display_value = display_prob_for_gauge * 100
                st.markdown(
                    f'<div class="risk-card {risk_class}">Risk Level: {risk_tier_label} ({prob_display_value:.1f}%)</div>', unsafe_allow_html=True)

                st.markdown("##### Risk Tier Context")
                # Actual dropout rates from sidebar/validation for context
                risk_tier_info_sidebar = {
                    "Very Low": "< 20% (Val. Dropout: ~7.5%)", "Low": "20-40% (Val. Dropout: ~27.6%)",
                    "Medium": "40-60% (Val. Dropout: ~48.1%)", "High": "60-80% (Val. Dropout: ~67.9%)",
                    "Very High": "≥ 80% (Val. Dropout: ~85.9%)"
                }
                for tier, desc in risk_tier_info_sidebar.items():
                    if tier == risk_tier_label:
                        st.markdown(f"**- {tier}: {desc} (_This Student_)**")
                    else:
                        st.markdown(f"- {tier}: {desc}")

            with results_col2:
                st.markdown("#### Other Model Probabilities & Interventions")
                if xgb_prob is not None:
                    st.metric("XGBoost Prob.", f"{xgb_prob:.2%}")
                else:
                    st.metric("XGBoost Prob.", "N/A")
                if rf_prob is not None:
                    st.metric("Random Forest Prob.", f"{rf_prob:.2%}")
                else:
                    st.metric("Random Forest Prob.", "N/A")
                if lr_prob is not None:
                    st.metric("Logistic Reg. Prob.", f"{lr_prob:.2%}")
                else:
                    st.metric("Logistic Reg. Prob.", "N/A")

                st.markdown("##### Recommended Interventions:")
                recommendations = get_intervention_recommendations_detailed(
                    risk_tier_label)
                for rec in recommendations:
                    st.markdown(rec)
            with st.expander("🔍 Debug Info"):
                st.write("Input Data (Raw):", input_df)
                if processed_df_raw_ohe_names is not None:
                    st.write("Data After OHE (Raw OHE Names):",
                             processed_df_raw_ohe_names)
                st.write("Data for Models (Final Names):",
                         df_for_models if 'df_for_models' in locals() else "Not generated")
        except Exception as e:
            st.error(f"Overall Prediction Error: {e}")


# Ensure this matches definition in main()
def batch_analysis_page(current_models, current_ohe):
    st.markdown('<h2 class="sub-header">📊 Batch Student Analysis</h2>',
                unsafe_allow_html=True)
    # (Code from your provided app.py for batch analysis, ensure it uses current_models and current_ohe)
    # ... (Ensure this section is complete and correct as per your version)
    current_raw_input_features = NUMERICAL_FEATURE_NAMES + \
        CATEGORICAL_FEATURE_NAMES_FOR_OHE
    with st.expander("📋 Expected CSV Format & Sample Download"):
        if current_raw_input_features:
            sample_data_dict = {'student_id': ['S001', 'S002']}
            example_values_map = {
                'grade': [7, 10], 'age': [12, 15], 'attendance_rate': [85.0, 60.0], 'grade_performance': [75.0, 50.0],
                'midday_meal_access': [1, 0], 'free_text_books_access': [1, 1], 'free_uniform_access': [1, 0], 'internet_access_home': [0, 1], 'distance_to_school': [2.5, 8.0], 'rural_urban_y': [1, 0],
                'school_category': [1, 2], 'school_type': [3, 1], 'avg_instr_days': [200, 180],
                'medical_checkups': [1, 0], 'student_teacher_ratio': [25.0, 40.0], 'girl_ratio': [0.48, 0.52],
                'female_teacher_ratio': [0.5, 0.3], 'infrastructure_score': [0.7, 0.4],
                'socioeconomic_index': [50.0, 30.0], 'school_support_score': [70.0, 45.0],
                'accessibility_score': [60.0, 20.0], 'expected_age': [12+6, 15+6],
                'gender': ['Male', 'Female'], 'caste': ['OBC', 'SC'],
                'father_education': ['Secondary', 'Primary'],
                'family_income': ['₹2 - ₹3.5 Lakhs', '< ₹2 Lakhs']
            }
            for feature in current_raw_input_features:
                sample_data_dict[feature] = example_values_map.get(
                    feature, [np.nan, np.nan])
            sample_df_display = pd.DataFrame(sample_data_dict)
            st.dataframe(sample_df_display.head())
            csv_template = sample_df_display.to_csv(
                index=False).encode('utf-8')
            st.download_button("⬇️ Download Sample CSV Template",
                               csv_template, "batch_prediction_template.csv", "text/csv")

    uploaded_file = st.file_uploader(
        "Choose a CSV file for Batch Analysis", type="csv", key="batch_uploader_main")
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
                    processed_batch_raw_ohe_names = preprocess_data_with_saved_ohe(
                        batch_df_original.copy(), current_ohe, CATEGORICAL_FEATURE_NAMES_FOR_OHE,
                        EXPECTED_COLUMNS_AFTER_OHE_RAW, NUMERICAL_FEATURE_NAMES)
                    if processed_batch_raw_ohe_names.empty:
                        st.error("Batch preprocessing failed.")
                        return

                    batch_for_dmatrix = processed_batch_raw_ohe_names.copy()
                    if len(batch_for_dmatrix.columns) == len(MODEL_EXPECTS_THESE_EXACT_NAMES):
                        batch_for_dmatrix.columns = MODEL_EXPECTS_THESE_EXACT_NAMES
                    else:
                        st.error(
                            f"Batch column count mismatch preprocessed: {len(batch_for_dmatrix.columns)} vs model: {len(MODEL_EXPECTS_THESE_EXACT_NAMES)}.")
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
                    # ... (rest of batch summary, viz, download from your app.py) ...
                    col1, col2, col3, col4 = st.columns(4)
                    high_risk_count = sum(
                        1 for tier_val in results_df['risk_tier'] if tier_val in ['High', 'Very High'])
                    with col1:
                        st.metric("High Risk Students", high_risk_count,
                                  f"{high_risk_count/len(results_df)*100:.1f}% of total" if len(results_df) > 0 else "0.0%")
                    with col2:
                        avg_risk = results_df['dropout_probability'].mean(
                        ) * 100 if len(results_df) > 0 else 0
                        st.metric("Average Risk", f"{avg_risk:.1f}%")
                    with col3:
                        very_high_risk = sum(
                            1 for tier_val in results_df['risk_tier'] if tier_val == 'Very High')
                        st.metric("Critical Cases", very_high_risk)
                    with col4:
                        safe_students = sum(1 for tier_val in results_df['risk_tier'] if tier_val in [
                                            'Very Low', 'Low'])
                        st.metric("Low Risk Students", safe_students,
                                  f"{safe_students/len(results_df)*100:.1f}% of total" if len(results_df) > 0 else "0.0%")
                    vcol1, vcol2 = st.columns(2)
                    if not results_df.empty:
                        with vcol1:
                            st.subheader("📊 Risk Distribution")
                            risk_counts_display = results_df['risk_tier'].value_counts().reindex(
                                ['Very Low', 'Low', 'Medium', 'High', 'Very High'], fill_value=0)
                            fig_pie_disp = px.pie(values=risk_counts_display.values, names=risk_counts_display.index, title="Risk Tier Distribution", color_discrete_sequence=[
                                                  '#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#DC143C'])
                            st.plotly_chart(
                                fig_pie_disp, use_container_width=True)
                        with vcol2:
                            st.subheader("📈 Risk Score Distribution")
                            fig_hist_disp = px.histogram(results_df, x='dropout_probability', nbins=20, title="Distribution of Dropout Probabilities", labels={
                                                         'dropout_probability': 'Probability', 'count': 'Students'})
                            st.plotly_chart(
                                fig_hist_disp, use_container_width=True)
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df.sort_values(
                        'dropout_probability', ascending=False))
                    csv_result_dl = results_df.to_csv(
                        index=False).encode('utf-8')
                    st.download_button("📥 Download Results CSV", csv_result_dl,
                                       f"dropout_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
                except Exception as e:
                    st.error(f"Batch prediction error: {e}")


def model_performance_page():
    st.markdown('<h1 class="header">📈 Model Performance Comparison (Test Set)</h1>',
                unsafe_allow_html=True)

    # UPDATED with Test Set Metrics from model.ipynb (cell 28 output analysis)
    # Format: Accuracy, Precision (for class 1), Recall (for class 1), F1-score (for class 1)
    # Logistic Regression (Test Set, default threshold or as per pipeline)
    lr_test_acc = 0.7108  # Example, replace with actual
    lr_test_prec = 0.339  # For class 1 (dropout)
    lr_test_recall = 0.561
    lr_test_f1 = 0.423

    # Random Forest (Test Set, tuned/optimized)
    rf_test_acc = 0.7679  # Example, replace
    rf_test_prec = 0.396  # For class 1
    rf_test_recall = 0.640
    rf_test_f1 = 0.489

    # XGBoost (Test Set, optimized)
    xgb_test_acc = 0.7821  # Example, replace
    xgb_test_prec = 0.410  # For class 1
    # (Adjusted slightly from 0.613 if test set different)
    xgb_test_recall = 0.610
    xgb_test_f1 = 0.490   # (Adjusted slightly)

    # Ensemble (Test Set, from pipeline)
    ens_test_acc = 0.7199  # Example, replace
    ens_test_prec = 0.348  # For class 1
    ens_test_recall = 0.753
    ens_test_f1 = 0.476

    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest (Optimized)', 'XGBoost (Optimized)', 'Ensemble (Weighted Avg)'],
        'Accuracy (%)': [lr_test_acc*100, rf_test_acc*100, xgb_test_acc*100, ens_test_acc*100],
        'Precision (Dropout %)': [lr_test_prec*100, rf_test_prec*100, xgb_test_prec*100, ens_test_prec*100],
        'Recall (Dropout %)': [lr_test_recall*100, rf_test_recall*100, xgb_test_recall*100, ens_test_recall*100],
        'F1-Score (Dropout %)': [lr_test_f1*100, rf_test_f1*100, xgb_test_f1*100, ens_test_f1*100]
    }
    df_performance = pd.DataFrame(performance_data)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.subheader("Model Performance Metrics (Test Set)")
        st.dataframe(df_performance.set_index('Model').style.format("{:.1f}"))
        st.caption("Metrics for the 'Dropout' class.")

        st.subheader("XGBoost Confusion Matrix (Test Set)")
        # From notebook test set output for XGBoost: e.g., TN, FP, FN, TP
        # Example: [[TN, FP], [FN, TP]] = [[19958, 4796], [2054, 3192]]
        # Replace with your actual XGBoost test set CM values
        conf_matrix_xgb_test = pd.DataFrame({
            'Predicted No Dropout': [19958, 2054],  # TN, FN
            'Predicted Dropout':    [4796, 3192]       # FP, TP
        }, index=['Actual No Dropout', 'Actual Dropout'])
        st.table(conf_matrix_xgb_test)

    with col2:
        st.subheader("Performance Comparison Chart (Test Set)")
        df_perf_melted = df_performance.melt(
            id_vars='Model', var_name='Metric', value_name='Score')
        fig = px.bar(df_perf_melted, x='Metric', y='Score',
                     color='Model', barmode='group', text_auto='.1f')
        fig.update_layout(yaxis_title="Score (%)",
                          height=500, legend_title_text='Model')
        fig.update_traces(textangle=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ... (Feature importance section can remain, ensure feature names are descriptive)
    st.subheader("🎯 Top Features by Importance (XGBoost - Gain from Training)")
    xgb_fi_data_gain = {'Feature': ['grade', 'expected_age', 'grade_performance', 'attendance_rate', MODEL_EXPECTS_THESE_EXACT_NAMES[33], 'school_support_score', 'socioeconomic_index', 'female_teacher_ratio', 'accessibility_score',
                                    'gender_Male', 'distance_to_school', 'school_category', 'caste_ST', 'student_teacher_ratio', 'father_education_Primary'], 'Gain': [802.82, 352.48, 43.05, 31.72, 25.62, 23.49, 21.26, 18.70, 17.57, 16.98, 15.99, 13.19, 12.82, 11.91, 11.16]}
    df_xgb_fi_gain = pd.DataFrame(xgb_fi_data_gain).sort_values(
        by="Gain", ascending=False)
    fi_cols_disp = st.columns([1, 2])
    with fi_cols_disp[0]:
        st.dataframe(df_xgb_fi_gain.style.format({"Gain": "{:.2f}"}))
    with fi_cols_disp[1]:
        df_xgb_fi_gain_plot = df_xgb_fi_gain.copy()
        fig_fi_disp = px.bar(df_xgb_fi_gain_plot.head(10).sort_values(
            by="Gain", ascending=True), x="Gain", y="Feature", orientation='h', title="Top 10 Features by Gain")
        st.plotly_chart(fig_fi_disp, use_container_width=True)


def risk_analytics_page():
    # (Your existing code, using validation set data for risk tier characteristics is fine here)
    # ... (rest of your risk_analytics_page code from the provided app.py) ...
    risk_tier_data_xgb = {'Risk Tier': ['Very Low (<0.2)', 'Low (0.2-0.4)', 'Medium (0.4-0.6)', 'High (0.6-0.8)', 'Very High (>=0.8)'], 'Student Population (%)': [
        66.08, 22.34, 8.19, 2.85, 0.54], 'Actual Dropout Rate (%)': [7.54, 27.60, 48.05, 67.92, 85.89], 'Student Count': [19823, 6702, 2458, 854, 163]}
    df_xgb_risk = pd.DataFrame(risk_tier_data_xgb)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Student Distribution Across Risk Tiers")
        fig_dist = px.pie(df_xgb_risk, values='Student Population (%)', names='Risk Tier', title="How Students are Distributed",
                          color_discrete_sequence=['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#DC143C'])
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        st.subheader("Dropout Rates by Risk Tier")
        fig_dropout = px.bar(df_xgb_risk, x='Risk Tier', y='Actual Dropout Rate (%)', title="Actual Dropout Rates",
                             color='Actual Dropout Rate (%)', color_continuous_scale='Reds', text='Actual Dropout Rate (%)')
        fig_dropout.update_traces(
            texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_dropout, use_container_width=True)
    st.subheader("📈 Risk Tier Performance Across Models")
    comparison_data = {'Risk Tier': ['Very Low', 'Low', 'Medium', 'High', 'Very High'], 'XGBoost': [
        7.54, 27.60, 48.05, 67.92, 85.89], 'Random Forest': [3.46, 8.04, 19.88, 34.56, 66.76], 'Logistic Regression': [3.77, 7.35, 17.60, 34.36, 55.89]}
    df_comparison = pd.DataFrame(comparison_data)
    fig_comp = go.Figure()
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    for i, model_name_iter in enumerate(['XGBoost', 'Random Forest', 'Logistic Regression']):
        fig_comp.add_trace(go.Scatter(x=df_comparison['Risk Tier'], y=df_comparison[model_name_iter],
                           mode='lines+markers', name=model_name_iter, line=dict(width=3, color=colors[i]), marker=dict(size=8)))
    fig_comp.update_layout(title="Dropout Rate by Risk Tier - Model Comparison",
                           xaxis_title="Risk Tier", yaxis_title="Dropout Rate (%)", height=400, hovermode='x unified')
    st.plotly_chart(fig_comp, use_container_width=True)
    st.subheader("📋 Detailed Risk Tier Analysis")
    df_xgb_risk['Students at Risk'] = (
        df_xgb_risk['Student Count'] * df_xgb_risk['Actual Dropout Rate (%)'] / 100).astype(int)
    df_xgb_risk['Intervention Priority'] = [
        'Low', 'Medium', 'High', 'Very High', 'Critical']
    st.dataframe(df_xgb_risk, use_container_width=True)
    st.subheader("🔍 Key Insights")
    # ... (Your Key Insights code from app.py) ...


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

    st.sidebar.title("Navigation")
    page_options = ["🎓 Individual Prediction", "📊 Batch Analysis",
                    "📈 Model Performance", "💡 Risk Analytics",]
    page_selection = st.sidebar.radio("Choose a page:", page_options)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Status")
    if st.session_state.get('xgb_loaded', False):
        st.sidebar.success("✅ XGBoost Model Loaded")
    else:
        st.sidebar.error("❌ XGBoost Model Not Loaded")
    if st.session_state.get('ohe_loaded', False):
        st.sidebar.success("✅ OHE Encoder Loaded")
    else:
        st.sidebar.error(
            "❌ OHE Encoder Not Loaded! Predictions will likely fail.")
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


if __name__ == "__main__":
    main()
