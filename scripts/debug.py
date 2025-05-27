import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Dropout Risk Prediction - Diagnostic Mode",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnostic-box {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .match { background-color: #d4edda; }
    .mismatch { background-color: #f8d7da; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_models():
    """Load trained models"""
    try:
        rf_model = joblib.load('models/rf_model.joblib')
        logistic_model = joblib.load('models/logistic_model.joblib')
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('models/xgboost_model.json')
        return rf_model, logistic_model, xgb_model
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure models are saved in the 'models/' directory.")
        return None, None, None


@st.cache_data
def load_test_data():
    """Load test.csv if available"""
    try:
        df = pd.read_csv('/Users/macbookpro/Desktop/POC1/scripts/test.csv')
        return df
    except:
        return None


def get_feature_names():
    """Get the exact feature names in order"""
    return [
        'grade', 'age', 'attendance_rate', 'grade_performance',
        'midday_meal_access', 'free_text_books_access', 'free_uniform_access',
        'internet_access_home', 'distance_to_school', 'rural_urban_y',
        'school_category', 'school_type', 'avg_instr_days', 'medical_checkups',
        'student_teacher_ratio', 'girl_ratio', 'female_teacher_ratio',
        'infrastructure_score', 'socioeconomic_index', 'school_support_score',
        'accessibility_score', 'expected_age', 'gender_Male', 'caste_OBC',
        'caste_SC', 'caste_ST', 'father_education_HigherSecondary',
        'father_education_PostGraduate', 'father_education_Primary',
        'father_education_Secondary', 'family_income_> ₹5 Lakhs',
        'family_income_₹2 - ₹3.5 Lakhs', 'family_income_₹3.5 - ₹5 Lakhs'
    ]


def reverse_engineer_from_features(features_dict):
    """Reverse engineer the original inputs from the feature vector"""
    # This helps us understand what the original values might have been
    result = {}

    # Basic features
    result['grade'] = features_dict.get('grade', 0)
    result['age'] = features_dict.get('age', 0)
    result['attendance_rate'] = features_dict.get('attendance_rate', 0)
    result['grade_performance'] = features_dict.get('grade_performance', 0)

    # Gender
    result['gender'] = 'Male' if features_dict.get(
        'gender_Male', False) else 'Female'

    # Caste
    if features_dict.get('caste_ST', False):
        result['caste_category'] = 'ST'
    elif features_dict.get('caste_SC', False):
        result['caste_category'] = 'SC'
    elif features_dict.get('caste_OBC', False):
        result['caste_category'] = 'OBC'
    else:
        result['caste_category'] = 'General'

    # Father education
    if features_dict.get('father_education_Primary', False):
        result['father_education'] = 'Primary'
    elif features_dict.get('father_education_Secondary', False):
        result['father_education'] = 'Secondary'
    elif features_dict.get('father_education_HigherSecondary', False):
        result['father_education'] = 'Higher Secondary'
    elif features_dict.get('father_education_PostGraduate', False):
        result['father_education'] = 'Post Graduate'
    else:
        result['father_education'] = 'No Education'

    # Family income
    if features_dict.get('family_income_> ₹5 Lakhs', False):
        result['family_income'] = '> 50000'
    elif features_dict.get('family_income_₹3.5 - ₹5 Lakhs', False):
        result['family_income'] = '25000-50000'
    elif features_dict.get('family_income_₹2 - ₹3.5 Lakhs', False):
        result['family_income'] = '15000-25000'
    else:
        result['family_income'] = '< 5000'

    # Other features
    result['distance_to_school'] = features_dict.get('distance_to_school', 0)
    result['rural_urban'] = 'Rural' if features_dict.get(
        'rural_urban_y', 0) == 1 else 'Urban'

    # School features
    result['midday_meal'] = bool(features_dict.get('midday_meal_access', 0))
    result['free_uniforms'] = bool(features_dict.get('free_uniform_access', 0))
    result['free_textbooks'] = bool(
        features_dict.get('free_text_books_access', 0))
    result['internet_access'] = bool(
        features_dict.get('internet_access_home', 0))
    result['medical_checkups'] = bool(features_dict.get('medical_checkups', 0))

    return result


def create_feature_vector_from_ui(student_data):
    """Create feature vector from UI inputs"""
    features = []

    # Extract all the UI inputs
    grade = student_data['grade']
    age = student_data['age']
    attendance_rate = student_data['attendance_rate']
    grade_performance = student_data['grade_performance']

    # Features 0-8
    features.extend([
        grade,
        age,
        attendance_rate,
        grade_performance,
        int(student_data['midday_meal']),
        int(student_data['free_textbooks']),
        int(student_data['free_uniforms']),
        int(student_data['internet_access']),
        student_data['distance_to_school']
    ])

    # Feature 9: rural_urban_y
    features.append(1 if student_data['rural_urban'] == 'Rural' else 0)

    # Features 10-17
    school_category_map = {"Primary": 1, "Upper Primary": 2,
                           "Secondary": 3, "Higher Secondary": 4}
    features.extend([
        school_category_map.get(student_data['school_category'], 1),
        3,  # school_type (always 3)
        student_data['avg_instr_days'],
        int(student_data['medical_checkups']),
        student_data['student_teacher_ratio'],
        student_data['girl_ratio'],
        student_data['female_teacher_ratio'],
        student_data['infrastructure_score']  # This needs RobustScaler
    ])

    # Features 18-21: Engineered features
    # Calculate these from the UI inputs
    socioeconomic_index = calculate_socioeconomic_index(
        student_data['father_education'],
        student_data['family_income'],
        student_data['caste_category']
    )
    school_support_score = calculate_school_support_score(
        student_data['midday_meal'],
        student_data['free_uniforms'],
        student_data['free_textbooks']
    )
    accessibility_score = calculate_accessibility_score(
        student_data['distance_to_school'])
    expected_age = grade + 5

    features.extend([
        socioeconomic_index,
        school_support_score,
        accessibility_score,
        expected_age
    ])

    # Features 22-32: One-hot encoded
    features.extend([
        1 if student_data['gender'] == 'Male' else 0,
        1 if student_data['caste_category'] == 'OBC' else 0,
        1 if student_data['caste_category'] == 'SC' else 0,
        1 if student_data['caste_category'] == 'ST' else 0,
        1 if student_data['father_education'] == 'Higher Secondary' else 0,
        1 if student_data['father_education'] == 'Post Graduate' else 0,
        1 if student_data['father_education'] == 'Primary' else 0,
        1 if student_data['father_education'] == 'Secondary' else 0,
        1 if student_data['family_income'] == '> 50000' else 0,
        1 if student_data['family_income'] == '15000-25000' else 0,
        1 if student_data['family_income'] == '25000-50000' else 0
    ])

    return features


def calculate_socioeconomic_index(father_education, family_income, caste_category):
    """Calculate socioeconomic index"""
    income_map = {
        '< 5000': 1,
        '5000-15000': 2,
        '15000-25000': 4,
        '25000-50000': 5,
        '> 50000': 6
    }

    education_map = {
        'No Education': 1,
        'Primary': 2,
        'Secondary': 3,
        'Higher Secondary': 4,
        'Graduate': 5,
        'Post Graduate': 6
    }

    caste_map = {
        'ST': 1,
        'SC': 2,
        'OBC': 3,
        'General': 4
    }

    income_score = income_map.get(family_income, 1)
    education_score = education_map.get(father_education, 1)
    caste_score = caste_map.get(caste_category, 1)

    income_score_norm = (income_score - 1) / 5
    education_score_norm = (education_score - 1) / 5
    caste_score_norm = (caste_score - 1) / 3

    socioeconomic_index = (0.5 * income_score_norm +
                           0.15 * education_score_norm +
                           0.35 * caste_score_norm) * 100

    return socioeconomic_index


def calculate_school_support_score(midday_meal, free_uniforms, free_textbooks):
    """Calculate school support score"""
    support_sum = int(midday_meal) + int(free_uniforms) + int(free_textbooks)
    return (support_sum / 3) * 100


def calculate_accessibility_score(distance_to_school):
    """Calculate accessibility score"""
    return 100 * np.exp(-0.5 * distance_to_school)


def diagnostic_page(xgb_model):
    st.header("🔬 Diagnostic Mode - Feature Comparison")

    test_df = load_test_data()
    if test_df is None:
        st.error(
            "test.csv not found. Please ensure it's in the same directory as the app.")
        return

    st.success(
        f"✅ Loaded test.csv with {len(test_df)} rows and {len(test_df.columns)} columns")

    # Show sample data
    st.subheader("📊 Sample Test Data (First 5 Rows)")
    st.dataframe(test_df.head())

    # Test predictions on sample data
    st.subheader("🧪 Test Model on Sample Data")

    n_samples = st.number_input(
        "Number of samples to test", min_value=1, max_value=100, value=10)

    if st.button("Run Test"):
        sample_df = test_df.head(n_samples)
        features = sample_df.values

        predictions = xgb_model.predict_proba(features)[:, 1]

        results_df = pd.DataFrame({
            'Sample': range(1, n_samples + 1),
            'Prediction': predictions,
            'Risk %': (predictions * 100).round(1)
        })

        st.dataframe(results_df)

        # Show distribution
        fig = px.histogram(x=predictions, nbins=20,
                           title="Distribution of Predictions on Test Data",
                           labels={'x': 'Dropout Probability', 'y': 'Count'})
        st.plotly_chart(fig)

    # Compare specific row
    st.subheader("🔍 Deep Dive: Compare Specific Row")

    row_idx = st.number_input(
        "Select row index to analyze", min_value=0, max_value=len(test_df)-1, value=0)

    if st.button("Analyze Row"):
        row = test_df.iloc[row_idx]
        st.write("**Selected Row Data:**")

        # Create two columns for better display
        col1, col2 = st.columns(2)

        feature_names = get_feature_names()

        with col1:
            st.write("**Features 0-16:**")
            for i in range(17):
                st.write(f"{i}. {feature_names[i]}: {row[feature_names[i]]}")

        with col2:
            st.write("**Features 17-32:**")
            for i in range(17, 33):
                st.write(f"{i}. {feature_names[i]}: {row[feature_names[i]]}")

        # Make prediction
        features_array = row.values.reshape(1, -1)
        prediction = xgb_model.predict_proba(features_array)[0][1]

        st.metric("Model Prediction", f"{prediction:.1%}")

        # Reverse engineer the original inputs
        features_dict = row.to_dict()
        original_inputs = reverse_engineer_from_features(features_dict)

        st.write("**Reverse Engineered Original Inputs:**")
        st.json(original_inputs)


def feature_matching_page(xgb_model):
    st.header("🔧 Feature Engineering Validation")

    st.write("Enter values to see how they're transformed into features")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Values")
        grade = st.selectbox("Grade", list(range(1, 13)), index=6)
        age = st.number_input("Age", min_value=5, max_value=20, value=12)
        gender = st.selectbox("Gender", ["Male", "Female"])
        caste_category = st.selectbox(
            "Caste Category", ["General", "SC", "ST", "OBC"])
        attendance_rate = st.slider("Attendance Rate (%)", 0.0, 100.0, 85.0)
        grade_performance = st.slider(
            "Grade Performance (%)", 0.0, 100.0, 75.0)
        father_education = st.selectbox("Father's Education",
                                        ["No Education", "Primary", "Secondary",
                                         "Higher Secondary", "Graduate", "Post Graduate"])
        family_income = st.selectbox("Monthly Family Income (₹)",
                                     ["< 5000", "5000-15000", "15000-25000",
                                      "25000-50000", "> 50000"])

    with col2:
        distance_to_school = st.number_input(
            "Distance to School (km)", 0.0, 50.0, 2.0)
        rural_urban = st.selectbox("Area Type", ["Rural", "Urban"])
        school_category = st.selectbox("School Category",
                                       ["Primary", "Upper Primary", "Secondary", "Higher Secondary"])
        midday_meal = st.checkbox("Midday Meal", value=True)
        free_uniforms = st.checkbox("Free Uniforms", value=True)
        free_textbooks = st.checkbox("Free Textbooks", value=True)
        internet_access = st.checkbox("Internet Access", value=False)
        medical_checkups = st.checkbox("Medical Checkups", value=True)
        avg_instr_days = st.number_input(
            "Avg Instructional Days", 50, 250, 200)
        student_teacher_ratio = st.number_input(
            "Student-Teacher Ratio", 5.0, 50.0, 20.0)

        # Additional parameters from test data
        girl_ratio = st.number_input(
            "Girl Ratio", 0.0, 1.0, 0.5, format="%.10f")
        female_teacher_ratio = st.number_input(
            "Female Teacher Ratio", 0.0, 1.0, 0.0)
        infrastructure_score = st.number_input(
            "Infrastructure Score (raw)", -2.0, 2.0, 0.0)

    if st.button("Generate Features"):
        # Create student data
        student_data = {
            'grade': grade, 'age': age, 'gender': gender, 'caste_category': caste_category,
            'attendance_rate': attendance_rate, 'grade_performance': grade_performance,
            'father_education': father_education, 'family_income': family_income,
            'distance_to_school': distance_to_school, 'rural_urban': rural_urban,
            'school_category': school_category, 'midday_meal': midday_meal,
            'free_uniforms': free_uniforms, 'free_textbooks': free_textbooks,
            'internet_access': internet_access, 'medical_checkups': medical_checkups,
            'avg_instr_days': avg_instr_days, 'student_teacher_ratio': student_teacher_ratio,
            'girl_ratio': girl_ratio, 'female_teacher_ratio': female_teacher_ratio,
            'infrastructure_score': infrastructure_score
        }

        # Generate features
        features = create_feature_vector_from_ui(student_data)

        # Display features
        st.subheader("Generated Feature Vector")
        feature_names = get_feature_names()

        # Create a dataframe for better display
        features_df = pd.DataFrame({
            'Index': range(33),
            'Feature Name': feature_names,
            'Value': features,
            'Type': ['numeric'] * 17 + ['engineered'] * 5 + ['one-hot'] * 11
        })

        # Highlight engineered features
        def highlight_engineered(row):
            if row['Type'] == 'engineered':
                return ['background-color: yellow'] * len(row)
            elif row['Type'] == 'one-hot':
                return ['background-color: lightblue'] * len(row)
            else:
                return [''] * len(row)

        st.dataframe(features_df.style.apply(highlight_engineered, axis=1))

        # Make prediction
        prediction = xgb_model.predict_proba([features])[0][1]
        st.metric("Dropout Probability", f"{prediction:.1%}")

        # Show engineered feature calculations
        st.subheader("📐 Engineered Feature Calculations")
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Socioeconomic Index**: {features[18]:.2f}
            - Father Education: {father_education}
            - Family Income: {family_income}
            - Caste: {caste_category}
            """)

            st.success(f"""
            **School Support Score**: {features[19]:.2f}
            - Midday Meal: {midday_meal}
            - Free Uniforms: {free_uniforms}
            - Free Textbooks: {free_textbooks}
            """)

        with col2:
            st.warning(f"""
            **Accessibility Score**: {features[20]:.2f}
            - Distance: {distance_to_school} km
            - Formula: 100 * exp(-0.5 * distance)
            """)

            st.error(f"""
            **Infrastructure Score**: {features[17]:.3f}
            - Raw input: {infrastructure_score}
            - Note: Needs RobustScaler from training
            """)


def direct_feature_input_page(xgb_model):
    st.header("🎯 Direct Feature Input")
    st.write("Enter all 33 feature values directly (useful for debugging)")

    feature_names = get_feature_names()

    # Load test data for reference
    test_df = load_test_data()

    if test_df is not None:
        st.info("💡 Tip: You can copy values from test.csv")
        use_test_row = st.checkbox("Use values from test.csv")

        if use_test_row:
            row_idx = st.number_input("Row index", 0, len(test_df)-1, 0)
            default_values = test_df.iloc[row_idx].values
        else:
            default_values = [0.0] * 33
    else:
        default_values = [0.0] * 33

    # Create input fields for all features
    features = []

    # Use columns for better layout
    col1, col2, col3 = st.columns(3)

    for i, feature_name in enumerate(feature_names):
        col = [col1, col2, col3][i % 3]
        with col:
            value = st.number_input(
                f"{i}: {feature_name}",
                value=float(default_values[i]),
                format="%.6f",
                key=f"feature_{i}"
            )
            features.append(value)

    if st.button("Predict with Direct Features"):
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = xgb_model.predict_proba(features_array)[0][1]

        st.success(f"✅ Dropout Probability: {prediction:.1%}")

        # Show feature summary
        st.subheader("Feature Summary")
        summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': features
        })

        # Highlight specific features
        st.write("**Key Features:**")
        key_features = ['attendance_rate', 'grade_performance', 'socioeconomic_index',
                        'expected_age', 'distance_to_school']
        for feat in key_features:
            if feat in feature_names:
                idx = feature_names.index(feat)
                st.write(f"- {feat}: {features[idx]:.2f}")


def main():
    st.markdown('<h1 class="main-header">🔬 Student Dropout Prediction - Diagnostic Mode</h1>',
                unsafe_allow_html=True)

    # Load models
    rf_model, logistic_model, xgb_model = load_models()

    if not xgb_model:
        st.error("XGBoost model not found!")
        return

    # Sidebar navigation
    st.sidebar.title("🧪 Diagnostic Tools")
    page = st.sidebar.selectbox(
        "Choose diagnostic tool:",
        ["Test Data Comparison", "Feature Engineering Validation", "Direct Feature Input"]
    )

    # Important notes in sidebar
    st.sidebar.markdown("---")
    st.sidebar.warning("""
    **⚠️ Known Issues:**
    - Infrastructure score needs RobustScaler
    - Family income mapping may differ
    - One-hot encoding order critical
    """)

    st.sidebar.info("""
    **📊 Expected Ranges:**
    - Attendance: 0-100 (raw)
    - Performance: 0-100 (raw)
    - Socioeconomic: 0-100
    - Infrastructure: ~(-2, 2)
    """)

    # Page routing
    if page == "Test Data Comparison":
        diagnostic_page(xgb_model)
    elif page == "Feature Engineering Validation":
        feature_matching_page(xgb_model)
    elif page == "Direct Feature Input":
        direct_feature_input_page(xgb_model)


if __name__ == "__main__":
    main()
