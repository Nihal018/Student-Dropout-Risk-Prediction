import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .very-low { background-color: #d4edda; color: #155724; }
    .low { background-color: #d1ecf1; color: #0c5460; }
    .medium { background-color: #fff3cd; color: #856404; }
    .high { background-color: #f8d7da; color: #721c24; }
    .very-high { background-color: #f5c6cb; color: #491217; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for preprocessing objects
if 'robust_scaler' not in st.session_state:
    st.session_state.robust_scaler = None


@st.cache_data
def load_models():
    """Load trained models and preprocessing objects"""
    try:
        rf_model = joblib.load('models/rf_model.joblib')
        logistic_model = joblib.load('models/logistic_model.joblib')
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('models/xgboost_model.json')

        # Try to load preprocessing objects if they exist
        try:
            robust_scaler = joblib.load('models/robust_scaler.joblib')
            st.session_state.robust_scaler = robust_scaler
        except:
            st.warning("RobustScaler not found. Using approximate values.")

        return rf_model, logistic_model, xgb_model
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure models are saved in the 'models/' directory.")
        return None, None, None


def get_risk_tier(probability):
    """Convert probability to risk tier based on bins: 0,20,40,60,80,100"""
    prob_percent = probability * 100
    if prob_percent < 20:
        return "Very Low", "very-low"
    elif prob_percent < 40:
        return "Low", "low"
    elif prob_percent < 60:
        return "Medium", "medium"
    elif prob_percent < 80:
        return "High", "high"
    else:
        return "Very High", "very-high"


def create_risk_visualization(probability):
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Dropout Risk Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "#d4edda"},
                {'range': [20, 40], 'color': "#d1ecf1"},
                {'range': [40, 60], 'color': "#fff3cd"},
                {'range': [60, 80], 'color': "#f8d7da"},
                {'range': [80, 100], 'color': "#f5c6cb"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def get_intervention_recommendations(risk_tier, features):
    """Provide intervention recommendations based on risk tier"""
    recommendations = {
        "Very High": [
            "🚨 **Immediate Intervention Required**",
            "• Assign dedicated counselor/mentor",
            "• Provide financial assistance/scholarships",
            "• Arrange transportation support if distance is high",
            "• Implement personalized learning plan",
            "• Regular family engagement sessions",
            "• Monitor daily attendance"
        ],
        "High": [
            "⚠️ **Urgent Support Needed**",
            "• Weekly counseling sessions",
            "• Academic tutoring support",
            "• Family counseling if needed",
            "• Track attendance weekly",
            "• Provide learning materials/resources"
        ],
        "Medium": [
            "📊 **Regular Monitoring**",
            "• Monthly check-ins with teachers",
            "• Peer support groups",
            "• Study skills workshops",
            "• Monitor academic performance trends"
        ],
        "Low": [
            "✅ **Preventive Measures**",
            "• Quarterly progress reviews",
            "• Motivational programs",
            "• Career guidance sessions"
        ],
        "Very Low": [
            "🌟 **Maintain Current Support**",
            "• Continue regular education programs",
            "• Peer mentoring opportunities",
            "• Leadership development programs"
        ]
    }
    return recommendations.get(risk_tier, ["No specific recommendations available"])


def calculate_socioeconomic_index(father_education, family_income, caste_category):
    """Calculate socioeconomic index matching training data logic"""
    # Map income to numeric values
    income_map = {
        '< 5000': 1,
        '5000-15000': 2,
        '15000-25000': 4,
        '25000-50000': 5,
        '> 50000': 6
    }

    # Map education to numeric values
    education_map = {
        'No Education': 1,
        'Primary': 2,
        'Secondary': 3,
        'Higher Secondary': 4,
        'Graduate': 5,
        'Post Graduate': 6
    }

    # Map caste to numeric values
    caste_map = {
        'ST': 1,
        'SC': 2,
        'OBC': 3,
        'General': 4
    }

    # Get scores
    income_score = income_map.get(family_income, 1)
    education_score = education_map.get(father_education, 1)
    caste_score = caste_map.get(caste_category, 1)

    # Normalize to 0-1 scale
    income_score_norm = (income_score - 1) / (6 - 1)
    education_score_norm = (education_score - 1) / (6 - 1)
    caste_score_norm = (caste_score - 1) / (4 - 1)

    # Weighted combination
    socioeconomic_index = (0.5 * income_score_norm +
                           0.15 * education_score_norm +
                           0.35 * caste_score_norm) * 100

    return socioeconomic_index


def calculate_school_support_score(midday_meal, free_uniforms, free_textbooks):
    """Calculate school support score"""
    support_sum = int(midday_meal) + int(free_uniforms) + int(free_textbooks)
    school_support_score = (support_sum / 3) * 100
    return school_support_score


def calculate_accessibility_score(distance_to_school):
    """Calculate accessibility score using exponential decay"""
    decay_rate = 0.5
    accessibility_score = 100 * np.exp(-decay_rate * distance_to_school)
    return accessibility_score


def calculate_infrastructure_score(raw_score):
    """Apply RobustScaler transformation to infrastructure score"""
    if st.session_state.robust_scaler is not None:
        # Use the actual scaler from training
        return st.session_state.robust_scaler.transform([[raw_score]])[0][0]
    else:
        median = 0
        iqr = 0.4
        return (raw_score - median) / iqr


def create_feature_vector(student_data):
    """Create feature vector exactly matching training data format (33 features)"""
    features = []

    # Extract values from student_data dictionary
    grade = student_data['grade']
    age = student_data['age']
    attendance_rate = student_data['attendance_rate']
    grade_performance = student_data['grade_performance']
    gender = student_data['gender']
    caste_category = student_data['caste_category']
    father_education = student_data['father_education']
    family_income = student_data['family_income']
    distance_to_school = student_data['distance_to_school']
    rural_urban = student_data['rural_urban']
    school_category = student_data['school_category']
    midday_meal = student_data['midday_meal']
    free_uniforms = student_data['free_uniforms']
    free_textbooks = student_data['free_textbooks']
    internet_access = student_data['internet_access']
    medical_checkups = student_data['medical_checkups']
    avg_instr_days = student_data['avg_instr_days']
    student_teacher_ratio = student_data['student_teacher_ratio']
    infrastructure_score_raw = student_data['infrastructure_score']

    # Get additional school characteristics from student_data or use defaults
    girl_ratio = student_data.get('girl_ratio', 0.48)
    female_teacher_ratio = student_data.get('female_teacher_ratio', 0.5)
    school_type = student_data.get('school_type', 3)

    # Features 0-8: Basic features
    features.extend([
        grade,                              # 0: grade
        age,                               # 1: age
        # 2: attendance_rate (raw 0-100 scale)
        attendance_rate,
        # 3: grade_performance (raw 0-100 scale)
        grade_performance,
        int(midday_meal),                  # 4: midday_meal_access
        int(free_textbooks),               # 5: free_text_books_access
        int(free_uniforms),                # 6: free_uniform_access
        int(internet_access),              # 7: internet_access_home
        distance_to_school                 # 8: distance_to_school
    ])

    # Features 9-17: School features
    school_category_map = {"Primary": 1, "Upper Primary": 2,
                           "Secondary": 3, "Higher Secondary": 4}

    features.extend([
        1 if rural_urban == "Rural" else 0,              # 9: rural_urban_y
        school_category_map.get(school_category, 1),     # 10: school_category
        school_type,                                      # 11: school_type
        avg_instr_days,                                   # 12: avg_instr_days
        # 13: medical_checkups
        int(medical_checkups),
        student_teacher_ratio,                            # 14: student_teacher_ratio
        girl_ratio,                                       # 15: girl_ratio
        female_teacher_ratio,                             # 16: female_teacher_ratio
        # 17: infrastructure_score (scaled)
        calculate_infrastructure_score(infrastructure_score_raw)
    ])

    # Features 18-21: Engineered features
    socioeconomic_index = calculate_socioeconomic_index(
        father_education, family_income, caste_category)
    school_support_score = calculate_school_support_score(
        midday_meal, free_uniforms, free_textbooks)
    accessibility_score = calculate_accessibility_score(distance_to_school)
    expected_age = grade + 5

    features.extend([
        socioeconomic_index,               # 18: socioeconomic_index
        school_support_score,              # 19: school_support_score
        accessibility_score,               # 20: accessibility_score
        expected_age                       # 21: expected_age
    ])

    # Features 22-32: One-hot encoded categorical features
    features.extend([
        1 if gender == "Male" else 0,                        # 22: gender_Male
        1 if caste_category == "OBC" else 0,                 # 23: caste_OBC
        1 if caste_category == "SC" else 0,                  # 24: caste_SC
        1 if caste_category == "ST" else 0,                  # 25: caste_ST
        # 26: father_education_HigherSecondary
        1 if father_education == "Higher Secondary" else 0,
        # 27: father_education_PostGraduate
        1 if father_education == "Post Graduate" else 0,
        # 28: father_education_Primary
        1 if father_education == "Primary" else 0,
        # 29: father_education_Secondary
        1 if father_education == "Secondary" else 0,
        # 30: family_income_> ₹5 Lakhs
        1 if family_income == "> 50000" else 0,
        # 31: family_income_₹2 - ₹3.5 Lakhs
        1 if family_income == "25000-50000" else 0,
        # 32: family_income_₹3.5 - ₹5 Lakhs
        1 if family_income == "15000-25000" else 0
    ])

    return features


def individual_prediction_page(rf_model, logistic_model, xgb_model):
    st.header("🎓 Individual Student Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Student Demographics")
        age = st.number_input("Age", min_value=5, max_value=50, value=12)
        grade = st.selectbox("Grade", list(range(1, 13)), index=6)
        gender = st.selectbox("Gender", ["Male", "Female"])
        caste_category = st.selectbox(
            "Caste Category", ["General", "SC", "ST", "OBC"])

        st.subheader("📚 Academic Performance")
        attendance_rate = st.slider(
            "Attendance Rate (%)", 0.0, 100.0, 85.0, 0.1)
        grade_performance = st.slider(
            "Grade Performance (%)", 0.0, 100.0, 75.0, 0.1)

        st.subheader("👨‍👩‍👧‍👦 Family Background")
        father_education = st.selectbox("Father's Education",
                                        ["Primary", "Secondary",
                                         "Higher Secondary", "Graduate", "Post Graduate"])
        family_income = st.selectbox("Monthly Family Income (₹)",
                                     ["< 5000", "5000-15000", "15000-25000",
                                      "25000-50000", "> 50000"])

    with col2:
        st.subheader("🏫 School Infrastructure")
        distance_to_school = st.number_input("Distance to School (km)",
                                             min_value=0.0, max_value=50.0, value=2.0, step=0.1)
        rural_urban = st.selectbox("Area Type", ["Rural", "Urban"])
        school_category = st.selectbox("School Category",
                                       ["Primary", "Upper Primary", "Secondary", "Higher Secondary"])

        st.subheader("🍽️ School Facilities")
        midday_meal = st.checkbox("Midday Meal Available", value=True)
        free_uniforms = st.checkbox("Free Uniforms Provided", value=True)
        free_textbooks = st.checkbox("Free Textbooks Provided", value=True)
        internet_access = st.checkbox("Internet Access at Home", value=False)
        medical_checkups = st.checkbox("Regular Medical Checkups", value=True)

        st.subheader("📈 School Characteristics")
        avg_instr_days = st.number_input("Average Instructional Days/Year",
                                         min_value=100, max_value=300, value=200)
        student_teacher_ratio = st.number_input("Student-Teacher Ratio",
                                                min_value=2.0, max_value=300.0, value=20.0, step=0.1)
        infrastructure_score = st.slider("School Infrastructure Score",
                                         0.0, 4.0, 0.7, 0.01,
                                         help="0 = Poor, 1 = Excellent")

    # Make prediction button
    if st.button("🔮 Predict Dropout Risk", type="primary"):
        # Create student data dictionary
        student_data = {
            'age': age,
            'grade': grade,
            'gender': gender,
            'caste_category': caste_category,
            'attendance_rate': attendance_rate,
            'grade_performance': grade_performance,
            'father_education': father_education,
            'family_income': family_income,
            'distance_to_school': distance_to_school,
            'rural_urban': rural_urban,
            'school_category': school_category,
            'midday_meal': midday_meal,
            'free_uniforms': free_uniforms,
            'free_textbooks': free_textbooks,
            'internet_access': internet_access,
            'medical_checkups': medical_checkups,
            'avg_instr_days': avg_instr_days,
            'student_teacher_ratio': student_teacher_ratio,
            'infrastructure_score': infrastructure_score
        }

        try:
            # Create feature vector
            features = create_feature_vector(student_data)
            features_array = np.array(features).reshape(1, -1)

            # Get predictions from all models
            xgb_prob = xgb_model.predict_proba(features_array)[0][1]
            rf_prob = rf_model.predict_proba(features_array)[0][1]
            lr_prob = logistic_model.predict_proba(features_array)[0][1]

            # Use XGBoost as primary model
            primary_prob = xgb_prob
            risk_tier, risk_class = get_risk_tier(primary_prob)

            # Display results
            st.markdown("---")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.plotly_chart(create_risk_visualization(
                    primary_prob), use_container_width=True)

            with col2:
                st.markdown(f'<div class="risk-card {risk_class}">Risk Level: {risk_tier}</div>',
                            unsafe_allow_html=True)

                st.markdown("### 📊 Model Predictions")
                st.metric("XGBoost (Primary)", f"{xgb_prob:.1%}")
                st.metric("Random Forest", f"{rf_prob:.1%}")
                st.metric("Logistic Regression", f"{lr_prob:.1%}")

            with col3:
                st.markdown("### 💡 Intervention Recommendations")
                recommendations = get_intervention_recommendations(
                    risk_tier, features)
                for rec in recommendations:
                    st.markdown(rec)

            # Show debug information in expander
            with st.expander("🔍 Debug Information (for developers)"):
                st.write(f"**Feature vector length:** {len(features)}")
                st.write("**Calculated engineered features:**")
                st.write(f"- Socioeconomic Index: {features[18]:.2f}")
                st.write(f"- School Support Score: {features[19]:.2f}")
                st.write(f"- Accessibility Score: {features[20]:.2f}")
                st.write(
                    f"- Infrastructure Score (scaled): {features[17]:.3f}")
                st.write(f"- Expected Age: {features[21]}")

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.error("Please check your inputs and try again.")


def batch_analysis_page(xgb_model):
    st.header("📊 Batch Analysis")
    st.write("Upload a CSV file with student data for batch prediction")

    # Sample data format helper
    with st.expander("📋 Expected CSV Format"):
        sample_data = pd.DataFrame({
            'student_id': ['STU001', 'STU002', 'STU003'],
            'age': [12, 15, 10],
            'grade': [7, 10, 5],
            'attendance_rate': [85.5, 92.0, 78.3],
            'grade_performance': [75.2, 88.1, 65.4],
            'gender': ['Male', 'Female', 'Male'],
            'caste_category': ['General', 'SC', 'OBC'],
            'father_education': ['Secondary', 'Graduate', 'Primary'],
            'family_income': ['15000-25000', '25000-50000', '5000-15000'],
            'distance_to_school': [2.5, 1.2, 4.8],
            'rural_urban': ['Urban', 'Rural', 'Urban'],
            'school_category': ['Secondary', 'Higher Secondary', 'Primary'],
            'midday_meal': [True, True, False],
            'free_uniforms': [True, True, True],
            'free_textbooks': [True, True, True],
            'internet_access': [False, True, False],
            'medical_checkups': [True, True, True],
            'avg_instr_days': [200, 220, 180],
            'student_teacher_ratio': [20.0, 15.0, 25.0],
            'infrastructure_score': [0.7, 0.8, 0.6]
        })
        st.dataframe(sample_data)

        csv = sample_data.to_csv(index=False)
        st.download_button(
            "⬇️ Download Sample CSV Template",
            csv,
            "sample_student_data.csv",
            "text/csv"
        )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(
                f"✅ File uploaded successfully! Found {len(df)} students")

            # Display data preview
            st.subheader("📄 Data Preview")
            st.dataframe(df.head(10))

            # Data validation
            required_columns = ['age', 'grade',
                                'attendance_rate', 'grade_performance']
            missing_columns = [
                col for col in required_columns if col not in df.columns]

            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                return

            if st.button("🚀 Run Batch Analysis", type="primary"):
                with st.spinner("Processing predictions..."):
                    predictions = []
                    risk_tiers = []

                    progress_bar = st.progress(0)

                    for idx, row in df.iterrows():
                        try:
                            # Create student data dictionary with defaults for missing values
                            student_data = {
                                'age': row.get('age', 12),
                                'grade': row.get('grade', 7),
                                'gender': row.get('gender', 'Male'),
                                'caste_category': row.get('caste_category', 'General'),
                                'attendance_rate': row.get('attendance_rate', 85),
                                'grade_performance': row.get('grade_performance', 75),
                                'father_education': row.get('father_education', 'Secondary'),
                                'family_income': row.get('family_income', '15000-25000'),
                                'distance_to_school': row.get('distance_to_school', 2.0),
                                'rural_urban': row.get('rural_urban', 'Urban'),
                                'school_category': row.get('school_category', 'Secondary'),
                                'midday_meal': row.get('midday_meal', True),
                                'free_uniforms': row.get('free_uniforms', True),
                                'free_textbooks': row.get('free_textbooks', True),
                                'internet_access': row.get('internet_access', False),
                                'medical_checkups': row.get('medical_checkups', True),
                                'avg_instr_days': row.get('avg_instr_days', 200),
                                'student_teacher_ratio': row.get('student_teacher_ratio', 20.0),
                                'infrastructure_score': row.get('infrastructure_score', 0.7)
                            }

                            features = create_feature_vector(student_data)
                            prob = xgb_model.predict_proba([features])[0][1]
                            risk_tier, _ = get_risk_tier(prob)

                            predictions.append(prob)
                            risk_tiers.append(risk_tier)

                        except Exception as e:
                            st.warning(
                                f"Error processing student {idx+1}: {str(e)}")
                            predictions.append(0.0)
                            risk_tiers.append("Unknown")

                        progress_bar.progress((idx + 1) / len(df))

                    # Add predictions to dataframe
                    df['dropout_probability'] = predictions
                    df['risk_tier'] = risk_tiers
                    df['dropout_risk_percentage'] = (
                        np.array(predictions) * 100).round(1)

                # Display results
                st.success("✅ Batch analysis completed!")

                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    high_risk_count = sum(1 for tier in risk_tiers if tier in [
                                          'High', 'Very High'])
                    st.metric("High Risk Students", high_risk_count,
                              f"{high_risk_count/len(df)*100:.1f}% of total")

                with col2:
                    avg_risk = np.mean(predictions) * 100
                    st.metric("Average Risk", f"{avg_risk:.1f}%")

                with col3:
                    very_high_risk = sum(
                        1 for tier in risk_tiers if tier == 'Very High')
                    st.metric("Critical Cases", very_high_risk)

                with col4:
                    safe_students = sum(1 for tier in risk_tiers if tier in [
                                        'Very Low', 'Low'])
                    st.metric("Low Risk Students", safe_students,
                              f"{safe_students/len(df)*100:.1f}% of total")

                # Risk distribution visualization
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Risk Distribution")
                    risk_counts = pd.Series(risk_tiers).value_counts().reindex(
                        ['Very Low', 'Low', 'Medium', 'High', 'Very High'], fill_value=0)

                    fig_risk_dist = px.pie(values=risk_counts.values, names=risk_counts.index,
                                           title="Risk Tier Distribution",
                                           color_discrete_sequence=['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#DC143C'])
                    st.plotly_chart(fig_risk_dist, use_container_width=True)

                with col2:
                    st.subheader("📈 Risk Score Distribution")
                    fig_hist = px.histogram(x=predictions, nbins=20,
                                            title="Distribution of Dropout Probabilities",
                                            labels={'x': 'Dropout Probability', 'y': 'Number of Students'})
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Detailed results table
                st.subheader("📋 Detailed Results")

                # Sort by risk level
                df_sorted = df.sort_values(
                    'dropout_probability', ascending=False)

                # Add action priority
                def get_action_priority(risk_tier):
                    priority_map = {
                        'Very High': '🚨 Immediate',
                        'High': '⚠️ Urgent',
                        'Medium': '📊 Monitor',
                        'Low': '✅ Routine',
                        'Very Low': '🌟 Maintain'
                    }
                    return priority_map.get(risk_tier, '❓ Unknown')

                df_sorted['action_priority'] = df_sorted['risk_tier'].apply(
                    get_action_priority)

                # Display results
                st.dataframe(df_sorted, use_container_width=True)

                # Download results
                csv_result = df_sorted.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv_result,
                    f"dropout_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check your CSV format and try again.")


def model_performance_page():
    st.header("📈 Model Performance Comparison")

    # Model performance data
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble'],
        'Accuracy': [70, 76, 78, 72],
        'Precision (Dropout)': [34, 39, 41, 35],
        'Recall (Dropout)': [75, 66, 61, 75],
        'F1-Score': [47, 49, 49, 48]
    }

    df_performance = pd.DataFrame(performance_data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Metrics")
        st.dataframe(df_performance)

        # Confusion Matrix for XGBoost
        st.subheader("XGBoost Confusion Matrix")
        conf_matrix = pd.DataFrame({
            'Predicted No Dropout': [20134, 2025],
            'Predicted Dropout': [4620, 3221]
        }, index=['Actual No Dropout', 'Actual Dropout'])
        st.dataframe(conf_matrix)

    with col2:
        # Create performance comparison chart
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'))

        models = df_performance['Model']
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728']

        fig.add_trace(go.Bar(x=models, y=df_performance['Accuracy'],
                             marker_color=colors, name='Accuracy'), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=df_performance['Precision (Dropout)'],
                             marker_color=colors, name='Precision'), row=1, col=2)
        fig.add_trace(go.Bar(x=models, y=df_performance['Recall (Dropout)'],
                             marker_color=colors, name='Recall'), row=2, col=1)
        fig.add_trace(go.Bar(x=models, y=df_performance['F1-Score'],
                             marker_color=colors, name='F1-Score'), row=2, col=2)

        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.subheader("🎯 Top Features by Importance (XGBoost)")
    feature_importance = {
        'Feature': ['Expected Age', 'Grade', 'Age', 'Grade Performance', 'Attendance Rate',
                    'Distance to School', 'Socioeconomic Index', 'School Support Score'],
        'Importance': [1217.05, 734.05, 104.71, 46.37, 30.84, 25.43, 22.18, 19.87],
        'Description': [
            'Expected age for the grade level',
            'Current grade/class of student',
            'Student age',
            'Academic performance in current grade',
            'Percentage attendance rate',
            'Distance from home to school',
            'Composite socioeconomic status',
            'School facility support level'
        ]
    }

    df_features = pd.DataFrame(feature_importance)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(df_features)

    with col2:
        fig_importance = px.bar(df_features, x='Importance', y='Feature',
                                orientation='h',
                                title="Feature Importance Scores",
                                color='Importance',
                                color_continuous_scale='Blues')
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)


def risk_analytics_page():
    st.header("📊 Risk Tier Analytics")

    # XGBoost risk tier data
    xgb_risk_data = {
        'Risk Tier': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        'Student Population (%)': [66.08, 22.34, 8.19, 2.85, 0.54],
        'Actual Dropout Rate (%)': [7.54, 27.60, 48.05, 67.92, 85.89],
        'Student Count': [19823, 6702, 2458, 854, 163]
    }

    df_xgb_risk = pd.DataFrame(xgb_risk_data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student Distribution Across Risk Tiers")
        fig_dist = px.pie(df_xgb_risk, values='Student Population (%)', names='Risk Tier',
                          title="How Students are Distributed",
                          color_discrete_sequence=['#2E8B57', '#4682B4', '#DAA520', '#CD853F', '#DC143C'])
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.subheader("Dropout Rates by Risk Tier")
        fig_dropout = px.bar(df_xgb_risk, x='Risk Tier', y='Actual Dropout Rate (%)',
                             title="Actual Dropout Rates",
                             color='Actual Dropout Rate (%)',
                             color_continuous_scale='Reds',
                             text='Actual Dropout Rate (%)')
        fig_dropout.update_traces(
            texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_dropout, use_container_width=True)

    # Model comparison
    st.subheader("📈 Risk Tier Performance Across Models")

    comparison_data = {
        'Risk Tier': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
        'XGBoost': [7.54, 27.60, 48.05, 67.92, 85.89],
        'Random Forest': [3.46, 8.04, 19.88, 34.56, 66.76],
        'Logistic Regression': [3.77, 7.35, 17.60, 34.36, 55.89]
    }

    df_comparison = pd.DataFrame(comparison_data)

    fig_comp = go.Figure()
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

    for i, model in enumerate(['XGBoost', 'Random Forest', 'Logistic Regression']):
        fig_comp.add_trace(go.Scatter(
            x=df_comparison['Risk Tier'],
            y=df_comparison[model],
            mode='lines+markers',
            name=model,
            line=dict(width=3, color=colors[i]),
            marker=dict(size=8)
        ))

    fig_comp.update_layout(
        title="Dropout Rate by Risk Tier - Model Comparison",
        xaxis_title="Risk Tier",
        yaxis_title="Dropout Rate (%)",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    # Detailed analytics
    st.subheader("📋 Detailed Risk Tier Analysis")

    # Add calculated metrics
    df_xgb_risk['Students at Risk'] = (
        df_xgb_risk['Student Count'] * df_xgb_risk['Actual Dropout Rate (%)'] / 100).astype(int)
    df_xgb_risk['Intervention Priority'] = [
        'Low', 'Medium', 'High', 'Very High', 'Critical']

    st.dataframe(df_xgb_risk, use_container_width=True)

    # Key insights
    st.subheader("🔍 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        high_risk_total = df_xgb_risk.iloc[-2:]['Student Count'].sum()
        st.metric("Total High-Risk Students",
                  f"{high_risk_total:,}",
                  delta="3.4% of population")

    with col2:
        avg_high_risk_dropout = df_xgb_risk.iloc[-2:
                                                 ]['Actual Dropout Rate (%)'].mean()
        st.metric("Avg High-Risk Dropout Rate",
                  f"{avg_high_risk_dropout:.1f}%",
                  delta="76.9% risk level")

    with col3:
        total_expected_dropouts = df_xgb_risk['Students at Risk'].sum()
        st.metric("Expected Total Dropouts",
                  f"{total_expected_dropouts:,}",
                  delta="Based on model prediction")

    # Recommendations
    st.subheader("💡 Strategic Recommendations")

    recommendations = [
        "🎯 **Focus on Very High & High Risk Tiers**: Only 3.4% of students but 76.9% average dropout rate",
        "📊 **Model Validation**: XGBoost shows clear risk progression (7.5% → 85.9% across tiers)",
        "⚡ **Early Intervention**: 66% of students in Very Low risk - maintain current support systems",
        "🔧 **Resource Allocation**: Medium tier (8.2% of students, 48% dropout rate) needs targeted programs",
        "📈 **Monitoring System**: Track movement between risk tiers over time",
        "🤝 **Stakeholder Engagement**: Share risk tier data with teachers and counselors"
    ]

    for rec in recommendations:
        st.markdown(rec)


def model_testing_page(xgb_model):
    st.header("🧪 Model Testing & Sensitivity Analysis")

    st.write("Test the model with extreme cases to verify it's working correctly")

    # Test 1: Extreme cases
    st.subheader("1️⃣ Extreme Case Testing")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚨 Very High Risk Profile")
        st.markdown("""
        - Grade 1 student aged 15 (severely overage)
        - 30% attendance, 25% performance
        - No school support (meals, books, uniforms)
        - 15km from school, rural area
        - ST caste, father no education, <₹5000 income
        """)

        high_risk_data = {
            'age': 15, 'grade': 1, 'gender': 'Female', 'caste_category': 'ST',
            'attendance_rate': 30.0, 'grade_performance': 25.0,
            'father_education': 'No Education', 'family_income': '< 5000',
            'distance_to_school': 15.0, 'rural_urban': 'Rural',
            'school_category': 'Primary', 'midday_meal': False,
            'free_uniforms': False, 'free_textbooks': False,
            'internet_access': False, 'medical_checkups': False,
            'avg_instr_days': 150, 'student_teacher_ratio': 45.0,
            'infrastructure_score': 0.2
        }

    with col2:
        st.markdown("### ✅ Very Low Risk Profile")
        st.markdown("""
        - Grade 10 student aged 15 (age-appropriate)
        - 95% attendance, 90% performance
        - Full school support
        - 0.5km from school, urban area
        - General caste, father graduate, >₹50000 income
        """)

        low_risk_data = {
            'age': 15, 'grade': 10, 'gender': 'Male', 'caste_category': 'General',
            'attendance_rate': 95.0, 'grade_performance': 90.0,
            'father_education': 'Graduate', 'family_income': '> 50000',
            'distance_to_school': 0.5, 'rural_urban': 'Urban',
            'school_category': 'Higher Secondary', 'midday_meal': True,
            'free_uniforms': True, 'free_textbooks': True,
            'internet_access': True, 'medical_checkups': True,
            'avg_instr_days': 240, 'student_teacher_ratio': 12.0,
            'infrastructure_score': 0.9
        }

    if st.button("🔬 Run Extreme Case Test"):
        try:
            # Create feature vectors
            high_risk_features = create_feature_vector(high_risk_data)
            low_risk_features = create_feature_vector(low_risk_data)

            # Get predictions
            high_risk_prob = xgb_model.predict_proba(
                [high_risk_features])[0][1]
            low_risk_prob = xgb_model.predict_proba([low_risk_features])[0][1]

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("High Risk Student",
                          f"{high_risk_prob:.1%}",
                          delta=f"+{(high_risk_prob - 0.5)*100:.1f}% from 50%")
                high_tier, _ = get_risk_tier(high_risk_prob)
                st.info(f"Risk Tier: **{high_tier}**")

            with col2:
                st.metric("Low Risk Student",
                          f"{low_risk_prob:.1%}",
                          delta=f"{(low_risk_prob - 0.5)*100:.1f}% from 50%")
                low_tier, _ = get_risk_tier(low_risk_prob)
                st.success(f"Risk Tier: **{low_tier}**")

            with col3:
                difference = high_risk_prob - low_risk_prob
                st.metric("Probability Difference",
                          f"{difference:.1%}")

                if difference > 0.5:
                    st.success("✅ Model shows excellent sensitivity!")
                elif difference > 0.3:
                    st.warning("⚠️ Model shows moderate sensitivity")
                else:
                    st.error("❌ Model may have sensitivity issues")

        except Exception as e:
            st.error(f"Error in extreme case test: {str(e)}")

    # Test 2: Feature impact analysis
    st.subheader("2️⃣ Feature Impact Analysis")
    st.write("See how changing individual features affects predictions")

    # Base student profile
    base_student = {
        'age': 12, 'grade': 7, 'gender': 'Male', 'caste_category': 'OBC',
        'attendance_rate': 75.0, 'grade_performance': 70.0,
        'father_education': 'Secondary', 'family_income': '15000-25000',
        'distance_to_school': 3.0, 'rural_urban': 'Rural',
        'school_category': 'Secondary', 'midday_meal': True,
        'free_uniforms': True, 'free_textbooks': True,
        'internet_access': False, 'medical_checkups': True,
        'avg_instr_days': 200, 'student_teacher_ratio': 25.0,
        'infrastructure_score': 0.6
    }

    feature_to_test = st.selectbox(
        "Select feature to analyze:",
        ['attendance_rate', 'grade_performance', 'distance_to_school',
         'age', 'student_teacher_ratio']
    )

    if st.button("📊 Analyze Feature Impact"):
        # Define test ranges
        test_ranges = {
            'attendance_rate': np.linspace(20, 100, 9),
            'grade_performance': np.linspace(20, 100, 9),
            'distance_to_school': np.linspace(0, 20, 9),
            'age': np.linspace(6, 18, 9),
            'student_teacher_ratio': np.linspace(10, 50, 9)
        }

        values = []
        probabilities = []

        for value in test_ranges[feature_to_test]:
            # Create modified student data
            test_student = base_student.copy()
            test_student[feature_to_test] = value

            # Get prediction
            features = create_feature_vector(test_student)
            prob = xgb_model.predict_proba([features])[0][1]

            values.append(value)
            probabilities.append(prob * 100)

        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=values,
            y=probabilities,
            mode='lines+markers',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title=f"Impact of {feature_to_test.replace('_', ' ').title()} on Dropout Risk",
            xaxis_title=feature_to_test.replace('_', ' ').title(),
            yaxis_title="Dropout Probability (%)",
            height=400,
            hovermode='x'
        )

        # Add risk tier backgrounds
        fig.add_hrect(y0=0, y1=20, fillcolor="#d4edda",
                      opacity=0.2, line_width=0)
        fig.add_hrect(y0=20, y1=40, fillcolor="#d1ecf1",
                      opacity=0.2, line_width=0)
        fig.add_hrect(y0=40, y1=60, fillcolor="#fff3cd",
                      opacity=0.2, line_width=0)
        fig.add_hrect(y0=60, y1=80, fillcolor="#f8d7da",
                      opacity=0.2, line_width=0)
        fig.add_hrect(y0=80, y1=100, fillcolor="#f5c6cb",
                      opacity=0.2, line_width=0)

        st.plotly_chart(fig, use_container_width=True)

        # Show insights
        prob_range = max(probabilities) - min(probabilities)
        st.info(f"📊 **Insight**: Changing {feature_to_test} from {min(values):.1f} to {max(values):.1f} "
                f"changes dropout probability by {prob_range:.1f} percentage points")


def main():
    st.markdown('<h1 class="main-header">🎓 Student Dropout Risk Prediction System</h1>',
                unsafe_allow_html=True)

    # Load models
    rf_model, logistic_model, xgb_model = load_models()

    if not all([rf_model, logistic_model, xgb_model]):
        st.error("⚠️ Please ensure all model files are in the 'models/' directory:")
        st.code("""
        models/
        ├── xgboost_model.json
        ├── rf_model.joblib
        ├── logistic_model.joblib
        └── robust_scaler.joblib (optional)
        """)
        st.stop()

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Individual Prediction", "Batch Analysis", "Model Performance",
         "Risk Analytics", "Model Testing"]
    )

    # Add information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Information")
    st.sidebar.info("""
    **Primary Model**: XGBoost
    - Accuracy: 78%
    - Features: 33
    - Training samples: 300,000
    """)

    st.sidebar.markdown("### 🎯 Risk Tiers")
    st.sidebar.markdown("""
    - **Very Low**: 0-20% (7.5% dropout)
    - **Low**: 20-40% (27.6% dropout)
    - **Medium**: 40-60% (48.1% dropout)
    - **High**: 60-80% (67.9% dropout)
    - **Very High**: 80-100% (85.9% dropout)
    """)

    # Page routing
    if page == "Individual Prediction":
        individual_prediction_page(rf_model, logistic_model, xgb_model)
    elif page == "Batch Analysis":
        batch_analysis_page(xgb_model)
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Risk Analytics":
        risk_analytics_page()
    elif page == "Model Testing":
        model_testing_page(xgb_model)


if __name__ == "__main__":
    main()
