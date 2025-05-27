#!/usr/bin/env python3
"""
Student Dropout Prediction Dashboard - Setup Script
This script helps set up the dashboard environment and validates the setup
"""

import os
import sys
import subprocess
import importlib.util


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def install_requirements():
    """Install required packages"""
    requirements = [
        'streamlit>=1.28.0',
        'plotly>=5.15.0',
        'xgboost>=1.7.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'joblib>=1.3.0'
    ]

    print("📦 Installing required packages...")

    for requirement in requirements:
        try:
            package_name = requirement.split('>=')[0]
            importlib.import_module(package_name.replace('-', '_'))
            print(f"✅ {package_name} already installed")
        except ImportError:
            print(f"📥 Installing {requirement}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", requirement])
            print(f"✅ {package_name} installed successfully")


def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'models',
        'data/processed',
        'data/raw',
        'notebooks',
        'src',
        'results'
    ]

    print("📁 Creating directory structure...")

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")


def check_model_files():
    """Check if model files exist"""
    model_files = [
        '/Users/macbookpro/Desktop/POC1/models/xgboost_model.json',
        '/Users/macbookpro/Desktop/POC1/models/rf_model.joblib',
        '/Users/macbookpro/Desktop/POC1/models/logistic_model.joblib'
    ]

    print("🔍 Checking model files...")

    missing_files = []
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found: {model_file}")
        else:
            print(f"❌ Missing: {model_file}")
            missing_files.append(model_file)

    if missing_files:
        print("\n⚠️  Missing model files detected!")
        print("Please ensure your trained models are saved in the following locations:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nRun your model training pipeline to generate these files.")
        return False

    return True


def create_sample_data():
    """Create sample CSV for testing"""
    sample_data = """student_id,age,grade,attendance_rate,grade_performance,gender,caste_category,father_education,mother_education,family_income,distance_to_school,midday_meal,free_uniforms,free_textbooks
STU001,12,7,85.5,75.2,Male,General,Secondary,Primary,15000-25000,2.5,True,True,True
STU002,15,10,92.0,88.1,Female,SC,Graduate,Secondary,25000-50000,1.2,True,True,True
STU003,10,5,78.3,65.4,Male,OBC,Primary,No Education,5000-15000,4.8,False,True,True
STU004,14,9,95.7,91.3,Female,General,Graduate,Higher Secondary,> 50000,0.8,True,True,True
STU005,13,8,67.2,58.9,Male,ST,No Education,Primary,< 5000,6.2,True,True,False
STU006,11,6,88.4,79.7,Female,OBC,Secondary,Secondary,15000-25000,3.1,True,True,True
STU007,16,11,74.6,82.1,Male,SC,Primary,Primary,5000-15000,2.9,True,False,True
STU008,9,4,91.8,87.5,Female,General,Higher Secondary,Graduate,25000-50000,1.7,True,True,True
STU009,17,12,69.3,73.2,Male,ST,Secondary,No Education,< 5000,8.4,False,True,True
STU010,12,7,83.1,76.8,Female,General,Graduate,Secondary,25000-50000,2.2,True,True,True"""

    if not os.path.exists('data/sample_students.csv'):
        with open('data/sample_students.csv', 'w') as f:
            f.write(sample_data)
        print("✅ Created sample data file: data/sample_students.csv")
    else:
        print("📄 Sample data file already exists")


def validate_dashboard():
    """Validate dashboard can be imported"""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
        import xgboost as xgb
        import pandas as pd
        import numpy as np
        import joblib
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main setup function"""
    print("🎓 Student Dropout Prediction Dashboard - Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return

    # Install requirements
    install_requirements()

    # Create directories
    create_directory_structure()

    # Create sample data
    create_sample_data()

    # Validate imports
    if not validate_dashboard():
        print("❌ Setup validation failed")
        return

    # Check model files
    models_ready = check_model_files()

    print("\n" + "=" * 50)
    print("🎯 Setup Summary")
    print("=" * 50)

    if models_ready:
        print("✅ Setup completed successfully!")
        print("🚀 You can now run the dashboard with:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Setup completed with warnings")
        print("📝 Action required:")
        print("   1. Train your models using your existing pipeline")
        print("   2. Ensure model files are saved in the models/ directory")
        print("   3. Run 'streamlit run app.py' once models are ready")

    print("\n📋 Next Steps:")
    print("   - Review the README.md for detailed usage instructions")
    print("   - Test individual predictions with sample data")
    print("   - Upload data/sample_students.csv for batch analysis testing")
    print("   - Customize intervention recommendations as needed")

    print("\n🎉 Happy predicting!")


if __name__ == "__main__":
    main()
