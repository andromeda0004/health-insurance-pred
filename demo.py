#!/usr/bin/env python3
"""
Demo script for the Health Insurance Cost Predictor
Shows example predictions for different user profiles
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import from our app
from app import load_data, load_or_train_model

def main():
    print("🏥 Health Insurance Cost Predictor - Demo")
    print("=" * 50)
    
    # Load model
    print("\n📊 Loading model and data...")
    model, *_ = load_or_train_model()
    
    if model is None:
        print("❌ Failed to load model!")
        return
    
    print("✅ Model loaded successfully!")
    
    # Demo profiles
    demo_profiles = [
        {
            'name': 'Young Non-Smoker',
            'age': 25, 'sex': 'male', 'bmi': 22.5, 
            'children': 0, 'smoker': 'no', 'region': 'southeast'
        },
        {
            'name': 'Middle-aged Parent',
            'age': 40, 'sex': 'female', 'bmi': 26.0, 
            'children': 2, 'smoker': 'no', 'region': 'northwest'
        },
        {
            'name': 'Older Smoker',
            'age': 55, 'sex': 'male', 'bmi': 30.5, 
            'children': 1, 'smoker': 'yes', 'region': 'southwest'
        },
        {
            'name': 'Young Smoker',
            'age': 28, 'sex': 'female', 'bmi': 24.0, 
            'children': 0, 'smoker': 'yes', 'region': 'northeast'
        }
    ]
    
    print("\n🎯 Sample Predictions:")
    print("-" * 70)
    
    for profile in demo_profiles:
        name = profile.pop('name')
        
        # Create dataframe for prediction
        input_df = pd.DataFrame([profile])
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            
            print(f"\n👤 {name}")
            print(f"   Age: {profile['age']}, Sex: {profile['sex'].title()}, BMI: {profile['bmi']}")
            print(f"   Children: {profile['children']}, Smoker: {profile['smoker'].title()}, Region: {profile['region'].title()}")
            print(f"   💰 Predicted Cost: ${prediction:,.2f}/year")
            
            # Risk assessment
            if profile['smoker'] == 'yes':
                print(f"   ⚠️  High risk: Smoking significantly increases premiums")
            elif profile['bmi'] > 30:
                print(f"   ⚠️  Moderate risk: BMI over 30 may increase premiums")
            elif profile['age'] > 50:
                print(f"   ⚠️  Moderate risk: Age over 50 may increase premiums")
            else:
                print(f"   ✅ Low risk profile")
                
        except Exception as e:
            print(f"❌ Error predicting for {name}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🚀 To run the full interactive app:")
    print("   streamlit run app.py")
    print("   or")
    print("   ./run_app.sh")

if __name__ == "__main__":
    main()