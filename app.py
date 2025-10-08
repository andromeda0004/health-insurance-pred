import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Health Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the insurance dataset"""
    try:
        data = pd.read_csv('insurance.csv')
        return data
    except FileNotFoundError:
        st.error("Insurance dataset not found. Please ensure 'insurance.csv' is in the project directory.")
        return None

@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one"""
    # Try to load existing model first
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load('insurance_pipeline.joblib')
            
        # Test if model works with a simple prediction
        test_data = pd.DataFrame({
            'age': [30], 'sex': ['male'], 'bmi': [25.0], 
            'children': [0], 'smoker': ['no'], 'region': ['southeast']
        })
        prediction = model.predict(test_data)
        
        return model, None, None, None, None, True
        
    except Exception as e:
        pass
        
    # Train new model if loading fails
    data = load_data()
    if data is None:
        return None, None, None, None, None, False
    
    # Prepare features and target
    X = data.drop('charges', axis=1)
    y = data['charges']
    
    # Define preprocessing for numerical and categorical features
    numerical_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']
    
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create pipeline with preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Calculate model performance
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    # Save the newly trained model
    try:
        joblib.dump(pipeline, 'insurance_pipeline_new.joblib')
    except Exception as e:
        pass
    
    return pipeline, train_score, test_score, X_test, y_test, False

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Health Insurance Cost Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict your health insurance premium using machine learning")
    
    # Sidebar for inputs
    st.sidebar.header("📋 Enter Your Information")
    
    # Load model
    model, train_score, test_score, X_test, y_test, is_loaded = load_or_train_model()
    
    if model is None:
        st.error("Failed to load or train the model. Please check your data file.")
        return
    
    # Input fields
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
    
    with col2:
        sex = st.selectbox("Sex", ["male", "female"])
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Data Analysis", "📈 Model Performance", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Insurance Cost Prediction</h2>', unsafe_allow_html=True)
        
        if st.button("💰 Predict Insurance Cost", type="primary", use_container_width=True):
            try:
                prediction = model.predict(input_data)[0]
                
                # Display prediction result
                st.markdown(f'''
                <div class="prediction-result">
                    <h2>Predicted Insurance Cost</h2>
                    <h1 style="color: #1f77b4; font-size: 3rem;">${prediction:,.2f}</h1>
                    <p style="font-size: 1.2rem;">per year</p>
                </div>
                ''', unsafe_allow_html=True)
                
                # Show input summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Age", f"{age} years")
                    st.metric("BMI", f"{bmi:.1f}")
                with col2:
                    st.metric("Children", children)
                    st.metric("Sex", sex.title())
                with col3:
                    st.metric("Smoker", smoker.title())
                    st.metric("Region", region.title())
                
                # Risk factors analysis
                st.markdown("### 🎯 Risk Factors Analysis")
                risk_factors = []
                
                if age > 50:
                    risk_factors.append("Higher age increases premium")
                if bmi > 30:
                    risk_factors.append("BMI over 30 (obesity) increases premium")
                if smoker == "yes":
                    risk_factors.append("Smoking significantly increases premium")
                if children > 3:
                    risk_factors.append("Multiple dependents increase premium")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.success("✅ Low risk profile - favorable premium rates!")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Dataset Analysis</h2>', unsafe_allow_html=True)
        
        data = load_data()
        if data is not None:
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Average Age", f"{data['age'].mean():.1f}")
            with col3:
                st.metric("Average BMI", f"{data['bmi'].mean():.1f}")
            with col4:
                st.metric("Average Cost", f"${data['charges'].mean():,.0f}")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                fig_age = px.histogram(data, x='age', nbins=20, title='Age Distribution')
                fig_age.update_layout(showlegend=False)
                st.plotly_chart(fig_age, use_container_width=True)
                
                # BMI distribution
                fig_bmi = px.histogram(data, x='bmi', nbins=20, title='BMI Distribution')
                fig_bmi.update_layout(showlegend=False)
                st.plotly_chart(fig_bmi, use_container_width=True)
            
            with col2:
                # Charges by smoker status
                fig_smoker = px.box(data, x='smoker', y='charges', title='Insurance Charges by Smoking Status')
                st.plotly_chart(fig_smoker, use_container_width=True)
                
                # Charges by region
                fig_region = px.box(data, x='region', y='charges', title='Insurance Charges by Region')
                st.plotly_chart(fig_region, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlations")
            # Encode categorical variables for correlation
            data_encoded = data.copy()
            data_encoded['sex'] = data_encoded['sex'].map({'male': 0, 'female': 1})
            data_encoded['smoker'] = data_encoded['smoker'].map({'yes': 1, 'no': 0})
            data_encoded['region'] = pd.Categorical(data_encoded['region']).codes
            
            corr_matrix = data_encoded.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
        
        if train_score is not None and test_score is not None:
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training R² Score", f"{train_score:.3f}")
            with col2:
                st.metric("Testing R² Score", f"{test_score:.3f}")
            with col3:
                overfitting = train_score - test_score
                st.metric("Overfitting Check", f"{overfitting:.3f}", 
                         delta="Good" if overfitting < 0.1 else "Monitor")
            
            # Prediction vs Actual plot
            if X_test is not None and y_test is not None:
                y_pred = model.predict(X_test)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=y_test, y=y_pred,
                    mode='markers',
                    name='Predictions',
                    opacity=0.6
                ))
                fig_pred.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                fig_pred.update_layout(
                    title='Predicted vs Actual Insurance Charges',
                    xaxis_title='Actual Charges',
                    yaxis_title='Predicted Charges',
                    showlegend=True
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Calculate additional metrics
                mse = metrics.mean_squared_error(y_test, y_pred)
                mae = metrics.mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MAE", f"${mae:,.2f}")
                with col2:
                    st.metric("RMSE", f"${rmse:,.2f}")
                with col3:
                    st.metric("MSE", f"${mse:,.2f}")
        else:
            st.info("Model performance metrics not available for pre-loaded model.")
            st.write("The model was loaded from a saved file. To see performance metrics, please retrain the model.")
            
            # Still show a simple validation with current data
            data = load_data()
            if data is not None and model is not None:
                st.subheader("Quick Model Validation")
                
                # Take a sample for validation
                sample_data = data.sample(min(100, len(data)), random_state=42)
                X_sample = sample_data.drop('charges', axis=1)
                y_sample = sample_data['charges']
                
                try:
                    y_pred_sample = model.predict(X_sample)
                    r2_sample = metrics.r2_score(y_sample, y_pred_sample)
                    mae_sample = metrics.mean_absolute_error(y_sample, y_pred_sample)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sample R² Score", f"{r2_sample:.3f}")
                    with col2:
                        st.metric("Sample MAE", f"${mae_sample:,.2f}")
                        
                except Exception as e:
                    st.error(f"Error validating model: {str(e)}")
    
    with tab4:
        st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Purpose
        This application predicts health insurance costs based on personal characteristics using machine learning.
        
        ### 📊 Dataset Features
        - **Age**: Age of the individual
        - **Sex**: Gender (male/female)
        - **BMI**: Body Mass Index (weight/height²)
        - **Children**: Number of dependents
        - **Smoker**: Smoking status (yes/no)
        - **Region**: Geographic region in the US
        
        ### 🤖 Model Information
        - **Algorithm**: Linear Regression
        - **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
        - **Pipeline**: Automated preprocessing and prediction pipeline
        
        ### 💡 Key Insights
        - Smoking status is the strongest predictor of insurance costs
        - Age and BMI also significantly impact premiums
        - Regional differences exist but are generally smaller
        - The model explains approximately 75-80% of the variance in insurance charges
        
        ### ⚠️ Disclaimer
        This is a demonstration model for educational purposes. Actual insurance pricing involves many more factors
        and complex business rules not captured in this simplified model.
        """)
        
        # Model feature importance (simplified for Linear Regression)
        if model is not None:
            st.subheader("🎯 Feature Impact Analysis")
            
            # Get feature names after preprocessing
            feature_names = (
                ['age', 'bmi', 'children'] +  # numerical features
                list(model.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(['sex', 'smoker', 'region']))
            )
            
            # Get coefficients
            coefficients = model.named_steps['regressor'].coef_
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Coefficient': np.abs(coefficients)
            }).sort_values('Abs_Coefficient', ascending=True)
            
            # Plot feature importance
            fig_importance = px.bar(
                importance_df.tail(10), 
                x='Abs_Coefficient', 
                y='Feature',
                orientation='h',
                title='Top 10 Most Important Features (Absolute Coefficients)',
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_importance, use_container_width=True)

if __name__ == "__main__":
    main()