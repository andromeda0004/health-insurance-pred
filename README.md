# 🏥 Health Insurance Cost Predictor

A modern Streamlit web application that predicts health insurance costs using machine learning. This project uses a Linear Regression model with advanced preprocessing to estimate insurance premiums based on personal characteristics.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## ✨ Features

- **Interactive Prediction Interface**: Easy-to-use sidebar for inputting personal information
- **Real-time Cost Estimation**: Instant insurance cost predictions with risk factor analysis
- **Comprehensive Data Analysis**: Visual exploration of the insurance dataset
- **Model Performance Metrics**: Detailed evaluation of prediction accuracy
- **Responsive Design**: Modern UI with custom styling and interactive charts
- **Smart Model Loading**: Automatically loads existing models or trains new ones

## 🎯 Key Capabilities

### 📊 Prediction Features
- Age, sex, BMI, number of children, smoking status, and region inputs
- Risk factor identification and warnings
- Formatted cost predictions with yearly estimates

### 📈 Data Visualization
- Interactive charts using Plotly
- Distribution plots for key variables
- Correlation analysis
- Regional and demographic breakdowns

### 🤖 Model Analytics
- R² score and performance metrics
- Prediction vs actual scatter plots
- Feature importance analysis
- Model validation statistics

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd health-insurance-pred
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501` to access the application

## 📁 Project Structure

```
health-insurance-pred/
├── app.py                              # Main Streamlit application
├── insurance.csv                       # Dataset
├── insurance_pipeline.joblib           # Pre-trained model (optional)
├── Medical_Insurance_Cost_Prediction.ipynb  # Original analysis notebook
├── requirements.txt                    # Python dependencies
└── README.md                          # Project documentation
```

## 🔧 Usage

### Making Predictions

1. **Enter Personal Information**
   - Use the sidebar to input your details
   - Age: 18-100 years
   - BMI: Body Mass Index
   - Children: Number of dependents
   - Sex: Male/Female
   - Smoker: Yes/No
   - Region: Southeast, Southwest, Northeast, Northwest

2. **Get Prediction**
   - Click "💰 Predict Insurance Cost"
   - View your estimated annual premium
   - Review risk factors and recommendations

3. **Explore Data**
   - Navigate to "📊 Data Analysis" tab
   - Explore dataset patterns and distributions
   - Understand feature correlations

4. **Model Performance**
   - Check "📈 Model Performance" tab
   - Review model accuracy metrics
   - Analyze prediction quality

## 🧠 Model Information

### Algorithm
- **Primary Model**: Linear Regression with preprocessing pipeline
- **Preprocessing**: StandardScaler for numerical features, OneHotEncoder for categorical features
- **Performance**: Typically achieves 75-80% R² score

### Features Used
- **Numerical**: Age, BMI, Number of Children
- **Categorical**: Sex, Smoking Status, Region

### Key Insights
- Smoking status is the strongest predictor of insurance costs
- Age and BMI significantly impact premiums
- Regional variations exist but are generally smaller
- Multiple children increase premium costs

## 📊 Dataset

The application uses a health insurance dataset with the following characteristics:
- **Size**: 1,338 records
- **Features**: 6 input features + 1 target (charges)
- **Target Variable**: Insurance charges (in USD)
- **Data Quality**: Clean dataset with no missing values

### Feature Descriptions
| Feature | Type | Description |
|---------|------|-------------|
| age | Numerical | Age of the individual |
| sex | Categorical | Gender (male/female) |
| bmi | Numerical | Body Mass Index |
| children | Numerical | Number of dependents |
| smoker | Categorical | Smoking status (yes/no) |
| region | Categorical | US region (northeast, northwest, southeast, southwest) |
| charges | Numerical | Insurance charges (target variable) |

## 🛠️ Customization

### Adding New Features
To add new prediction features:
1. Update the input form in `app.py`
2. Modify the preprocessing pipeline
3. Retrain the model with new features

### Styling
Custom CSS styling is included in the app. Modify the `st.markdown()` sections to change:
- Color schemes
- Font sizes
- Layout spacing
- Component styling

### Model Improvements
Consider implementing:
- More advanced algorithms (Random Forest, XGBoost)
- Feature engineering
- Cross-validation
- Hyperparameter tuning

## 🔍 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **plotly**: Interactive visualizations
- **joblib**: Model serialization

### Performance Optimization
- `@st.cache_data` for data loading
- `@st.cache_resource` for model loading
- Efficient data processing with pandas
- Lazy loading of visualizations

## ⚠️ Important Notes

### Model Limitations
- This is a demonstration model for educational purposes
- Real insurance pricing involves many more factors
- Actual insurance companies use complex proprietary models
- Results should not be used for actual insurance decisions

### Data Privacy
- No user data is stored or transmitted
- All processing happens locally
- Model predictions are temporary and not logged

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Original dataset from Kaggle
- Streamlit community for excellent documentation
- scikit-learn for machine learning capabilities
- Plotly for interactive visualizations

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section below
2. Review the documentation
3. Create an issue in the repository

## 🔧 Troubleshooting

### Common Issues

**App won't start**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Verify virtual environment is activated

**Model loading errors**
- The app will automatically train a new model if the existing one fails
- Check that `insurance.csv` is in the project directory
- Ensure sufficient memory for model training

**Prediction errors**
- Verify input values are within valid ranges
- Check that all required fields are filled
- Try refreshing the page

**Visualization issues**
- Ensure plotly is installed correctly
- Check browser compatibility
- Clear browser cache if needed

## 🚀 Deployment

### Local Deployment
Follow the Quick Start guide above.

### Cloud Deployment
The app can be deployed on:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web application platform
- **AWS/GCP/Azure**: Cloud platforms

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

---

Made with ❤️ using Streamlit and Python