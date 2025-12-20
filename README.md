# 🌍 World Happiness Report 2021 – Machine Learning Project

## 🧠 Project Overview

Happiness is increasingly recognized as a critical indicator of social progress and human development, extending beyond traditional economic measures such as GDP. The World Happiness Report provides a comprehensive, data-driven view of how people across different countries evaluate their quality of life, based on economic, social, and institutional factors.

This project leverages the World Happiness Report 2021 dataset to build a machine learning–based predictive system capable of estimating a country’s Happiness (Ladder) Score using key socio-economic indicators. The goal is not only to predict happiness scores accurately but also to understand the relative importance of different factors influencing well-being.

The project follows a complete end-to-end machine learning workflow, including data preprocessing, feature selection, model training, evaluation, explainability, and deployment through an interactive web application.

## 🔍 Problem Statement

Given a set of socio-economic indicators such as income level, social support, health, freedom of choice, generosity, and perceptions of corruption, can we:

1. Accurately predict a country’s happiness score, and

2. Identify which factors contribute most significantly to overall happiness?

This problem is well-suited for supervised regression models, particularly tree-based ensemble methods, due to the non-linear and interacting nature of socio-economic variables.

## 📂 Project Structure
<pre>
  World_Happiness_report_2021-ML-model/
│
├── app.py                     # Streamlit application
├── generate_report.py         # Model evaluation & report generator
├── world-happiness-report-2021.csv
├── models/
│   └── happiness_model.pkl    # Trained model artifact
│
├── feature_importance.png     # Feature importance plot
├── model_value_report.md      # Auto-generated model report
├── requirements.txt
└── README.md

</pre>

## 📊 Dataset Information

Source: World Happiness Report 2021

Target Variable:

- Ladder score (Happiness score)

Features Used:

- Logged GDP per capita

- Social support

- Healthy life expectancy

- Freedom to make life choices

- Generosity

- Perceptions of corruption

## ⚙️ Machine Learning Pipeline

### 🔹 Data Preprocessing
- Missing value handling  
- Feature selection based on socio-economic relevance  
- Train–test split (80/20)  

### 🔹 Model Used
- **Random Forest Regressor**  
- Handles non-linear relationships effectively  
- Robust to noise and multicollinearity  
- Suitable for tabular socio-economic data  

### 🔹 Evaluation Metrics
- **R² Score**  
- **Mean Squared Error (MSE)**  

## 🤖 Model Performance (Typical)

- R² Score: ~0.75–0.80

- MSE: Low error indicating strong predictive power

Performance details are automatically documented in model_value_report.md.

## 📈 Feature Importance

The model identifies the most influential factors affecting happiness scores, including:

- Logged GDP per capita

- Social support

- Healthy life expectancy

A visual feature importance plot is generated automatically.


## 🌐 Streamlit Web Application
### Key Features

- Interactive UI for entering socio-economic indicators

- Real-time happiness score prediction

- Automatic country identification based on closest data match

- Clean, user-friendly layout

### App Logic

- User enters socio-economic values

- Model predicts happiness score

- App identifies the closest matching country from the dataset

- Result is displayed with contextual interpretation

## 📝 Automated Model Report

The generate_report.py script:

- Evaluates model performance

- Generates feature importance visualization

- Creates a markdown report (model_value_report.md)

This ensures transparency and reproducibility.

## 🧰 Tech Stack
<p align="left"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" height="40"/> &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" height="40"/> &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" height="40"/> &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikitlearn-original.svg" height="40"/> &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/streamlit/streamlit-original.svg" height="40"/> </p>

## 🚀 How to Run the Project
```bash
git clone https://github.com/your-username/World_Happiness_report_2021-ML-model.git
cd World_Happiness_report_2021-ML-model
streamlit run app.py
python generate_report.py
```

## 👤 Author

Arnab Ghosh


