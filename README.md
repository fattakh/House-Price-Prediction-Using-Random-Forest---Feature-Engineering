House Price Prediction Using Machine Learning

This repository contains a complete, production-style machine learning pipeline for house price prediction using synthetic real-estate data. The project demonstrates best practices in data preprocessing, model training, evaluation, and interpretability using Scikit-Learn.

📘 Overview

The project simulates a real-world housing dataset and trains a RandomForestRegressor to estimate property prices based on structural and location-specific features. A unified Scikit-Learn pipeline is used to ensure reproducibility, clean transformations, and modular design.

🔧 Key Features

Synthetic dataset generation (800 samples)

Preprocessing pipeline:

StandardScaler for numerical features

OneHotEncoder for categorical variables

Regression model using Random Forest

Model evaluation with MAE and R²

Feature-importance ranking to identify the top drivers of price

Easy-to-extend, clean, and readable code

📊 Dataset Summary
Feature	Type	Description
bedrooms	Numeric	Number of bedrooms
bathrooms	Numeric	Number of bathrooms
area_sqft	Numeric	Property area (square feet)
location	Categorical	Major housing areas (e.g., DHA, Clifton)
age_years	Numeric	Age of the house
price_pkr	Target	Final house price in PKR

The price is computed using meaningful coefficients and random noise to simulate real-world variability.

🧠 Methodology
1. Preprocessing

A ColumnTransformer is used to:

Scale numeric features

One-hot encode location

2. Modeling

A RandomForestRegressor with 400 estimators is applied.
This model is chosen due to its robustness and ability to capture non-linear relationships.

3. Evaluation Metrics

MAE (Mean Absolute Error) – measures prediction error in PKR

R² Score – explains model variance

Feature importance analysis is included to improve interpretability.

📈 Example Output
MAE (PKR): ~300000
R^2: ~0.90

Top Feature Drivers:
- area_sqft
- bedrooms
- location_DHA
- bathrooms
- location_Clifton

🗂️ Project Structure
│── house_price_model.py
│── README.md

🚀 Getting Started
Install Dependencies
pip install -r requirements.txt


or manually:

pip install numpy pandas scikit-learn

Run the Script
python house_price_model.py

📌 Future Enhancements

Add visualizations (feature distributions, prediction vs actual)

Hyperparameter tuning using GridSearchCV or Optuna

Replace synthetic data with real-world housing datasets

Deploy model using Streamlit or FastAPI
