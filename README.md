# House Price Prediction using Linear Regression

This project implements a Linear Regression model to predict housing prices based on key features like square footage, number of bedrooms, and number of bathrooms.

# Dataset

We use the House Prices - Advanced Regression Techniques dataset from Kaggle.
Train set: Used for model training and validation.
Test set: Provided by Kaggle for final evaluation.

# Project Structure

├── house_price_prediction.   # Main code (EDA + model + evaluation)

├── requirements.txt                # Python dependencies

├── README.md                       # Project overview

└── gui.py                          # Optional: GUI interface to test predictions

# Features Used

GrLivArea (Above ground living area square feet)
BedroomAbvGr (Number of bedrooms above ground)
FullBath (Number of full bathrooms)
More features can be added easily to improve performance.

# Model

Algorithm: Linear Regression
Framework: scikit-learn

# Metrics:

Validation RMSE: 52,975
Validation R² Score: 0.634
Test R² Score: 0.74 (example output)

# GUI (Optional)

We provide a simple Tkinter-based GUI that allows users to enter square footage, bedrooms, and bathrooms to get a live price prediction.

# Requirements

Install dependencies using:
pip install -r requirements.txt

# Skills Applied

Data Preprocessing
Linear Regression Modeling
Model Evaluation (RMSE, R²)
Data Visualization
GUI Development with Tkinter
Model Saving/Loading (joblib)

# Acknowledgements

Dataset: Kaggle House Prices Challenge
Built as part of my learning journey in Machine Learning.
