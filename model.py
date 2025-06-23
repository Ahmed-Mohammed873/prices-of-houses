import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
train_df = pd.read_csv(r'C:\Users\ahmed\Downloads\house-prices-advanced-regression-techniques\train.csv')
test_df = pd.read_csv(r'C:\Users\ahmed\Downloads\house-prices-advanced-regression-techniques\test.csv')
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'
train_df = train_df[features + [target]].dropna()
X = train_df[features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)
print("📊 Validation RMSE:", np.sqrt(mean_squared_error(y_val, y_val_pred)))
print("📈 Validation R² Score:", r2_score(y_val, y_val_pred))
X_test = test_df[features].fillna(0)  
test_preds = model.predict(X_test)

import joblib
joblib.dump(model, 'linear_model.pkl')
