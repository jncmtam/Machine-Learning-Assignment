import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess_data

# Define top 10 features
top_10_features = [
    'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt',
    'FullBath', 'Neighborhood', 'ExterQual', '1stFlrSF', 'BsmtFinSF1'
]

# Load training data
train_df = pd.read_csv('data/train.csv')

# Load the preprocessor used during training
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Preprocess data with top 10 features
X_train, y_train, _ = preprocess_data(train_df, is_train=True, preprocessor=preprocessor, selected_features=top_10_features)
print(f"Number of features in X_train: {X_train.shape[1]}")

# Load models
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Evaluate with top 10 features
rf_pred = rf_model.predict(X_train)
rf_rmse = np.sqrt(mean_squared_error(y_train, rf_pred))
rf_r2 = r2_score(y_train, rf_pred)

# Adjust for log-transformed Linear Regression
lr_pred = lr_model.predict(X_train)
lr_pred = np.expm1(lr_pred)  # Inverse log transformation
y_train_eval = y_train  # Use original SalePrice for metrics
if y_train.min() < 0:  # Check if y_train is log-transformed
    y_train_eval = np.expm1(y_train)

lr_rmse = np.sqrt(mean_squared_error(y_train_eval, lr_pred))
lr_r2 = r2_score(y_train_eval, lr_pred)

print("Performance with top 10 features:")
print(f"Random Forest - RMSE: {rf_rmse:.2f}, R^2: {rf_r2:.4f}")
print(f"Linear Regression - RMSE: {lr_rmse:.2f}, R^2: {lr_r2:.4f}")

# Inspect Linear Regression coefficients
feature_names = X_train.columns
lr_coefs = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_model.coef_
})
print("\nLinear Regression Coefficients (Top 10 by magnitude):")
print(lr_coefs.reindex(lr_coefs['Coefficient'].abs().sort_values(ascending=False).index).head(10))