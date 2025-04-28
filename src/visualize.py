import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess_data
import os

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Load training data
train_df = pd.read_csv('data/train.csv')

# Preprocess data
X_train, y_train, preprocessor = preprocess_data(train_df, is_train=True)

# Load models
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Make predictions
rf_pred = rf_model.predict(X_train)
lr_pred = lr_model.predict(X_train)

# Compute metrics
rf_rmse = np.sqrt(mean_squared_error(y_train, rf_pred))
rf_r2 = r2_score(y_train, rf_pred)
lr_rmse = np.sqrt(mean_squared_error(y_train, lr_pred))
lr_r2 = r2_score(y_train, lr_pred)

# 1. Predicted vs. Actual Scatter Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, rf_pred, alpha=0.5, color='blue', label='Random Forest')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Random Forest: Predicted vs. Actual')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_train, lr_pred, alpha=0.5, color='green', label='Linear Regression')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Linear Regression: Predicted vs. Actual')
plt.legend()

plt.tight_layout()
plt.savefig('output/predicted_vs_actual.png')
plt.close()

# 2. Residual Plots
rf_residuals = y_train - rf_pred
lr_residuals = y_train - lr_pred

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(rf_residuals, kde=True, color='blue')
plt.title('Random Forest: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(lr_residuals, kde=True, color='green')
plt.title('Linear Regression: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('output/residuals_distribution.png')
plt.close()

# 3. Bar Chart for Metrics Comparison
metrics = pd.DataFrame({
    'Model': ['Random Forest', 'Linear Regression', 'Random Forest', 'Linear Regression'],
    'Metric': ['RMSE', 'RMSE', 'R²', 'R²'],
    'Value': [rf_rmse, lr_rmse, rf_r2, lr_r2]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Value', y='Metric', hue='Model', data=metrics)
plt.title('Model Performance Comparison')
plt.savefig('output/model_comparison.png')
plt.close()

print("Model comparison visualizations saved to output/ folder.")