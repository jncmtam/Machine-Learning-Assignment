import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from preprocess import load_data, preprocess_data
from sklearn.model_selection import train_test_split

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load training data only
train_df, _ = load_data('data/train.csv')
# Drop "Id" column before preprocess
train_df = train_df.drop('Id', axis=1)

# Split into train and validation sets
X, y = preprocess_data(train_df, is_train=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate Linear Regression
print("Evaluating Linear Regression Model:")
lr_model = load_model('models/linear_regression.pkl')
lr_pred = lr_model.predict(X_val)
print("MAE:", mean_absolute_error(y_val, lr_pred))
print("RMSE:", root_mean_squared_error(y_val, lr_pred))
print("R² Score:", r2_score(y_val, lr_pred))
print()

# Evaluate Random Forest
print("Evaluating Random Forest Model:")
rf_model = load_model('models/random_forest.pkl')
rf_pred = rf_model.predict(X_val)
print("MAE:", mean_absolute_error(y_val, rf_pred))
print("RMSE:", root_mean_squared_error(y_val, rf_pred))
print("R² Score:", r2_score(y_val, rf_pred))

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_val)), y_val, color='blue', label='True SalePrice', alpha=0.5)
plt.scatter(range(len(lr_pred)), lr_pred, color='orange', label='Linear Regression Predictions', alpha=0.5)
plt.scatter(range(len(rf_pred)), rf_pred, color='green', label='Random Forest Predictions', alpha=0.5)
plt.xlabel('Sample Index')
plt.ylabel('SalePrice')
plt.title('True SalePrice vs Model Predictions')
plt.legend()
plt.grid(True)
plt.savefig('output/model_comparison.png')
plt.show()
print("Comparison plot saved to output/model_comparison.png")