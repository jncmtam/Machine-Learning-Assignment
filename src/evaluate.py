import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from preprocess import preprocess_data


train_df = pd.read_csv('data/train.csv')

with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

X_train, y_train, _ = preprocess_data(train_df, is_train=True, preprocessor=preprocessor)
print(f"Number of features in X_train: {X_train.shape[1]}")

with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_rmse_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
rf_r2_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')

gb_rmse_scores = cross_val_score(gb_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
gb_r2_scores = cross_val_score(gb_model, X_train, y_train, cv=kf, scoring='r2')

print("\nCross-Validation Results (All Features):")
print(f"Random Forest - RMSE: {-rf_rmse_scores.mean():.2f} (+/- {rf_rmse_scores.std() * 2:.2f}), R^2: {rf_r2_scores.mean():.4f} (+/- {rf_r2_scores.std() * 2:.4f})")
print(f"Gradient Boosting - RMSE: {-gb_rmse_scores.mean():.2f} (+/- {gb_rmse_scores.std() * 2:.2f}), R^2: {gb_r2_scores.mean():.4f} (+/- {gb_r2_scores.std() * 2:.4f})")

rf_pred = rf_model.predict(X_train)
rf_rmse = np.sqrt(mean_squared_error(y_train, rf_pred))
rf_r2 = r2_score(y_train, rf_pred)

gb_pred = gb_model.predict(X_train)
gb_rmse = np.sqrt(mean_squared_error(y_train, gb_pred))
gb_r2 = r2_score(y_train, gb_pred)

print("\nPerformance on Full Training Set (All Features):")
print(f"Random Forest - RMSE: {rf_rmse:.2f}, R^2: {rf_r2:.4f}")
print(f"Gradient Boosting - RMSE: {gb_rmse:.2f}, R^2: {gb_r2:.4f}")

feature_names = X_train.columns
rf_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
})
print("\nRandom Forest Feature Importances (Top 10):")
print(rf_importances.sort_values(by='Importance', ascending=False).head(10))

gb_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': gb_model.feature_importances_
})
print("\nGradient Boosting Feature Importances (Top 10):")
print(gb_importances.sort_values(by='Importance', ascending=False).head(10))