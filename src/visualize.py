import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import shap
from preprocess import preprocess_data
import os


sns.set_style('whitegrid')
sns.set_palette("husl")


os.makedirs('output', exist_ok=True)


train_df = pd.read_csv('data/train.csv')


with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)


X_train, y_train, _ = preprocess_data(train_df, is_train=True, preprocessor=preprocessor)


with open('models/gb_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)  


rf_pred = rf_model.predict(X_train)
gb_pred = gb_model.predict(X_train)


rf_rmse = np.sqrt(mean_squared_error(y_train, rf_pred))
rf_mae = mean_absolute_error(y_train, rf_pred)
rf_r2 = r2_score(y_train, rf_pred)
gb_rmse = np.sqrt(mean_squared_error(y_train, gb_pred))
gb_mae = mean_absolute_error(y_train, gb_pred)
gb_r2 = r2_score(y_train, gb_pred)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_train, rf_pred, alpha=0.5, color='blue', label='Random Forest')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title(f'Random Forest: Predicted vs. Actual\nRMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_train, gb_pred, alpha=0.5, color='green', label='Gradient Boosting')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title(f'Gradient Boosting: Predicted vs. Actual\nRMSE: {gb_rmse:.2f}, R²: {gb_r2:.4f}')
plt.legend()

plt.tight_layout()
plt.savefig('output/predicted_vs_actual.png')
plt.close()


rf_residuals = y_train - rf_pred
gb_residuals = y_train - gb_pred

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(rf_residuals, kde=True, color='blue')
plt.title('Random Forest: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(gb_residuals, kde=True, color='green')
plt.title('Gradient Boosting: Residuals Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('output/residuals_distribution.png')
plt.close()


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(rf_pred, rf_residuals, alpha=0.5, color='blue')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted SalePrice')
plt.ylabel('Residuals')
plt.title('Random Forest: Residuals vs. Fitted')

plt.subplot(1, 2, 2)
plt.scatter(gb_pred, gb_residuals, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted SalePrice')
plt.ylabel('Residuals')
plt.title('Gradient Boosting: Residuals vs. Fitted')

plt.tight_layout()
plt.savefig('output/residuals_vs_fitted.png')
plt.close()


def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_sizes, scoring='neg_mean_squared_error'
    )
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.sqrt(train_scores_mean), label='Training RMSE')
    plt.plot(train_sizes, np.sqrt(val_scores_mean), label='Validation RMSE')
    plt.fill_between(train_sizes, np.sqrt(train_scores_mean - train_scores_std),
                     np.sqrt(train_scores_mean + train_scores_std), alpha=0.1)
    plt.fill_between(train_sizes, np.sqrt(val_scores_mean - val_scores_std),
                     np.sqrt(val_scores_mean + val_scores_std), alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

plot_learning_curve(rf_model, 'Random Forest: Learning Curve', X_train, y_train)
plt.savefig('output/rf_learning_curve.png')
plt.close()

plot_learning_curve(gb_model, 'Gradient Boosting: Learning Curve', X_train, y_train)
plt.savefig('output/gb_learning_curve.png')
plt.close()


metrics = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'] * 3,
    'Metric': ['RMSE'] * 2 + ['MAE'] * 2 + ['R²'] * 2,
    'Value': [rf_rmse, gb_rmse, rf_mae, gb_mae, rf_r2, gb_r2]
})

plt.figure(figsize=(10, 6))
sns.barplot(x='Value', y='Metric', hue='Model', data=metrics)
plt.title('Model Performance Comparison (RMSE, MAE, R²)')
plt.savefig('output/error_metrics_comparison.png')
plt.close()


rf_perm_importance = permutation_importance(rf_model, X_train, y_train, n_repeats=10, random_state=42)
gb_perm_importance = permutation_importance(gb_model, X_train, y_train, n_repeats=10, random_state=42)

rf_perm_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False).head(10)

gb_perm_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gb_perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Importance', y='Feature', data=rf_perm_df, color='blue')
plt.title('Random Forest: Top 10 Permutation Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=gb_perm_df, color='green')
plt.title('Gradient Boosting: Top 10 Permutation Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.tight_layout()
plt.savefig('output/permutation_importance.png')
plt.close()


key_features = ['GrLivArea', 'TotalBsmtSF', 'YearBuilt']
for feature in key_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(train_df[feature], kde=True, color='purple')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.savefig(f'output/{feature}_distribution.png')
    plt.close()


plt.figure(figsize=(12, 6))
sns.heatmap(train_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.savefig('output/missing_values_heatmap.png')
plt.close()


for feature in key_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=train_df[feature], color='orange')
    plt.title(f'Boxplot of {feature} for Outlier Detection')
    plt.xlabel(feature)
    plt.savefig(f'output/{feature}_boxplot_outliers.png')
    plt.close()


train_df['AgeOfHouse'] = train_df['YrSold'] - train_df['YearBuilt']
plt.figure(figsize=(8, 6))
sns.scatterplot(x=train_df['AgeOfHouse'], y=train_df['SalePrice'], alpha=0.5, color='teal')
plt.title('Age of House vs. SalePrice')
plt.xlabel('Age of House (YrSold - YearBuilt)')
plt.ylabel('SalePrice')
plt.savefig('output/age_of_house_vs_saleprice.png')
plt.close()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.colorbar(label='SalePrice')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA 2D Projection of Training Data')
plt.savefig('output/pca_2d_plot.png')
plt.close()



explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_train)


explainer_gb = shap.TreeExplainer(gb_model)
shap_values_gb = explainer_gb.shap_values(X_train)


plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_rf, X_train, plot_type="bar", show=False)
plt.title('Random Forest: SHAP Feature Importance')
plt.tight_layout()
plt.savefig('output/rf_shap_summary.png')
plt.close()


plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_gb, X_train, plot_type="bar", show=False)
plt.title('Gradient Boosting: SHAP Feature Importance')
plt.tight_layout()
plt.savefig('output/gb_shap_summary.png')
plt.close()


plt.figure(figsize=(10, 6))
sns.histplot(rf_pred, kde=True, color='blue', label='Random Forest', bins=50, alpha=0.5)
sns.histplot(gb_pred, kde=True, color='green', label='Gradient Boosting', bins=50, alpha=0.5)
plt.title('Distribution of Predicted SalePrice (Training Set)')
plt.xlabel('Predicted SalePrice')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('output/predictions_distribution_train.png')
plt.close()


feature_names = X_train.columns
rf_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)

gb_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': gb_model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Importance', y='Feature', data=rf_importances, color='blue')
plt.title('Random Forest: Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=gb_importances, color='green')
plt.title('Gradient Boosting: Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.tight_layout()
plt.savefig('output/feature_importances.png')
plt.close()

print("Model comparison and data storytelling visualizations saved to output/ folder.")