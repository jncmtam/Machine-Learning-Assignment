import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from preprocess import preprocess_data

# Ensure output directory exists
os.makedirs('output/eda_preprocessed', exist_ok=True)

# Load training data
train_df = pd.read_csv('data/train.csv')

# Preprocess data with all features (no feature selection)
X_train, y_train, preprocessor = preprocess_data(train_df, is_train=True)

# Combine preprocessed features and target for analysis
processed_df = X_train.copy()
processed_df['SalePrice'] = y_train

# Define key features for visualization (based on feature importance from previous analysis)
key_numeric_features = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'BsmtFinSF1']
key_categorical_features = ['Neighborhood', 'ExterQual']

# 1. Histogram of SalePrice
plt.figure(figsize=(10, 6))
sns.histplot(y_train, kde=True, color='blue')
plt.title('Distribution of SalePrice (Preprocessed Data)')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.savefig('output/eda_preprocessed/saleprice_histogram.png')
plt.close()

# 2. Correlation Heatmap (Numerical Features)
# Select numerical columns from preprocessed data (excluding one-hot encoded columns)
numeric_cols = [col for col in X_train.columns if not col.startswith('Neighborhood_') and not col.startswith('ExterQual_')]
corr_matrix = processed_df[numeric_cols + ['SalePrice']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features (Preprocessed)')
plt.savefig('output/eda_preprocessed/correlation_heatmap.png')
plt.close()

# 3. Scatter Plots: Key Numerical Features vs. SalePrice
for feature in key_numeric_features:
    if feature in X_train.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_train[feature], y=y_train, alpha=0.5)
        plt.title(f'{feature} (Scaled) vs. SalePrice')
        plt.xlabel(f'{feature} (Scaled)')
        plt.ylabel('SalePrice')
        plt.savefig(f'output/eda_preprocessed/{feature}_vs_saleprice_scatter.png')
        plt.close()

# 4. Box Plots: Key Categorical Features vs. SalePrice
for feature in key_categorical_features:
    # Find one-hot encoded columns for this feature
    encoded_cols = [col for col in X_train.columns if col.startswith(f'{feature}_')]
    if encoded_cols:
        # Create a temporary column indicating the category
        temp_col = f'{feature}_category'
        processed_df[temp_col] = X_train[encoded_cols].idxmax(axis=1).str.replace(f'{feature}_', '')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=temp_col, y=processed_df['SalePrice'], data=processed_df)
        plt.title(f'SalePrice Distribution by {feature} (Post-Encoding)')
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.xticks(rotation=45)
        plt.savefig(f'output/eda_preprocessed/saleprice_by_{feature}_boxplot.png')
        plt.close()
        # Drop temporary column
        processed_df = processed_df.drop(columns=[temp_col])

# 5. Distribution of Key Numerical Features (Post-Scaling)
for feature in key_numeric_features:
    if feature in X_train.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(X_train[feature], kde=True, color='purple')
        plt.title(f'Distribution of {feature} (Scaled)')
        plt.xlabel(f'{feature} (Scaled)')
        plt.ylabel('Frequency')
        plt.savefig(f'output/eda_preprocessed/{feature}_distribution.png')
        plt.close()

# 6. Missing Value Heatmap (Post-Preprocessing)
plt.figure(figsize=(12, 6))
sns.heatmap(X_train.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap (Post-Preprocessing)')
plt.savefig('output/eda_preprocessed/missing_values_heatmap.png')
plt.close()

print("EDA visualizations for preprocessed data saved to output/eda_preprocessed/ folder.")