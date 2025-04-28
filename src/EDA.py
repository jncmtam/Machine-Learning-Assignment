import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Load training data
train_df = pd.read_csv('data/train.csv')

# 1. Histogram of SalePrice
plt.figure(figsize=(10, 6))
sns.histplot(train_df['SalePrice'], kde=True, color='blue')
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.savefig('output/saleprice_histogram.png')
plt.close()

# 2. Correlation Heatmap (Numerical Features)
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = train_df[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('output/correlation_heatmap.png')
plt.close()

# 3. Scatter Plots: Key Numerical Features vs. SalePrice
key_numeric_features = ['LotArea', 'GrLivArea', 'TotalBsmtSF']
for feature in key_numeric_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=train_df[feature], y=train_df['SalePrice'], alpha=0.5)
    plt.title(f'{feature} vs. SalePrice')
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.savefig(f'output/{feature}_vs_saleprice_scatter.png')
    plt.close()

# 4. Box Plots: Key Categorical Features vs. SalePrice
key_categorical_features = ['Neighborhood', 'ExterQual']
for feature in key_categorical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=feature, y='SalePrice', data=train_df)
    plt.title(f'SalePrice Distribution by {feature}')
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.xticks(rotation=45)
    plt.savefig(f'output/saleprice_by_{feature}_boxplot.png')
    plt.close()

print("EDA visualizations saved to output/ folder.")