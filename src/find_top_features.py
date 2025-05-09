import pandas as pd
import pickle
import numpy as np


train_df = pd.read_csv('data/train.csv')


with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)


with open('models/gb_model.pkl', 'rb') as f:
    model = pickle.load(f)


X = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']


print("NaN values before transformation:")
print(X.isna().sum()[X.isna().sum() > 0])


preprocessor.fit(X)


X_transformed = preprocessor.transform(X)


print("NaN values after transformation:")
print(pd.DataFrame(X_transformed).isna().sum()[pd.DataFrame(X_transformed).isna().sum() > 0])


print("NaN values in y:")
print(y.isna().sum())




feature_names = preprocessor.named_steps['col_trans'].get_feature_names_out()


importances = model.coef_  


print(f"Length of feature_names: {len(feature_names)}")
print(f"Length of importances: {len(importances)}")
print(f"Feature names: {feature_names}")
print(f"Importances: {importances}")


if len(feature_names) != len(importances):
    raise ValueError(f"Length mismatch: feature_names ({len(feature_names)}) and importances ({len(importances)}) must have the same length")


feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(importances)  
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
important_of_features = feature_importance_df.head(37)


onehot_categories = preprocessor.named_steps['col_trans'].named_transformers_['nom'].categories_
print("Neighborhood categories:", onehot_categories[0])
print("ExterQual categories:", onehot_categories[1])

print("\nTop 10 features:")
print(important_of_features)