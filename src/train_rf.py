import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from preprocess import preprocess_data

# Define top 10 features
top_10_features = [
    'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt',
    'FullBath', 'Neighborhood', 'ExterQual', '1stFlrSF', 'BsmtFinSF1'
]

# Load training data
train_df = pd.read_csv('data/train.csv')

# Preprocess data with top 10 features
X_train, y_train, preprocessor = preprocess_data(train_df, is_train=True, selected_features=top_10_features)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessor
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("Random Forest model (top 10 features) and preprocessor saved successfully!")