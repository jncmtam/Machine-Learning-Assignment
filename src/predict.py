import pandas as pd
import pickle
from preprocess import preprocess_data

# Load test data
test_df = pd.read_csv('data/test.csv')
test_ids = test_df['Id']

# Load preprocessor and models (reduced features)
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Preprocess test data
X_test = preprocess_data(test_df, is_train=False, preprocessor=preprocessor)

# Make predictions
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Average predictions from both models (optional ensemble)
final_pred = (rf_pred + lr_pred) / 2

# Create submission DataFrame
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_pred
})

# Save predictions
submission.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")