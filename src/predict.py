import pickle
import pandas as pd
from preprocess import load_data, preprocess_data

def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

test_df, _ = load_data('data/test.csv')
ids = test_df['Id']
# Drop "Id" column before preprocessing
X_test = preprocess_data(test_df.drop('Id', axis=1), is_train=False)

model = load_model('models/random_forest.pkl')  # Hoặc linear_regression.pkl
predictions = model.predict(X_test)

output = pd.DataFrame({'Id': ids, 'SalePrice': predictions})
output.to_csv('output/predictions.csv', index=False)
print("Predictions saved to output/predictions.csv")