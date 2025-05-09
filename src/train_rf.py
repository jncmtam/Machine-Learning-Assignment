import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from preprocess import preprocess_data


train_df = pd.read_csv('data/train.csv')


X_train, y_train, preprocessor = preprocess_data(train_df, is_train=True)


model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)


with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("Random Forest model (all features) and preprocessor saved successfully!")