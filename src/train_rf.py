import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_data, preprocess_data

train_df, _ = load_data('data/train.csv')
# Drop "Id" column
train_df = train_df.drop('Id', axis=1)
X_train, y_train = preprocess_data(train_df)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Random Forest model saved!")