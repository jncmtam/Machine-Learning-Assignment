import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from preprocess import load_data, preprocess_data

train_df, _ = load_data('data/train.csv')
# Drop "Id" column
train_df = train_df.drop('Id', axis=1)
X_train, y_train = preprocess_data(train_df)

model = LinearRegression()
model.fit(X_train, y_train)

with open('models/linear_regression.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Linear Regression model saved!")