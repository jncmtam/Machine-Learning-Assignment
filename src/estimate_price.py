import streamlit as st
import pandas as pd
import pickle
from preprocess import load_data, preprocess_user_input, get_encoders

st.title("House Price Estimator")

# Load data and models
train_df, _ = load_data('data/train.csv')

# Drop 'Id' và 'SalePrice' out of feature trainset
FEATURES = train_df.drop(['Id', 'SalePrice'], axis=1).columns.tolist()
CATEGORICAL = train_df.select_dtypes(include=['object']).columns.tolist()
encoders = get_encoders(train_df, CATEGORICAL)
lr_model = pickle.load(open('models/linear_regression.pkl', 'rb'))
rf_model = pickle.load(open('models/random_forest.pkl', 'rb'))

# Form
st.write("Enter house details (leave blank for defaults):")
user_input = {}
for feature in FEATURES:
    if feature in CATEGORICAL:
        options = [''] + train_df[feature].unique().tolist()
        user_input[feature] = st.selectbox(feature, options, index=0)
    else:
        user_input[feature] = st.number_input(feature, value=None, step=1.0 if 'Area' not in feature else 0.1)

if st.button("Estimate Price"):
    # Filter out empty inputs
    user_input = {k: v for k, v in user_input.items() if v is not None and v != ''}
    if not user_input:
        st.error("Please enter at least one feature.")
    else:
        input_df = preprocess_user_input(user_input, FEATURES, encoders, train_df)
        input_df = input_df.reindex(columns=FEATURES, fill_value=0)
        lr_price = lr_model.predict(input_df)[0]
        rf_price = rf_model.predict(input_df)[0]
        st.success(f"Linear Regression: ${lr_price:,.2f}")
        st.success(f"Random Forest: ${rf_price:,.2f}")