import streamlit as st
import pandas as pd
import pickle
from preprocess import preprocess_user_input

# Load preprocessor and models
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load training data for reference
train_df = pd.read_csv('data/train.csv')

# Define feature columns
feature_cols = train_df.drop(['Id', 'SalePrice'], axis=1).columns

# Streamlit UI
st.title("House Price Prediction")
st.write("Enter the details of your house to get a predicted price.")

# Create user input fields for selected features
input_data = {}
for feature in feature_cols:
    if train_df[feature].dtype == 'object':  # Categorical
        options = train_df[feature].fillna('missing').unique().tolist()
        input_data[feature] = st.selectbox(feature, options, index=0)
    else:  # Numerical
        median_val = float(train_df[feature].median())
        input_data[feature] = st.number_input(feature, value=median_val)

# Predict button
if st.button("Predict Price"):
    # Preprocess user input
    input_processed = preprocess_user_input(input_data, feature_cols, preprocessor, train_df)
    
    # Predict using both models
    rf_pred = rf_model.predict(input_processed)[0]
    lr_pred = lr_model.predict(input_processed)[0]
    
    st.write(f"**Random Forest Prediction:** ${rf_pred:,.2f}")
    st.write(f"**Linear Regression Prediction:** ${lr_pred:,.2f}")