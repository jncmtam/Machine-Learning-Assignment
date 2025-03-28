import streamlit as st
import pandas as pd
import pickle

# Load trained models
with open("random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("linear_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Load training dataset for feature names
train_df = pd.read_csv("train.csv")
features = train_df.drop(columns=["Id", "SalePrice"]).columns.tolist()

# Streamlit UI
st.title("House Price Prediction")
st.write("Enter the details of your house to get a predicted price.")

# Create user input fields for selected features
input_data = {}
for feature in features:
    if train_df[feature].dtype == 'object':  # Categorical
        options = ["Select"] + train_df[feature].dropna().unique().tolist()
        input_data[feature] = st.selectbox(feature, options)
    else:  # Numerical
        input_data[feature] = st.number_input(feature, value=0)

# Predict button
if st.button("Predict Price"):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    input_df.replace("Select", None, inplace=True)
    
    # Predict using both models
    rf_pred = rf_model.predict(input_df)[0]
    lr_pred = lr_model.predict(input_df)[0]
    
    st.write(f"**Random Forest Prediction:** ${rf_pred:,.2f}")
    st.write(f"**Linear Regression Prediction:** ${lr_pred:,.2f}")
