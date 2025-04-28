import streamlit as st
import pandas as pd
import pickle
import numpy as np
from preprocess import preprocess_user_input

# Load preprocessor and models
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

# Load training data to get feature information
train_df = pd.read_csv('data/train.csv')

# Define the top 10 important features
top_10_features = [
    'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt',
    'FullBath', 'Neighborhood', 'ExterQual', '1stFlrSF', 'BsmtFinSF1'
]

# Streamlit UI
st.title("Công cụ dự đoán giá nhà")
st.write("Nhập thông tin ngôi nhà để nhận dự đoán giá (chỉ cần nhập 10 đặc trưng quan trọng):")

# Create input fields for the top 10 features
input_data = {}
for feature in top_10_features:
    if train_df[feature].dtype == 'object':  # Categorical feature
        options = train_df[feature].fillna('missing').unique().tolist()
        input_data[feature] = st.selectbox(f"{feature} (ví dụ: {options[0]})", options, index=0)
    else:  # Numerical feature
        median_val = float(train_df[feature].median())
        if feature == 'OverallQual':
            input_data[feature] = st.slider(f"{feature} (Chất lượng tổng thể, 1-10)", 1, 10, int(median_val))
        elif feature == 'YearBuilt':
            input_data[feature] = st.slider(f"{feature} (Năm xây dựng)", 1900, 2025, int(median_val))
        elif feature == 'GarageCars':
            input_data[feature] = st.slider(f"{feature} (Số xe trong garage)", 0, 4, int(median_val))
        elif feature == 'FullBath':
            input_data[feature] = st.slider(f"{feature} (Số phòng tắm đầy đủ)", 0, 4, int(median_val))
        else:
            input_data[feature] = st.number_input(f"{feature} (ví dụ: {int(median_val)})", value=median_val)

# Predict button
if st.button("Dự đoán giá nhà"):
    # Log raw input values
    st.write("### Raw Input Values")
    st.write(input_data)
    
    # Preprocess user input
    input_processed = preprocess_user_input(input_data, top_10_features, preprocessor, train_df)
    
    # Make predictions using both models
    rf_pred = rf_model.predict(input_processed)[0]
    lr_pred = lr_model.predict(input_processed)[0]
    
    # Inverse log transformation for Linear Regression
    lr_pred = np.expm1(lr_pred)
    
    # Clip negative predictions
    rf_pred = max(0, rf_pred)
    lr_pred = max(0, lr_pred)
    
    # Display prediction results
    st.write("### Kết quả dự đoán")
    st.write(f"**Dự đoán từ Random Forest:** ${rf_pred:,.2f}")
    st.write(f"**Dự đoán từ Linear Regression:** ${lr_pred:,.2f}")