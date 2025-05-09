import streamlit as st
import pandas as pd
import pickle
import numpy as np
from preprocess import preprocess_user_input


with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('models/rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('models/gb_model.pkl', 'rb') as f:
    gb_model = pickle.load(f)


train_df = pd.read_csv('data/train.csv')


top_10_features = [
    'OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt',
    'FullBath', 'Neighborhood', 'ExterQual', '1stFlrSF', 'BsmtFinSF1'
]


all_features = [col for col in train_df.columns if col not in ['Id', 'SalePrice']]


st.title("Công cụ dự đoán giá nhà")
st.write("Nhập thông tin ngôi nhà để nhận dự đoán giá (chỉ cần nhập 10 đặc trưng quan trọng):")


input_data = {}
for feature in top_10_features:
    if train_df[feature].dtype == 'object':  
        options = train_df[feature].fillna('missing').unique().tolist()
        input_data[feature] = st.selectbox(f"{feature} (ví dụ: {options[0]})", options, index=0)
    else:  
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


for feature in all_features:
    if feature not in input_data:
        if train_df[feature].dtype == 'object':  
            input_data[feature] = 'missing'  
        else:  
            input_data[feature] = train_df[feature].median()  


if st.button("Dự đoán giá nhà"):
    
    st.write("## Raw Input Value")
    st.write({k: input_data[k] for k in top_10_features})  
    
    
    input_processed = preprocess_user_input(input_data, all_features, preprocessor, train_df)
    
    
    rf_pred = rf_model.predict(input_processed)[0]
    gb_pred = gb_model.predict(input_processed)[0]
    
    
    rf_pred = max(0, rf_pred)
    gb_pred = max(0, gb_pred)
    
    
    st.write("## Kết quả dự đoán giá nhà")
    st.write(f"**Dự đoán từ Random Forest:** ${rf_pred:,.2f}")
    st.write(f"**Dự đoán từ Gradient Boosting:** ${gb_pred:,.2f}")