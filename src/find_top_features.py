import pandas as pd
import pickle
import numpy as np

# Đọc dữ liệu
train_df = pd.read_csv('data/train.csv')

# Đọc preprocessor
with open('models/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Đọc mô hình Ridge Regression
with open('models/lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Chuẩn bị dữ liệu
X = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']

# Kiểm tra NaN trước khi biến đổi
print("NaN values before transformation:")
print(X.isna().sum()[X.isna().sum() > 0])

# Fit preprocessor với dữ liệu
preprocessor.fit(X)

# Biến đổi dữ liệu
X_transformed = preprocessor.transform(X)

# Kiểm tra NaN sau khi biến đổi
print("NaN values after transformation:")
print(pd.DataFrame(X_transformed).isna().sum()[pd.DataFrame(X_transformed).isna().sum() > 0])

# Kiểm tra NaN trong y
print("NaN values in y:")
print(y.isna().sum())

# Lấy feature names từ bước col_trans trong Pipeline
# Giả định Pipeline có các bước: ['label', 'col_trans', ...]
# Chỉ lấy bước col_trans (ColumnTransformer) để gọi get_feature_names_out
feature_names = preprocessor.named_steps['col_trans'].get_feature_names_out()

# Lấy hệ số từ mô hình Ridge Regression
importances = model.coef_  # Sử dụng .coef_ cho Ridge Regression

# Debug: Kiểm tra độ dài của feature_names và importances
print(f"Length of feature_names: {len(feature_names)}")
print(f"Length of importances: {len(importances)}")
print(f"Feature names: {feature_names}")
print(f"Importances: {importances}")

# Đảm bảo độ dài khớp nhau
if len(feature_names) != len(importances):
    raise ValueError(f"Length mismatch: feature_names ({len(feature_names)}) and importances ({len(importances)}) must have the same length")

# Tạo DataFrame với hệ số tuyệt đối để xác định tầm quan trọng
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(importances)  # Sử dụng giá trị tuyệt đối để xếp hạng
})

# Sắp xếp và hiển thị top 10 đặc trưng
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
important_of_features = feature_importance_df.head(37)

# Lấy danh sách các hạng mục sau khi mã hóa one-hot
onehot_categories = preprocessor.named_steps['col_trans'].named_transformers_['nom'].categories_
print("Neighborhood categories:", onehot_categories[0])
print("ExterQual categories:", onehot_categories[1])
# In kết quả
print("\nTop 10 features:")
print(important_of_features)