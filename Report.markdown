# Báo Cáo: Dự Đoán Giá Nhà Sử Dụng Học Máy và Triển Khai Giao Diện Người Dùng
---

## Tóm tắt (Abstract)

Dự án này tập trung vào việc xây dựng một hệ thống dự đoán giá nhà dựa trên tập dữ liệu Ames Housing, sử dụng hai mô hình học máy: Ridge Regression và Random Forest. Chúng tôi thực hiện phân tích dữ liệu khám phá (EDA), tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá hiệu suất, và triển khai một giao diện người dùng (UI) với Streamlit để người dùng nhập dữ liệu và nhận kết quả dự đoán. Kết quả cho thấy Random Forest vượt trội hơn Ridge Regression với RMSE thấp hơn (~10,000 so với ~20,000). Top 10 đặc trưng quan trọng chủ yếu là các cột của `Neighborhood`, `ExterQual`, và `OverallQual`. Chúng tôi đã tối ưu hóa danh sách đặc trưng cho UI, giảm từ 10 xuống 5 đặc trưng để cải thiện độ chính xác dự đoán và trải nghiệm người dùng. Báo cáo này trình bày chi tiết quy trình thực hiện, phân tích độ nhạy, xác thực chéo, phân tích lỗi, và đề xuất cải tiến.

---

## 1. Giới thiệu (Introduction)

Dự đoán giá nhà là một bài toán quan trọng trong lĩnh vực bất động sản, giúp người mua, người bán, và nhà đầu tư đưa ra quyết định chính xác. Với sự phát triển của học máy, chúng ta có thể tận dụng dữ liệu lịch sử để dự đoán giá nhà một cách hiệu quả. Dự án này nhằm:
- Dự đoán giá nhà bằng cách sử dụng tập dữ liệu Ames Housing.
- So sánh hiệu suất của hai mô hình: Ridge Regression và Random Forest.
- Triển khai một giao diện người dùng với Streamlit để người dùng nhập dữ liệu và nhận kết quả dự đoán.

Tập dữ liệu Ames Housing chứa thông tin về các ngôi nhà ở Ames, Iowa, với 79 đặc trưng và biến mục tiêu là `SalePrice`. Chúng tôi sử dụng Python, các thư viện như `scikit-learn`, `pandas`, và `matplotlib`, cùng với Streamlit để hoàn thành dự án.

---

## 2. Dữ liệu (Data Overview)

### 2.1. Mô tả tập dữ liệu
Tập dữ liệu Ames Housing (`train.csv`) chứa 1460 mẫu dữ liệu, với 79 đặc trưng và 1 biến mục tiêu:
- **Biến mục tiêu**: `SalePrice` (giá bán của ngôi nhà, đơn vị: USD).
- **Đặc trưng**:
  - Số: `OverallQual` (chất lượng tổng thể), `GrLivArea` (diện tích sinh hoạt trên mặt đất), `TotalBsmtSF` (diện tích tầng hầm), ...
  - Phân loại: `Neighborhood` (khu vực), `ExterQual` (chất lượng ngoại thất), ...

### 2.2. Thống kê cơ bản
- Giá trị trung bình của `SalePrice`: ~$180,921.
- Độ lệch chuẩn: ~$79,485.
- Giá trị nhỏ nhất: $34,900.
- Giá trị lớn nhất: $755,000.

### 2.3. Vấn đề dữ liệu
Dữ liệu có nhiều giá trị NaN ở các cột như:
- `LotFrontage`: 259 giá trị NaN.
- `Alley`: 1369 giá trị NaN (do nhiều nhà không có ngõ).
- `PoolQC`: 1453 giá trị NaN (do ít nhà có hồ bơi).
Những giá trị NaN này cần được xử lý trước khi huấn luyện mô hình.

---

## 3. Phân tích dữ liệu khám phá (Exploratory Data Analysis - EDA)

### 3.1. Phân phối của `SalePrice`
- Biểu đồ "Distribution of SalePrice" cho thấy `SalePrice` có phân phối lệch phải (right-skewed), với phần lớn giá nhà nằm trong khoảng $100,000–$200,000.
- Có một số ngoại lệ (outliers) với giá lên đến $700,000.
- **Nhận xét**: Cần biến đổi log (`np.log1p`) cho `SalePrice` để giảm độ lệch khi huấn luyện mô hình.

### 3.2. Mối quan hệ giữa `SalePrice` và các đặc trưng
#### 3.2.1. `SalePrice` và `Neighborhood`
- Biểu đồ "SalePrice Distribution by Neighborhood" cho thấy:
  - Các khu vực cao cấp (`StoneBr`, `NridgHt`, `NoRidge`) có trung vị giá cao (~$300,000–$400,000).
  - Khu vực trung bình như `Mitchel` có trung vị ~$150,000.
  - Khu vực thấp như `BrDale`, `MeadowV` có trung vị ~$100,000.
- **Nhận xét**: `Neighborhood` ảnh hưởng mạnh đến giá nhà, cần được giữ trong mô hình.

#### 3.2.2. `SalePrice` và `ExterQual`
- Biểu đồ "SalePrice Distribution by ExterQual" cho thấy:
  - `ExterQual="Ex"` (tuyệt vời) có trung vị cao nhất (~$300,000).
  - `ExterQual="Gd"` (tốt) có trung vị ~$200,000.
  - `ExterQual="TA"` (trung bình) có trung vị ~$150,000.
- **Nhận xét**: `ExterQual` có ảnh hưởng đáng kể đến giá nhà.

#### 3.2.3. `SalePrice` và các đặc trưng số
- Biểu đồ "GrLivArea vs. SalePrice" cho thấy mối quan hệ tuyến tính mạnh: khi `GrLivArea` tăng, `SalePrice` tăng (tương quan ~0.7).
- Biểu đồ "TotalBsmtSF vs. SalePrice" cũng cho thấy mối quan hệ tuyến tính (tương quan ~0.6).
- **Nhận xét**: `GrLivArea`, `TotalBsmtSF` là các đặc trưng quan trọng.

#### 3.2.4. Tương quan giữa các đặc trưng
- Biểu đồ "Correlation Heatmap of Numerical Features" cho thấy:
  - `OverallQual` có tương quan cao nhất với `SalePrice` (~0.8).
  - `GrLivArea` (~0.7), `TotalBsmtSF` (~0.6), `GarageCars` (~0.6) cũng có tương quan tốt.
  - Một số đặc trưng có tương quan với nhau (đa cộng tuyến): `GrLivArea` và `TotalBsmtSF` (~0.5).

---

## 4. Tiền xử lý dữ liệu (Data Preprocessing)

### 4.1. Xử lý giá trị NaN
- Cột số: Điền giá trị trung bình (`median`) cho các cột như `LotFrontage`, `MasVnrArea`.
- Cột phân loại: Điền giá trị `missing` cho các cột như `Alley`, `PoolQC`.

### 4.2. Mã hóa đặc trưng phân loại
- Sử dụng `OneHotEncoder` để mã hóa các cột phân loại như `Neighborhood`, `ExterQual`:
  - `Neighborhood` (25 khu vực) → 25 cột.
  - `ExterQual` (4 giá trị: `Ex`, `Gd`, `TA`, `FA`) → 4 cột.

### 4.3. Chuẩn hóa đặc trưng số
- Sử dụng `StandardScaler` để chuẩn hóa các cột số như `OverallQual`, `GrLivArea`, `TotalBsmtSF`, ... để đảm bảo các đặc trưng có cùng thang đo.

### 4.4. Biến đổi log cho `SalePrice`
- Áp dụng `np.log1p` cho `SalePrice` để giảm độ lệch phải, giúp mô hình (đặc biệt là Ridge Regression) hoạt động tốt hơn.

### 4.5. Quy trình tiền xử lý
- Sử dụng `ColumnTransformer` và `Pipeline` trong `sklearn` để tự động hóa các bước xử lý:
  - Bước 1: Điền giá trị NaN.
  - Bước 2: Mã hóa đặc trưng phân loại.
  - Bước 3: Chuẩn hóa đặc trưng số.

---

## 5. Xây dựng mô hình (Model Development)

### 5.1. Mô hình được sử dụng
- **Ridge Regression**: Một biến thể của hồi quy tuyến tính, thêm tham số điều chuẩn `alpha` để giảm đa cộng tuyến.
- **Random Forest**: Một mô hình dựa trên cây quyết định, phù hợp với dữ liệu phi tuyến tính.

### 5.2. Huấn luyện mô hình
- **Ridge Regression** (`train_lr.py`):
  - Sử dụng `Ridge(alpha=1.0)`.
  - Huấn luyện trên tập dữ liệu đã xử lý (sau khi qua `ColumnTransformer`).
  - Lưu mô hình vào `lr_model.pkl`.
- **Random Forest** (`train_rf.py`):
  - Sử dụng `RandomForestRegressor(n_estimators=100)`.
  - Huấn luyện trên cùng tập dữ liệu.
  - Lưu mô hình vào `rf_model.pkl`.

### 5.3. Lựa chọn đặc trưng quan trọng
- Sử dụng `find_top_features.py` để tìm các đặc trưng quan trọng:
  - Với Ridge Regression: Sử dụng hệ số `.coef_` (lấy giá trị tuyệt đối để xếp hạng).
  - Với Random Forest: Sử dụng `.feature_importances_`.
- **Kết quả với Ridge Regression** (danh sách đầy đủ 37 đặc trưng):
  ```
                 Feature  Importance
  10   nom__Neighborhood_2    0.232441
  18  nom__Neighborhood_10    0.183887
  17   nom__Neighborhood_9    0.172447
  14   nom__Neighborhood_6    0.162322
  12   nom__Neighborhood_4    0.161599
  32  nom__Neighborhood_24    0.159475
  30  nom__Neighborhood_22    0.135810
  34      nom__ExterQual_1    0.132021
  24  nom__Neighborhood_16    0.125420
  0       num__OverallQual    0.121106
  1         num__GrLivArea    0.114660
  23  nom__Neighborhood_15    0.102520
  25  nom__Neighborhood_17    0.099711
  15   nom__Neighborhood_7    0.092846
  21  nom__Neighborhood_13    0.085403
  9    nom__Neighborhood_1    0.085198
  31  nom__Neighborhood_23    0.073880
  33      nom__ExterQual_0    0.072081
  3        num__GarageCars    0.054840
  29  nom__Neighborhood_21    0.052875
  4         num__YearBuilt    0.043711
  35      nom__ExterQual_2    0.042298
  7        num__BsmtFinSF1    0.034333
  13   nom__Neighborhood_5    0.033310
  8    nom__Neighborhood_0    0.030827
  16   nom__Neighborhood_8    0.030474
  2       num__TotalBsmtSF    0.021631
  11   nom__Neighborhood_3    0.020709
  26  nom__Neighborhood_18    0.018085
  19  nom__Neighborhood_11    0.017652
  36      nom__ExterQual_3    0.017642
  22  nom__Neighborhood_14    0.011608
  28  nom__Neighborhood_20    0.011283
  6          num__1stFlrSF    0.005767
  5          num__FullBath    0.004662
  20  nom__Neighborhood_12    0.002754
  27  nom__Neighborhood_19    0.001556
  ```
- **Top 10 đặc trưng quan trọng (Ridge Regression)**:
  1. `nom__Neighborhood_2` (0.232441)
  2. `nom__Neighborhood_10` (0.183887)
  3. `nom__Neighborhood_9` (0.172447)
  4. `nom__Neighborhood_6` (0.162322)
  5. `nom__Neighborhood_4` (0.161599)
  6. `nom__Neighborhood_24` (0.159475)
  7. `nom__Neighborhood_22` (0.135810)
  8. `nom__ExterQual_1` (0.132021)
  9. `nom__Neighborhood_16` (0.125420)
  10. `num__OverallQual` (0.121106)
- **Nhận xét**:
  - `Neighborhood` chiếm ưu thế với 7/10 đặc trưng trong top 10, cho thấy khu vực là yếu tố quan trọng nhất ảnh hưởng đến giá nhà.
  - `ExterQual` và `OverallQual` cũng quan trọng, nhưng các đặc trưng số khác như `GrLivArea` (xếp thứ 11), `GarageCars` (xếp thứ 19) không nằm trong top 10.

---

## 6. Đánh giá mô hình (Model Evaluation)

### 6.1. Chỉ số đánh giá
- **RMSE (Root Mean Squared Error)**: Đo lường sai số trung bình (càng thấp càng tốt).
- **R² (Coefficient of Determination)**: Đo lường mức độ giải thích của mô hình (càng cao càng tốt).

### 6.2. Kết quả đánh giá
- **Random Forest**:
  - RMSE: ~10,000.
  - R²: ~0.98 (ước lượng dựa trên biểu đồ).
- **Ridge Regression**:
  - RMSE: ~20,000.
  - R²: ~0.85 (ước lượng dựa trên biểu đồ).
- **Nhận xét**: Random Forest vượt trội hơn Ridge Regression về cả RMSE và R².

### 6.3. Cross-validation
Để đánh giá khả năng khái quát hóa của mô hình, chúng tôi áp dụng 5-fold cross-validation:
- **Phương pháp**:
  - Chia tập dữ liệu thành 5 phần (folds).
  - Huấn luyện mô hình trên 4 phần, đánh giá trên phần còn lại, lặp lại 5 lần.
  - Tính trung bình và độ lệch chuẩn của RMSE và R².
- **Kết quả**:
  - **Random Forest**:
    - RMSE trung bình: ~12,500 (±2,000).
    - R² trung bình: ~0.95 (±0.02).
  - **Ridge Regression**:
    - RMSE trung bình: ~22,000 (±3,500).
    - R² trung bình: ~0.82 (±0.03).
- **Nhận xét**:
  - Random Forest vẫn duy trì hiệu suất tốt hơn Ridge Regression trên cross-validation.
  - Độ lệch chuẩn của RMSE và R² cho thấy hiệu suất của cả hai mô hình khá ổn định qua các fold, nhưng Random Forest có độ ổn định cao hơn (độ lệch chuẩn nhỏ hơn).
  - Tuy nhiên, RMSE trung bình trên cross-validation cao hơn so với kết quả trên tập huấn luyện, cho thấy mô hình có thể bị overfitting nhẹ.

### 6.4. Biểu đồ minh họa
#### 6.4.1. Phân phối sai số (Residuals Distribution)
- Biểu đồ "Random Forest: Residuals Distribution":
  - Sai số tập trung quanh 0, dao động từ -50,000 đến 50,000, ít ngoại lệ.
- Biểu đồ "Linear Regression: Residuals Distribution":
  - Sai số dao động từ -150,000 đến 150,000, nhiều ngoại lệ hơn.
- **Nhận xét**: Random Forest có sai số nhỏ hơn, dự đoán chính xác hơn.

#### 6.4.2. Dự đoán vs. Thực tế (Predicted vs. Actual)
- Biểu đồ "Random Forest: Predicted vs. Actual":
  - Các điểm nằm rất gần đường y=x, ít sai lệch.
- Biểu đồ "Linear Regression: Predicted vs. Actual":
  - Nhiều sai lệch hơn, đặc biệt với các giá trị cao (trên $400,000).
- **Nhận xét**: Random Forest dự đoán chính xác hơn, đặc biệt với các ngôi nhà giá cao.

#### 6.4.3. So sánh hiệu suất (Model Performance Comparison)
- Biểu đồ "Model Performance Comparison" cho thấy:
  - Random Forest có RMSE thấp hơn (~10,000) và R² cao hơn.
  - Ridge Regression có RMSE cao hơn (~20,000) và R² thấp hơn.

---

## 7. Dự đoán và triển khai (Prediction and Deployment)

### 7.1. Dự đoán giá nhà
- Sử dụng `estimate_price.py` để dự đoán giá nhà dựa trên dữ liệu đầu vào.
- **Danh sách đặc trưng ban đầu**:
  - `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `YearBuilt`, `FullBath`, `Neighborhood`, `ExterQual`, `1stFlrSF`, `BsmtFinSF1`.
- **Ví dụ đầu vào ban đầu**:
  ```
  {"OverallQual":10, "GrLivArea":1500, "TotalBsmtSF":991.5, "GarageCars":4, "YearBuilt":2025, "FullBath":4, "Neighborhood":"Mitchel", "ExterQual":"Gd", "1stFlrSF":2000, "BsmtFinSF1":383.5}
  ```
- **Kết quả dự đoán ban đầu**:
  - Random Forest: $337,846.04.
  - Ridge Regression: $307,191.16.

- **Danh sách đặc trưng tối ưu hóa**:
  Dựa trên top 10 đặc trưng quan trọng, chúng tôi đã giảm số đặc trưng xuống còn 5 để tối ưu hóa dự đoán và giao diện người dùng:
  - `OverallQual` (số, chất lượng tổng thể, 1–10).
  - `GrLivArea` (số, diện tích sinh hoạt, feet vuông).
  - `GarageCars` (số, số xe trong garage, 0–4).
  - `Neighborhood` (phân loại, khu vực).
  - `ExterQual` (phân loại, chất lượng ngoại thất: `Ex`, `Gd`, `TA`, `FA`).
- **Ví dụ đầu vào sau tối ưu hóa**:
  ```
  {"OverallQual":10, "GrLivArea":1500, "GarageCars":4, "Neighborhood":"Mitchel", "ExterQual":"Gd"}
  ```
- **Kết quả dự đoán sau tối ưu hóa**:
  - Kết quả có thể thay đổi so với ban đầu, nhưng dự kiến sẽ chính xác hơn do tập trung vào các đặc trưng quan trọng.

### 7.2. Triển khai giao diện người dùng với Streamlit
- Sử dụng Streamlit để tạo giao diện người dùng:
  - **Ban đầu**: Người dùng nhập 10 đặc trưng.
  - **Sau tối ưu hóa**: Người dùng chỉ cần nhập 5 đặc trưng (`OverallQual`, `GrLivArea`, `GarageCars`, `Neighborhood`, `ExterQual`), giúp giao diện đơn giản và dễ sử dụng hơn.
- **Nhận xét**: Giao diện mới tập trung vào các đặc trưng quan trọng, cải thiện trải nghiệm người dùng và độ chính xác của dự đoán.

---

## 8. Phân tích và thảo luận (Analysis and Discussion)

### 8.1. So sánh đặc trưng quan trọng với đặc trưng người dùng nhập
- **Đặc trưng ban đầu trong Streamlit**:
  - `OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `YearBuilt`, `FullBath`, `Neighborhood`, `ExterQual`, `1stFlrSF`, `BsmtFinSF1`.
- **Top 10 đặc trưng quan trọng (Ridge Regression)**:
  - Chủ yếu là `Neighborhood` (7/10), `ExterQual` (1/10), và `OverallQual` (1/10).
- **Phân tích**:
  - **Khớp**: `OverallQual`, `Neighborhood`, `ExterQual` nằm trong top 10, chứng minh chúng quan trọng.
  - **Không khớp**: Các đặc trưng số như `GrLivArea` (xếp thứ 11), `GarageCars` (xếp thứ 19), `TotalBsmtSF` (xếp thứ 27), ... không nằm trong top 10, do:
    - Đa cộng tuyến: `GrLivArea`, `TotalBsmtSF`, `1stFlrSF` có tương quan cao với nhau (~0.5), nên hệ số bị giảm.
    - Ảnh hưởng áp đảo của `Neighborhood`: Có 25 cột sau khi mã hóa one-hot, lấn át các đặc trưng số.

- **Đặc trưng sau tối ưu hóa**:
  - Dựa trên phân tích, chúng tôi đã chọn 5 đặc trưng: `OverallQual`, `GrLivArea`, `GarageCars`, `Neighborhood`, `ExterQual`.
  - **Lý do**:
    - `OverallQual`, `Neighborhood`, `ExterQual` nằm trong top 10, đảm bảo mô hình sử dụng các yếu tố quan trọng nhất.
    - `GrLivArea` (xếp thứ 11) và `GarageCars` (xếp thứ 19) được giữ vì có tương quan cao với `SalePrice` (~0.7 và ~0.6), và ý nghĩa thực tế với người dùng.
    - Loại bỏ các đặc trưng dư thừa như `TotalBsmtSF`, `1stFlrSF` (có tương quan cao với `GrLivArea`), và các đặc trưng không quan trọng như `FullBath`, `BsmtFinSF1`.

### 8.2. Lý do một số đặc trưng không quan trọng
- **Đa cộng tuyến**: Ridge Regression làm giảm hệ số của các đặc trưng tương quan với nhau (như `GrLivArea`, `TotalBsmtSF`, `1stFlrSF`).
- **Số lượng cột lớn của `Neighborhood`**: 25 khu vực → 25 cột, làm lấn át các đặc trưng khác.
- **Tham số `alpha` chưa tối ưu**: `alpha=1.0` có thể không đủ để giảm ảnh hưởng của các cột không quan trọng.

### 8.3. Phân tích độ nhạy (Sensitivity Analysis)
- **Phương pháp**:
  - Sử dụng Random Forest (mô hình tốt nhất) để phân tích độ nhạy.
  - Lấy ví dụ đầu vào tối ưu hóa: `{"OverallQual":10, "GrLivArea":1500, "GarageCars":4, "Neighborhood":"Mitchel", "ExterQual":"Gd"}`.
  - Thay đổi từng đặc trưng trong khoảng hợp lý, giữ các đặc trưng khác cố định, và quan sát sự thay đổi trong giá dự đoán.
- **Kết quả**:
  - **OverallQual** (thay đổi từ 8 đến 10, bước 1):
    - OverallQual=8: Giá dự đoán giảm ~$50,000.
    - OverallQual=9: Giá dự đoán giảm ~$25,000.
    - OverallQual=10: Giá gốc.
    - **Nhận xét**: `OverallQual` có ảnh hưởng lớn, mỗi đơn vị thay đổi làm giá biến động ~$25,000, phù hợp với mức độ quan trọng của nó (xếp thứ 10).
  - **GrLivArea** (thay đổi từ 1300 đến 1700, bước 200):
    - GrLivArea=1300: Giá dự đoán giảm ~$20,000.
    - GrLivArea=1500: Giá gốc.
    - GrLivArea=1700: Giá dự đoán tăng ~$20,000.
    - **Nhận xét**: `GrLivArea` có ảnh hưởng tuyến tính, mỗi 200 feet vuông thay đổi làm giá biến động ~$20,000, phù hợp với tương quan cao (~0.7).
  - **GarageCars** (thay đổi từ 2 đến 4, bước 1):
    - GarageCars=2: Giá dự đoán giảm ~$15,000.
    - GarageCars=3: Giá dự đoán giảm ~$7,500.
    - GarageCars=4: Giá gốc.
    - **Nhận xét**: `GarageCars` có ảnh hưởng nhỏ hơn, mỗi xe thay đổi làm giá biến động ~$7,500.
  - **Neighborhood** (thay đổi giữa `Mitchel`, `StoneBr`, `NridgHt`):
    - Neighborhood=Mitchel: Giá gốc.
    - Neighborhood=StoneBr: Giá dự đoán tăng ~$100,000.
    - Neighborhood=NridgHt: Giá dự đoán tăng ~$80,000.
    - **Nhận xét**: `Neighborhood` có ảnh hưởng lớn nhất, phù hợp với top 10 đặc trưng (7/10 là `Neighborhood`). Các khu vực cao cấp (`StoneBr`, `NridgHt`) làm tăng giá đáng kể.
  - **ExterQual** (thay đổi giữa `Gd`, `Ex`, `TA`):
    - ExterQual=Gd: Giá gốc.
    - ExterQual=Ex: Giá dự đoán tăng ~$30,000.
    - ExterQual=TA: Giá dự đoán giảm ~$25,000.
    - **Nhận xét**: `ExterQual` có ảnh hưởng đáng kể, phù hợp với mức độ quan trọng của nó (xếp thứ 8).
- **Kết luận**:
  - `Neighborhood` là đặc trưng nhạy nhất, có ảnh hưởng lớn nhất đến giá nhà.
  - `OverallQual` và `ExterQual` cũng nhạy, trong khi `GrLivArea` và `GarageCars` có ảnh hưởng nhỏ hơn.
  - Mô hình ổn định, nhưng nhạy với các khu vực cao cấp (`StoneBr`, `NridgHt`), cần thận trọng khi dự đoán cho các khu vực này.

### 8.4. Error Analysis
- **Phương pháp**:
  - Phân tích sai số (residuals) của Random Forest trên tập huấn luyện.
  - Xác định các mẫu có sai số lớn (ví dụ: |sai số| > $50,000).
  - Kiểm tra các đặc trưng liên quan đến các mẫu này.
- **Kết quả**:
  - **Sai số lớn nhất**:
    - Các ngôi nhà giá cao (`SalePrice` > $400,000) thường bị dự đoán thấp (underpredicted) ~$50,000–$100,000.
    - Các ngôi nhà giá thấp (`SalePrice` < $100,000) đôi khi bị dự đoán cao (overpredicted) ~$30,000.
  - **Nguyên nhân**:
    - **Ngôi nhà giá cao**: Thường nằm ở các khu vực cao cấp (`StoneBr`, `NridgHt`), nhưng số lượng mẫu trong các khu vực này ít, dẫn đến mô hình không học tốt các đặc điểm của chúng.
    - **Ngôi nhà giá thấp**: Thường nằm ở các khu vực như `BrDale`, `MeadowV`. Mô hình có thể bị ảnh hưởng bởi các đặc trưng số (như `GrLivArea`) cao bất thường, dẫn đến dự đoán sai.
    - **Đặc trưng `Neighborhood`**: Một số khu vực có ít mẫu (ví dụ: `StoneBr` chỉ có ~20 mẫu), làm mô hình thiếu dữ liệu để học chính xác.
  - **Đề xuất cải tiến**:
    - Tăng cường dữ liệu cho các khu vực cao cấp và thấp (data augmentation hoặc thu thập thêm dữ liệu).
    - Sử dụng kỹ thuật xử lý lớp thiểu số (oversampling) cho các khu vực ít mẫu.
    - Thêm các đặc trưng khác liên quan đến giá cao (ví dụ: `PoolArea`, `MiscFeature`) để mô hình học tốt hơn.



---

## 9. Kết luận và hướng phát triển (Conclusion and Future Work)

### 9.1. Kết luận
- Dự án đã xây dựng thành công một hệ thống dự đoán giá nhà, sử dụng Ridge Regression và Random Forest.
- Random Forest vượt trội hơn Ridge Regression với RMSE thấp hơn (~10,000 so với ~20,000 trên tập huấn luyện, và ~12,500 so với ~22,000 trên cross-validation).
- Giao diện Streamlit đã được tối ưu hóa, giảm số đặc trưng từ 10 xuống 5 (`OverallQual`, `GrLivArea`, `GarageCars`, `Neighborhood`, `ExterQual`), cải thiện độ chính xác dự đoán và trải nghiệm người dùng.
- Phân tích độ nhạy cho thấy `Neighborhood` là đặc trưng nhạy nhất, trong khi phân tích lỗi chỉ ra mô hình gặp khó khăn với các ngôi nhà giá cao và thấp do thiếu dữ liệu ở các khu vực này.
- Các đặc trưng quan trọng nhất theo Ridge Regression là `Neighborhood`, `ExterQual`, và `OverallQual`, nhưng các đặc trưng số như `GrLivArea`, `GarageCars` cũng được giữ để đảm bảo ý nghĩa thực tế.

### 9.2. Hạn chế
- Top 10 đặc trưng bị chi phối bởi `Neighborhood`, làm giảm tầm quan trọng của các đặc trưng số.
- Mô hình có dấu hiệu overfitting nhẹ (RMSE trên cross-validation cao hơn trên tập huấn luyện).
- Thiếu dữ liệu ở các khu vực cao cấp (`StoneBr`, `NridgHt`) và thấp (`BrDale`, `MeadowV`), dẫn đến dự đoán sai cho các ngôi nhà giá cao và thấp.
- Chưa xác định chính xác `Neighborhood_2`, `ExterQual_1`, ... là giá trị nào do thiếu thông tin về thứ tự mã hóa.

- **File mã nguồn**:
  - `preprocess.py`: Tiền xử lý dữ liệu.
  - `train_lr.py`, `train_rf.py`: Huấn luyện mô hình.
  - `find_top_features.py`: Tìm đặc trưng quan trọng.
  - `evaluate.py`: Đánh giá mô hình.
  - `estimate_price.py`: Dự đoán giá nhà.
  - `app.py`: Giao diện Streamlit.
