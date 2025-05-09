Dưới đây là phần trình bày bằng tiếng Việt về công thức của Gradient Boosting và cách nó hoạt động, được thiết kế theo định dạng slide phù hợp cho bài thuyết trình của bạn.

---

### Slide 1: Công thức của Gradient Boosting

#### Tổng quan

- **Thuật toán**: Gradient Boosting cho bài toán hồi quy.
- **Mục tiêu**: Dự đoán giá trị liên tục (ví dụ: giá nhà) bằng cách tối ưu hóa hàm mất mát.
- **Ý tưởng cốt lõi**: Kết hợp tuần tự các mô hình yếu (cây quyết định) để sửa lỗi dự đoán.

#### Mô hình chung

- Gradient Boosting xây dựng mô hình \( F(x) \) là tổng các mô hình yếu:
  $$
  F(x) = F*0(x) + \sum*{m=1}^M \eta \cdot h_m(x)
  $$
  - $ F_0(x) $: Giá trị dự đoán ban đầu (ví dụ: trung bình của giá trị mục tiêu).
  - $ h_m(x) $: Mô hình yếu (cây quyết định) ở lần lặp \( m \).
  - $ \eta $: Tốc độ học (giảm mức đóng góp của mỗi cây).
  - $ M $: Số lần lặp (số cây, tức là `n_estimators`).

---

### Slide 2: Hàm mất mát và tối ưu hóa

#### Hàm mất mát

- Đối với hồi quy, thường sử dụng Sai số Bình phương Trung bình (MSE):
  \[
  L(y, F(x)) = \frac{1}{2} \sum\_{i=1}^n (y_i - F(x_i))^2
  \]
  - \( y_i \): Giá trị thực tế của mẫu \( i \).
  - \( F(x_i) \): Giá trị dự đoán của mẫu \( i \).
  - Hệ số \( \frac{1}{2} \) đơn giản hóa tính gradient.

#### Tối ưu hóa bằng Gradient Descent

- Ở mỗi lần lặp \( m \), Gradient Boosting cập nhật mô hình bằng cách:

  1. **Tính giá trị dư (Residuals)** (gradient âm của hàm mất mát):
     \[
     r*{i,m} = -\frac{\partial L(y_i, F*{m-1}(x*i))}{\partial F*{m-1}(x*i)}
     \]
     Với MSE:
     \[
     r*{i,m} = y*i - F*{m-1}(x_i)
     \]

     - \( F\_{m-1}(x_i) \): Dự đoán từ mô hình trước đó.
     - \( r\_{i,m} \): Giá trị dư (lỗi) của mẫu \( i \).

  2. **Phù hợp với mô hình yếu**: Huấn luyện cây quyết định \( h*m(x) \) để dự đoán \( r*{i,m} \).

  3. **Cập nhật mô hình**:
     \[
     F*m(x) = F*{m-1}(x) + \eta \cdot h_m(x)
     \]
     - \( \eta \): Tốc độ học (ví dụ: 0.1 mặc định, được tinh chỉnh trong mã của bạn thành `[0.01, 0.1]`).

---

### Slide 3: Gradient Boosting hoạt động như thế nào

#### Quy trình từng bước

1. **Khởi tạo mô hình**:

   - Bắt đầu với một giá trị hằng số:
     \[
     F*0(x) = \arg\min*{\gamma} \sum*{i=1}^n L(y_i, \gamma)
     \]
     Với MSE, \( F_0(x) \) là trung bình của mục tiêu: \( F_0(x) = \frac{1}{n} \sum*{i=1}^n y_i \).

2. **Lặp từ \( m = 1 \) đến \( M \)**:

   - Tính giá trị dư \( r*{i,m} = y_i - F*{m-1}(x_i) \).
   - Phù hợp cây quyết định \( h*m(x) \) để dự đoán \( r*{i,m} \).
   - Cập nhật mô hình: \( F*m(x) = F*{m-1}(x) + \eta \cdot h_m(x) \).

3. **Dự đoán cuối cùng**:
   - Sau \( M \) lần lặp, mô hình cuối là:
     \[
     F(x) = F*0(x) + \eta \cdot \sum*{m=1}^M h_m(x)
     \]

#### Cơ chế chính

- Mỗi cây sửa lỗi (giá trị dư) của mô hình trước đó.
- Tốc độ học \( \eta \) kiểm soát bước điều chỉnh, cân bằng giữa tốc độ và độ ổn định.

---

### Slide 4: Triển khai trong mã của bạn (`train_gb.py`)

#### Tham số

- **Tinh chỉnh bằng `GridSearchCV`**:
  - `n_estimators`: [100, 200] (số cây \( M \)).
  - `learning_rate`: [0.01, 0.1] (bước điều chỉnh \( \eta \)).
  - `max_depth`: [3, 5] (độ sâu của mỗi cây \( h_m \)).
- **Giá trị mặc định giữ nguyên**:
  - `loss='squared_error'` (sử dụng MSE làm hàm mất mát).
  - `criterion='friedman_mse'` (cho phân chia trong cây).

#### Quy trình

- **Khởi tạo**: \( F_0(x) \) là trung bình của `SalePrice`.
- **Lặp**:
  - Tính giá trị dư dựa trên MSE.
  - Phù hợp cây với độ sâu `max_depth` đã tinh chỉnh.
  - Cập nhật dự đoán với `learning_rate` đã tinh chỉnh.
- **Kết quả**: Mô hình tốt nhất sau \( M = n_estimators \) lần lặp.

---

### Slide 5: Cách hoạt động trong thực tế

#### Ví dụ dự đoán giá nhà

- **Dự đoán ban đầu**: Trung bình `SalePrice` (ví dụ: 4 triệu VND).
- **Lần lặp 1**:
  - Giá trị dư: \( y_i - 4,000,000 \).
  - Cây 1 dự đoán giá trị dư (độ sâu=3).
  - Cập nhật: \( F_1(x) = 4,000,000 + 0.1 \cdot \text{Cây}\_1(x) \).
- **Lần lặp 2**:
  - Giá trị dư mới dựa trên \( F_1(x) \).
  - Cây 2 dự đoán giá trị dư mới.
  - Cập nhật: \( F_2(x) = F_1(x) + 0.1 \cdot \text{Cây}\_2(x) \).
- **Tiếp tục**: Lặp lại với 100 hoặc 200 cây (dựa trên `n_estimators`).

#### Điểm mạnh

- Bắt được các mẫu phức tạp bằng cách tập trung vào lỗi.
- Có thể tinh chỉnh qua `learning_rate` và `max_depth` (như trong mã của bạn).

---

### Slide 6: Kết luận

- **Công thức**: \( F(x) = F*0(x) + \sum*{m=1}^M \eta \cdot h_m(x) \), tối ưu hóa MSE.
- **Cơ chế**: Lặp lại việc phù hợp cây với giá trị dư, cập nhật dự đoán.
- **Triển khai của bạn**:
  - Tinh chỉnh các tham số chính để tối ưu hóa hiệu suất.
  - Sử dụng hàm mất mát MSE, phù hợp với lý thuyết.
- **Tác động**: Gradient Boosting đạt độ chính xác cao nhưng cần tinh chỉnh cẩn thận để tránh overfitting.

---

### Ghi chú cho thuyết trình

- Sử dụng sơ đồ để minh họa quy trình lặp (ví dụ: giá trị dư → phù hợp cây → cập nhật).
- Giản hóa công thức cho khán giả; tập trung vào trực giác (sửa lỗi từng bước).
- Nhấn mạnh cách mã của bạn (`GridSearchCV`) phù hợp với lý thuyết bằng cách tinh chỉnh \( \eta \) (`learning_rate`) và \( M \) (`n_estimators`).
- Giữ slide ngắn gọn; giải thích ví dụ bằng lời để làm rõ quy trình.

Hãy cho tôi biết nếu bạn cần điều chỉnh thêm!
