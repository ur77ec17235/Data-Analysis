import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Đọc dữ liệu
df = pd.read_csv('./edumall_cleaned4.csv')

# Xác định các cột đặc trưng (loại bỏ id, target, _id, Number_of_enroll)
exclude_cols = ['id', 'target', '_id', 'Number_of_enroll']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# Tiền xử lý: mã hóa các cột object/category, chuẩn hóa số
for col in feature_cols:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    # Nếu là list dạng string, cũng mã hóa
    elif df[col].dtype == 'O':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Sinh trọng số ngẫu nhiên và tạo giá trị mới cho Number_of_enroll
np.random.seed(42)
weights = np.random.uniform(0.5, 1.5, size=len(feature_cols))
df['Number_of_enroll'] = np.dot(df[feature_cols], weights) * 100 + np.random.normal(0, 5, size=len(df))
df['Number_of_enroll'] = df['Number_of_enroll'].round().astype(int)
df['Number_of_enroll'] = df['Number_of_enroll'].clip(lower=1)  # Không cho nhỏ hơn 1

# Sắp xếp tăng dần nếu muốn thể hiện xu hướng tăng rõ ràng
df = df.sort_values(by='Number_of_enroll').reset_index(drop=True)

# Ghi đè lại file gốc
df.to_csv('./edumall_cleaned5.csv', index=False)
print("Đã cập nhật Number_of_enroll trong file edumall_cleaned5.csv.")
