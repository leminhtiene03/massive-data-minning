import pandas as pd

# BƯỚC 1: Chọn 2 khách hàng làm "diễn viên" cho hôm bảo vệ
# Bạn có thể mở file submission_v2.csv, copy ngẫu nhiên 2 chuỗi ID dán vào đây
DEMO_USERS = [
    '000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ea7ad4d2868fe69f', 
    '00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2'
]

print("Đang tải dữ liệu gốc (Sẽ mất khoảng 1-2 phút vì file 3.4GB)...")
# BƯỚC 2: Đường dẫn tới các file bự của bạn
df_sub = pd.read_csv("data/submission_v2.csv")
df_tx = pd.read_csv("data/transactions_train.csv")

print("Đang lọc dữ liệu cho Demo...")
# BƯỚC 3: Chỉ giữ lại dữ liệu của 2 người này
sub_sample = df_sub[df_sub['customer_id'].isin(DEMO_USERS)]
tx_sample = df_tx[df_tx['customer_id'].isin(DEMO_USERS)]

print("Đang lưu ra file nhỏ...")
# BƯỚC 4: Lưu ra file sample siêu nhẹ
sub_sample.to_csv("data/submission_sample.csv", index=False)
tx_sample.to_csv("data/transactions_sample.csv", index=False)

print("Xong! Bạn đã có file transactions_sample.csv và submission_sample.csv.")