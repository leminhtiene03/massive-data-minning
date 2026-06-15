import streamlit as st
import pandas as pd
import os
from PIL import Image

# Cấu hình trang Web
st.set_page_config(page_title="H&M Recommender System", layout="wide")
st.title("Hệ thống Gợi ý Thời trang Đa phương thức (H&M V2)")
st.markdown("---")

# 1. Hàm Load Dữ liệu
@st.cache_data
def load_data():
    transactions = pd.read_csv("data/transactions_sample.csv")
    articles = pd.read_csv("data/articles.csv")
    submission = pd.read_csv("data/submission_sample.csv")
    return transactions, articles, submission

# Hàm lấy đường dẫn ảnh (Chỉ thẳng vào thư mục images)
def get_image_path(article_id):
    article_str = str(article_id).zfill(10)
    return f"images/{article_str}.jpg"

# Tải dữ liệu vào bộ nhớ
transactions, articles, submission = load_data()

# 2. Tạo thanh Sidebar để nhập User ID
st.sidebar.header("Tùy chỉnh Demo")
user_id = st.sidebar.text_input("Nhập Customer ID:", value="00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2")

if user_id in submission['customer_id'].values:
    # --- PHẦN 1: LỊCH SỬ MUA HÀNG ---
    st.subheader("Lịch sử Mua hàng (Implicit Feedback)")
    
    user_history = transactions[transactions['customer_id'] == user_id].tail(5)
    
    if not user_history.empty:
        cols = st.columns(5)
        for i, row in enumerate(user_history.itertuples()):
            art_id = row.article_id
            art_id_str = str(art_id).zfill(10)
            img_path = get_image_path(art_id)
            
            art_name = articles[articles['article_id'] == art_id]['prod_name'].values[0]
            
            with cols[i % 5]:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                else:
                    # Báo lỗi thông minh nếu thiếu ảnh
                    st.info(f"Đang thiếu ảnh:\n{art_id_str}.jpg")
                    
                st.caption(f"Mã: `{art_id_str}`")
                st.caption(f"**{art_name}**")
    else:
        st.info("Khách hàng này là User mới (Cold-Start).")

    st.markdown("---")

    # --- PHẦN 2: KẾT QUẢ GỢI Ý TOP 12 ---
    st.subheader("✨ Top 12 Gợi ý Tương lai (LightGBM Ranker)")
    st.markdown("*Mô hình kết hợp Tabular Features (Sales Trend) và Visual Features (PCA 32-dim) để xếp hạng:*")
    
    preds_string = submission[submission['customer_id'] == user_id]['prediction'].values[0]
    top_12_articles = preds_string.split()[:12]
    
    cols12 = st.columns(6)
    for i, art_id_str in enumerate(top_12_articles):
        art_id_str = str(art_id_str).zfill(10)
        art_id = int(art_id_str)
        img_path = get_image_path(art_id)
        
        try:
            art_name = articles[articles['article_id'] == art_id]['prod_name'].values[0]
        except IndexError:
            art_name = "Unknown Product"
            
        with cols12[i % 6]:
            if os.path.exists(img_path):
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            else:
                 # Báo lỗi thông minh nếu thiếu ảnh
                st.warning(f"Đang thiếu ảnh:\n{art_id_str}.jpg")
            
            st.caption(f"Rank #{i+1} | Mã: `{art_id_str}`")
            st.caption(f"*{art_name}*")
else:
    st.warning("Vui lòng nhập một Customer ID hợp lệ để xem demo.")