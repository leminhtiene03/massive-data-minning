import os
from pathlib import Path
from datetime import date

# 1. THIẾT LẬP ĐƯỜNG DẪN (PATHS)
ROOT     = Path(__file__).parent.parent.resolve()  # Đảm bảo đường dẫn tuyệt đối
DATA_DIR = ROOT / 'data'
OUT_DIR  = ROOT / 'outputs'

# Các thư mục con
CAND_DIR = OUT_DIR / 'candidates'
FEAT_DIR = OUT_DIR / 'features'
EMB_DIR  = OUT_DIR / 'embeddings'
MOD_DIR  = OUT_DIR / 'models'
SUB_DIR  = OUT_DIR / 'submissions'
V2_DIR = Path('/content/v2_temp')

# Tạo thư mục nếu chưa tồn tại
for d in [DATA_DIR, OUT_DIR, CAND_DIR, FEAT_DIR, EMB_DIR, MOD_DIR, SUB_DIR, V2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 2. HẰNG SỐ THỜI GIAN (DATES)
# Dùng Python date thay vì pl.date để tránh lỗi
LAST_TRAIN_DATE = date(2020, 9, 15)
TEST_START_DATE = date(2020, 9, 16)
VAL_START_DATE  = date(2020, 9, 9)
VAL_END_DATE    = date(2020, 9, 15)

DATE_7D  = date(2020, 9, 8)
DATE_14D = date(2020, 9, 1)
DATE_30D = date(2020, 8, 16)
DATE_90D = date(2020, 6, 17)

# 3. CÁC ĐẶC TRƯNG CHO MÔ HÌNH (FEATURES)
# 3. CÁC ĐẶC TRƯNG CHO MÔ HÌNH (FEATURES)
FEATURE_COLS_V2 = [
    'age', 'is_club_member', 'fashion_news_freq',
    'n_purchases_7d', 'n_purchases_14d', 'n_purchases_30d',
    'n_unique_articles_30d', 'avg_price_30d', 'days_since_last_purchase',
    'article_popularity_7d', 'article_popularity_14d', 'article_popularity_age_group_14d',
    'days_since_article_last_sold',
    'product_type_no', 'colour_group_code', 'department_no',
    'user_has_bought_before', 'user_bought_same_product_type', 'user_bought_same_department',
    'embedding_similarity',
    'user_item_decay_weight',   
    'sales_trend',              
    'candidate_source_repurchase', 'candidate_source_global_popular',
    'candidate_source_age_popular', 'candidate_source_als',
    'candidate_source_embedding_sim', 'candidate_source_item_item',
    # --- 2 Cột Cross-Features đã thêm ở vòng trước ---
    'user_item_purchase_count', 'user_dept_affinity',
    # --- 3 Cột Advanced Features chuẩn bị chạy ---
    'price_deviation', 'article_age_days', 'autumn_2019_sales'
]