import os
import gc
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path

# Vô hiệu hóa log cảnh báo khó chịu của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.decomposition import PCA

# Import từ file config của bạn
from src.config import DATA_DIR
from src.data_loader import load_data

def generate_resnet_pca_embeddings(images_dir, articles_df, batch_size=256):
    """
    Dùng ResNet50 trích xuất đặc trưng ảnh, sau đó dùng PCA nén xuống 32 chiều
    và lưu thành file image_features_32d.parquet.
    """
    out_path = DATA_DIR / 'image_features_32d.parquet'
    if out_path.exists():
        print(f'[{datetime.now().strftime("%H:%M:%S")}] File 32d đã tồn tại. Bỏ qua chạy lại.')
        return

    print(f'[{datetime.now().strftime("%H:%M:%S")}] Khởi tạo mô hình ResNet50 (Pre-trained ImageNet)...')
    # Load ResNet50 bỏ đi lớp phân loại cuối, dùng Average Pooling để ra vector 2048 chiều
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    article_ids = articles_df['article_id'].unique().to_list()
    valid_article_ids = []
    features_list = []
    missing_count = 0

    print(f'[{datetime.now().strftime("%H:%M:%S")}] Bắt đầu quét ảnh (Tổng: {len(article_ids):,} sản phẩm)...')
    
    batch_imgs = []
    batch_ids = []

    # Hàm xử lý từng mẻ (batch) để GPU/CPU chạy nhanh hơn
    def process_batch(b_imgs, b_ids):
        batch_tensor = np.vstack(b_imgs)
        # ResNet50 trích xuất ra vector 2048 chiều
        features = base_model.predict(batch_tensor, batch_size=len(b_imgs), verbose=0)
        features_list.extend(features)
        valid_article_ids.extend(b_ids)

    for idx, aid in enumerate(article_ids):
        # Bộ dữ liệu H&M thường chia folder theo 3 số đầu của article_id
        folder_name = str(aid)[:3]
        img_path = os.path.join(images_dir, folder_name, f"{aid}.jpg")
        
        # Nếu bạn gộp tất cả ảnh vào 1 thư mục không chia folder thì dùng dòng này:
        # img_path = os.path.join(images_dir, f"{aid}.jpg")

        try:
            # Load và resize ảnh về 224x224 chuẩn của ResNet
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x) # Chuẩn hóa màu sắc

            batch_imgs.append(x)
            batch_ids.append(aid)

            # Đủ 1 mẻ (batch) thì đem đi dự đoán
            if len(batch_imgs) >= batch_size:
                process_batch(batch_imgs, batch_ids)
                batch_imgs, batch_ids = [], []

            if idx % 10000 == 0 and idx > 0:
                print(f'  -> Đã quét: {idx:,} / {len(article_ids):,}')

        except Exception:
            missing_count += 1

    # Xử lý mẻ cuối cùng còn sót lại
    if len(batch_imgs) > 0:
        process_batch(batch_imgs, batch_ids)

    print(f'[{datetime.now().strftime("%H:%M:%S")}] Trích xuất ResNet xong. Ảnh hợp lệ: {len(valid_article_ids):,}. Ảnh lỗi/thiếu: {missing_count:,}')
    
    # Giải phóng RAM mô hình Deep Learning
    del base_model, batch_imgs, batch_ids
    gc.collect()

    # ---------------------------------------------------------
    # CHẠY PCA NÉN DỮ LIỆU
    # ---------------------------------------------------------
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Bắt đầu chạy PCA để nén từ 2048 chiều xuống 32 chiều...')
    features_array = np.array(features_list)
    
    pca = PCA(n_components=32, random_state=42)
    features_32d = pca.fit_transform(features_array)

    # ---------------------------------------------------------
    # LƯU RA FILE PARQUET CHO SYSTEM
    # ---------------------------------------------------------
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tạo bảng Polars và lưu xuống ổ cứng...')
    
    # Tạo tên cột chuẩn xác: img_feat_0 -> img_feat_31
    cols = [f'img_feat_{i}' for i in range(32)]
    
    df_feats = pl.DataFrame(features_32d, schema=cols, orient="row")
    df_ids = pl.DataFrame({'article_id': valid_article_ids})

    # Gộp cột ID và cột Feature lại
    final_df = pl.concat([df_ids, df_feats], how="horizontal")

    # Lưu thẳng ra file parquet
    final_df.write_parquet(out_path)
    
    print(f'[{datetime.now().strftime("%H:%M:%S")}] HOÀN TẤT! Đã lưu file tại: {out_path}')
    print(f'Tỉ lệ phương sai giữ lại được (Explained Variance): {np.sum(pca.explained_variance_ratio_)*100:.2f}%')


if __name__ == "__main__":
    # 1. Nạp dữ liệu (Chỉ cần lấy danh sách sản phẩm)
    _, _, _, articles, _ = load_data()
    
    # 2. Cấu hình đường dẫn tới thư mục chứa file ảnh gốc của H&M
    # BẠN CẦN SỬA DÒNG NÀY THEO MÁY CỦA BẠN:
    IMAGES_DIR = '/content/drive/MyDrive/inputs2/images' 
    
    # 3. Chạy dây chuyền
    generate_resnet_pca_embeddings(images_dir=IMAGES_DIR, articles_df=articles)