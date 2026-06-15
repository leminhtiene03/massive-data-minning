import gc
import pickle
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from datetime import datetime

from src.config import V2_DIR, MOD_DIR, FEATURE_COLS_V2

def sample_negatives(ratio=20, seed=42): # [ĐÃ SỬA]: Tăng từ 10 lên 20
    """
    Lấy mẫu âm (Negative Sampling) theo từng chunk để cân bằng dữ liệu train.
    """
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Bắt đầu lấy mẫu âm (Ratio 1:{ratio})...')
    chunk_files = sorted(V2_DIR.glob('train_feat_*.parquet'))
    sampled_paths = []

    for i, p in enumerate(chunk_files):
        sp = V2_DIR / f'sampled_{i:04d}.parquet'
        if sp.exists():
            sampled_paths.append(sp)
            continue
            
        chunk = pl.read_parquet(p)
        pos = chunk.filter(pl.col('label') == 1)
        neg = chunk.filter(pl.col('label') == 0)
        
        n_neg = min(len(pos) * ratio, len(neg))
        if n_neg > 0:
            sampled = pl.concat([pos, neg.sample(n=n_neg, seed=seed)])
        else:
            sampled = pos
            
        sampled.write_parquet(sp)
        sampled_paths.append(sp)
        del chunk, pos, neg, sampled; gc.collect()
        
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đã lấy mẫu âm xong cho {len(chunk_files)} chunks.')
    return sampled_paths

def build_lgb_dataset(sampled_paths):
    """
    Gom các chunk đã lấy mẫu thành Dataset chuẩn của LightGBM (cần X, y, và group).
    """
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang build LGBM Train Dataset...')
    X_parts, y_parts, g_parts = [], [], []

    for sp in sampled_paths:
        # Sắp xếp theo customer_id là bắt buộc đối với thuật toán Lambdarank
        chunk = pl.read_parquet(sp).sort(['customer_id', 'label'], descending=[False, True])
        
        # Đảm bảo chỉ lấy những cột feature đã cấu hình
        cols_to_use = [c for c in FEATURE_COLS_V2 if c in chunk.columns]
        
        X_parts.append(chunk.select(cols_to_use).to_pandas())
        y_parts.append(chunk['label'].to_numpy())
        g_parts.append(chunk.group_by('customer_id', maintain_order=True).len()['len'].to_numpy())
        del chunk; gc.collect()

    X_train = pd.concat(X_parts, ignore_index=True)
    y_train = np.concatenate(y_parts)
    g_train = np.concatenate(g_parts)
    del X_parts, y_parts, g_parts; gc.collect()

    lgb_train = lgb.Dataset(X_train, label=y_train, group=g_train, free_raw_data=True)
    lgb_train.construct()
    del X_train, y_train, g_train; gc.collect()
    
    return lgb_train

def train_lgbm_model(lgb_train, lgb_val=None):
    """
    Thiết lập tham số và huấn luyện mô hình xếp hạng.
    """
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Bắt đầu huấn luyện LightGBM...')
    # [ĐÃ SỬA]: Thay đổi toàn bộ bộ tham số để tăng độ chính xác
    params = {  
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [12],
        'learning_rate': 0.02,        # Giảm tốc độ học
        'num_leaves': 255,            # Tăng độ sâu (127 -> 255)
        'min_child_samples': 30,      # Tăng số mẫu tối thiểu trên mỗi lá (20 -> 30)
        'verbose': -1,
        'device': 'cpu', 
    }

    valid_sets = [lgb_val] if lgb_val is not None else [lgb_train]
    
    model = lgb.train(
        params, 
        lgb_train,
        num_boost_round=1000,         # [ĐÃ SỬA]: Tăng số vòng huấn luyện lên 1000
        valid_sets=valid_sets,
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )

    model_path = MOD_DIR / 'lgbm_ranker_v2.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Model đã được lưu tại {model_path}')
    
    # In ra top features quan trọng
    importance = sorted(zip(FEATURE_COLS_V2, model.feature_importance('gain')), key=lambda x: -x[1])
    print('\nTop-10 Đặc trưng quan trọng nhất (theo Gain):')
    for fname, score in importance[:10]:
        print(f'  {fname:<40s} {score:.1f}')

if __name__ == "__main__":
    # Luồng chạy chuẩn
    # [ĐÃ SỬA]: Gọi ratio=20 thay vì 10
    sampled_paths = sample_negatives(ratio=20)
    lgb_train = build_lgb_dataset(sampled_paths)
    
    # Ở đây tạm thời bỏ qua tập lgb_val để train thẳng. 
    # Nếu muốn dùng Early Stopping chuẩn chỉ, bạn có thể build thêm tập Val tương tự hàm build_lgb_dataset.
    train_lgbm_model(lgb_train)