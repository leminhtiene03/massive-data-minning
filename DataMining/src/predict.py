import gc
import pickle
import polars as pl
from datetime import datetime

# Import cấu hình và các hàm đã viết
from src.config import V2_DIR, CAND_DIR, MOD_DIR, SUB_DIR, DATE_7D, FEATURE_COLS_V2
from src.data_loader import load_data
from src.features import compute_customer_stats, compute_article_stats, build_features_v2

def build_test_features(train_tx_clean, articles, customers):
    """Tính toán lại đặc trưng cho tập ứng viên cần dự đoán"""
    # Chuẩn bị stats dict giống hệt lúc train để tái sử dụng hàm build_features_v2
    cust_7d, cust_14d, cust_30d, last_purchase, customers_feat = compute_customer_stats(train_tx_clean, customers)
    art_pop_7d, art_pop_14d, art_age_pop, art_last_sold, articles_feat = compute_article_stats(train_tx_clean, articles, customers_feat)
    
    stats_dict = {
        'customers_feat': customers_feat, 'cust_7d': cust_7d, 'cust_14d': cust_14d,
        'cust_30d': cust_30d, 'last_purchase': last_purchase,
        'art_pop_7d': art_pop_7d, 'art_pop_14d': art_pop_14d,
        'art_age_pop': art_age_pop, 'art_last_sold': art_last_sold,
        'articles_feat': articles_feat
    }

    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang build test features...')
    all_cands_v2 = pl.read_parquet(CAND_DIR / 'all_candidates_v2.parquet')
    pred_cids = all_cands_v2['customer_id'].unique().to_list()
    pred_chunks = [pred_cids[i:i+100_000] for i in range(0, len(pred_cids), 100_000)]

    for i, cid_chunk in enumerate(pred_chunks):
        pp = V2_DIR / f'pred_feat_{i:04d}.parquet'
        if pp.exists():
            print(f'  Pred chunk {i+1}/{len(pred_chunks)} đã tồn tại — Bỏ qua')
            continue
        
        cid_set = set(cid_chunk)
        cands_c = all_cands_v2.filter(pl.col('customer_id').is_in(cid_set))
        feat_c = build_features_v2(cands_c, stats_dict, label_tx=None)
        feat_c.write_parquet(pp)
        del cands_c, feat_c; gc.collect()
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Pred chunk {i+1}/{len(pred_chunks)} hoàn thành')

    del all_cands_v2; gc.collect()
    test_feat_path = V2_DIR / 'test_features_v2.parquet'
    pl.scan_parquet(V2_DIR / 'pred_feat_*.parquet').sink_parquet(test_feat_path)
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đã lưu Test features tổng hợp')
    return test_feat_path

def score_and_rank(train_tx, customers):
    """Dùng mô hình LGBM chấm điểm, xếp hạng và xuất ra Top 12"""
    # 1. Chấm điểm (Scoring) theo chunk
    with open(MOD_DIR / 'lgbm_ranker_v2.pkl', 'rb') as f:
        model = pickle.load(f)
        
    all_cids = pl.scan_parquet(V2_DIR / 'test_features_v2.parquet').select('customer_id').unique().collect()['customer_id'].to_list()
    chunks = [all_cids[i:i+100_000] for i in range(0, len(all_cids), 100_000)]
    scored_paths = []

    for i, cid_chunk in enumerate(chunks):
        sp = V2_DIR / f'scored_{i:04d}.parquet'
        if not sp.exists():
            cid_set = set(cid_chunk)
            chunk = pl.scan_parquet(V2_DIR / 'test_features_v2.parquet').filter(pl.col('customer_id').is_in(cid_set)).collect()
            cols_to_use = [c for c in FEATURE_COLS_V2 if c in chunk.columns]
            
            scores = model.predict(chunk.select(cols_to_use).to_pandas())
            scored = chunk.select(['customer_id', 'article_id']).with_columns(pl.Series('score', scores))
            scored.write_parquet(sp)
            del chunk, scores, scored; gc.collect()
        scored_paths.append(sp)
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Chấm điểm chunk {i+1}/{len(chunks)} hoàn thành')

    # 2. Xếp hạng (Ranking) và lọc Top 12
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang xếp hạng và lọc Top 12...')
    top50_paths = []
    for i, sp in enumerate(scored_paths):
        tp = V2_DIR / f'top50_{i:04d}.parquet'
        if not tp.exists():
            chunk = pl.read_parquet(sp)
            top50 = (
                chunk.sort(['customer_id', 'score'], descending=[False, True])
                .group_by('customer_id', maintain_order=True)
                .agg(pl.col('article_id').head(50).alias('top50'), pl.col('score').head(50).alias('scores'))
            )
            top50.write_parquet(tp)
            del chunk, top50; gc.collect()
        top50_paths.append(tp)

    merged = pl.concat([pl.read_parquet(p) for p in top50_paths])
    ranked = (
        merged.explode(['top50', 'scores'])
        .rename({'top50': 'article_id', 'scores': 'score'})
        .sort(['customer_id', 'score'], descending=[False, True])
        .group_by('customer_id', maintain_order=True)
        .agg(pl.col('article_id').head(12).alias('top12'))
    )
    del merged; gc.collect()

    # 3. Fallback: Xử lý khách hàng không có ứng viên
    global_top12 = (
        train_tx.filter(pl.col('t_dat') > DATE_7D)
        .group_by('article_id').len()
        .sort('len', descending=True)
        .head(12)['article_id'].to_list()
    )

    ranked_cids = ranked.select('customer_id').unique()
    missing = customers.select('customer_id').join(ranked_cids, on='customer_id', how='anti')['customer_id'].to_list()
    if missing:
        print(f'Có {len(missing)} khách hàng trống, áp dụng Fallback...')
        ranked = pl.concat([ranked, pl.DataFrame({'customer_id': missing, 'top12': [global_top12] * len(missing)})])

    # 4. Ghi ra file CSV nộp bài
    submission = (
        customers.select('customer_id')
        .join(ranked.with_columns(pl.col('top12').list.join(' ').alias('prediction')).select(['customer_id', 'prediction']), on='customer_id', how='left')
        .with_columns(pl.col('prediction').fill_null(' '.join(global_top12)))
    )

    sub_path = SUB_DIR / 'submission_v2.csv'
    submission.write_csv(sub_path)
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đã lưu file kết quả tại: {sub_path}')

if __name__ == "__main__":
    _, train_tx, train_tx_clean, articles, customers = load_data()
    build_test_features(train_tx_clean, articles, customers)
    score_and_rank(train_tx, customers)