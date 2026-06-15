import gc
import numpy as np
import polars as pl
from datetime import datetime

# Import cấu hình
from src.config import TEST_START_DATE, DATE_7D, SUB_DIR, OUT_DIR, CAND_DIR, V2_DIR
from src.data_loader import load_data

# -----------------------------------------
# CÁC HÀM TÍNH TOÁN METRICS (MAP@12)
# -----------------------------------------
def apk(actual, predicted, k=12):
    if not actual: return 0.0
    predicted = predicted[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p in actual:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(actual), k)

def mapk(actuals, predicteds, k=12):
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)]))

# -----------------------------------------
# ĐÁNH GIÁ CHUNG VÀ ABLATION STUDY
# -----------------------------------------
def evaluate_map(transactions):
    """Tính điểm MAP@12 cho submission_v2"""
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tính toán MAP@12...')
    
    # Lấy ground truth (sự thật) từ tuần test
    test_tx = transactions.filter(pl.col('t_dat') >= TEST_START_DATE)
    ground_truth = (
        test_tx.group_by('customer_id')
        .agg(pl.col('article_id').unique().alias('bought'))
    )
    print(f'Số khách hàng mua trong tuần test: {ground_truth.shape[0]:,}')

    sub_path = SUB_DIR / 'submission_v2.csv'
    if not sub_path.exists():
        print(f"Lỗi: Không tìm thấy file {sub_path}. Hãy chạy predict.py trước.")
        return ground_truth, None

    sub_v2 = pl.read_csv(sub_path)
    eval_df = ground_truth.join(
        sub_v2.with_columns(pl.col('prediction').str.split(' ').alias('top12'))
              .select(['customer_id', 'top12']),
        on='customer_id', how='inner'
    )

    map12_v2 = mapk(eval_df['bought'].to_list(), eval_df['top12'].to_list())
    print(f'\n=== KẾT QUẢ CHÍNH ===')
    print(f'MAP@12 V2: {map12_v2:.4f}\n')
    
    return ground_truth, map12_v2

def run_ablation_study(transactions, customers, ground_truth, baseline_map):
    """
    Chạy kiểm định loại trừ từng nguồn ứng viên để đo lường mức độ hiệu quả.
    Lưu ý: Yêu cầu đã có các file scored_*.parquet trong thư mục v2/
    """
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Bắt đầu Ablation Study...')
    
    sources = ['repurchase', 'global_popular', 'age_popular', 'als', 'embedding_sim', 'item_item']
    ablation_results = [{'Removed': 'None (full)', 'MAP@12': baseline_map, 'Delta': 0.0}]

    global_top12 = (
        transactions.filter(pl.col('t_dat') > DATE_7D)
        .group_by('article_id').len()
        .sort('len', descending=True)
        .head(12)['article_id'].to_list()
    )

    all_customer_ids = customers['customer_id'].to_list()
    scored_files = sorted(V2_DIR.glob('scored_*.parquet'))
    
    if not scored_files:
        print("Không tìm thấy các file chunk đã chấm điểm (scored_*.parquet). Bỏ qua Ablation.")
        return

    for src in sources:
        print(f'Đang xử lý loại bỏ nguồn: - {src}')
        current_src_ranked_chunks = []

        for i, scored_file_path in enumerate(scored_files):
            scored_chunk = pl.read_parquet(scored_file_path)
            chunk_customer_ids = scored_chunk['customer_id'].unique().to_list()

            # Lọc bỏ nguồn hiện tại khỏi tập candidates
            ablated_cands_chunk = (
                pl.scan_parquet(CAND_DIR / 'all_candidates_v2.parquet')
                .filter(pl.col('customer_id').is_in(chunk_customer_ids))
                .filter(pl.col('source') != src)
                .select(['customer_id', 'article_id'])
                .collect()
            )

            scored_ablated_chunk = scored_chunk.join(ablated_cands_chunk, on=['customer_id', 'article_id'], how='inner')

            ranked_abl_chunk = (
                scored_ablated_chunk
                .sort(['customer_id', 'score'], descending=[False, True])
                .group_by('customer_id', maintain_order=True)
                .agg(pl.col('article_id').head(12).alias('top12'))
            )
            current_src_ranked_chunks.append(ranked_abl_chunk)

            del scored_chunk, ablated_cands_chunk, scored_ablated_chunk, ranked_abl_chunk; gc.collect()

        ranked_abl = pl.concat(current_src_ranked_chunks)
        del current_src_ranked_chunks; gc.collect()

        # Fallback
        ranked_abl_customer_set = set(ranked_abl['customer_id'].to_list())
        missing_cids = [c for c in all_customer_ids if c not in ranked_abl_customer_set]
        if missing_cids:
            ranked_abl = pl.concat([ranked_abl, pl.DataFrame({'customer_id': missing_cids, 'top12': [global_top12] * len(missing_cids)})])

        # Đánh giá lại
        eval_abl = ground_truth.join(ranked_abl.select(['customer_id', 'top12']), on='customer_id', how='inner')
        m = mapk(eval_abl['bought'].to_list(), eval_abl['top12'].to_list())
        delta = m - baseline_map
        
        ablation_results.append({'Removed': f'- {src}', 'MAP@12': m, 'Delta': delta})
        print(f'  => {src:<15s} MAP@12={m:.4f} | delta={delta:+.4f}')
        del ranked_abl, eval_abl, ranked_abl_customer_set; gc.collect()

    ablation_df = pl.DataFrame(ablation_results)
    ablation_df.write_csv(OUT_DIR / 'ablation_v2.csv')
    print('\n=== BẢNG ABLATION KẾT QUẢ ===')
    print(ablation_df)

if __name__ == "__main__":
    transactions, _, _, _, customers = load_data()
    ground_truth, baseline_map = evaluate_map(transactions)
    
    if baseline_map is not None:
        run_ablation_study(transactions, customers, ground_truth, baseline_map)