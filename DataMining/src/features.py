import gc
<<<<<<< HEAD
import os
import polars as pl
from datetime import datetime, date

# Import cấu hình
from src.config import (
    LAST_TRAIN_DATE, DATE_7D, DATE_14D, DATE_30D,
=======
import polars as pl
from datetime import datetime
from src.config import (
    LAST_TRAIN_DATE, DATE_7D, DATE_14D, DATE_30D,
    VAL_START_DATE, VAL_END_DATE,
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
    FEAT_DIR, CAND_DIR, V2_DIR
)

def compute_customer_stats(train_tx_clean, customers):
<<<<<<< HEAD
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tính toán Customer stats...')
=======
    """Tính toán các chỉ số lịch sử mua hàng của khách hàng"""
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tính toán Customer stats...')
    
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
    def cust_stats(tx, days, suffix):
        cutoff = LAST_TRAIN_DATE - pl.duration(days=days)
        return (tx.filter(pl.col('t_dat') > cutoff)
                .group_by('customer_id')
                .agg([pl.len().alias(f'n_purchases_{suffix}'),
                      pl.col('article_id').n_unique().alias(f'n_unique_articles_{suffix}'),
                      pl.col('price').mean().alias(f'avg_price_{suffix}')]))

    cust_7d  = cust_stats(train_tx_clean, 7,  '7d')
    cust_14d = cust_stats(train_tx_clean, 14, '14d')
    cust_30d = cust_stats(train_tx_clean, 30, '30d')

    last_purchase = (
        train_tx_clean.group_by('customer_id')
        .agg(pl.col('t_dat').max().alias('last_purchase_date'))
        .with_columns(((pl.lit(LAST_TRAIN_DATE) - pl.col('last_purchase_date')).dt.total_days())
                      .alias('days_since_last_purchase'))
        .select(['customer_id', 'days_since_last_purchase'])
    )

    age_median = customers['age'].drop_nulls().median()
    customers_feat = (
        customers.select(['customer_id', 'age', 'club_member_status', 'fashion_news_frequency'])
        .with_columns([
            pl.col('age').fill_null(age_median),
            (pl.col('club_member_status') == 'ACTIVE').cast(pl.Int8).alias('is_club_member'),
            pl.when(pl.col('fashion_news_frequency').is_in(['Regularly', 'Monthly']))
              .then(pl.lit(1)).otherwise(pl.lit(0)).cast(pl.Int8).alias('fashion_news_freq'),
        ])
        .with_columns(
            pl.when(pl.col('age') < 25).then(pl.lit('<25'))
            .when(pl.col('age') < 35).then(pl.lit('25-35'))
            .when(pl.col('age') < 50).then(pl.lit('35-50'))
            .otherwise(pl.lit('50+')).alias('age_group')
        )
    )
    return cust_7d, cust_14d, cust_30d, last_purchase, customers_feat

def compute_article_stats(train_tx_clean, articles, customers_feat):
<<<<<<< HEAD
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tính toán Article stats...')
=======
    """Tính toán các chỉ số thống kê của sản phẩm"""
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tính toán Article stats...')
    
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
    def article_pop(tx, days, suffix):
        cutoff = LAST_TRAIN_DATE - pl.duration(days=days)
        return (tx.filter(pl.col('t_dat') > cutoff)
                .group_by('article_id').len()
                .rename({'len': f'article_popularity_{suffix}'}))

    art_pop_7d  = article_pop(train_tx_clean, 7,  '7d')
    art_pop_14d = article_pop(train_tx_clean, 14, '14d')

    art_age_pop = (
        train_tx_clean.filter(pl.col('t_dat') > DATE_14D)
        .join(customers_feat.select(['customer_id', 'age_group']), on='customer_id', how='left')
        .group_by(['age_group', 'article_id']).len()
        .rename({'len': 'article_popularity_age_group_14d'})
    )

    art_last_sold = (
        train_tx_clean.group_by('article_id')
        .agg(pl.col('t_dat').max().alias('last_sold_date'))
        .with_columns(((pl.lit(LAST_TRAIN_DATE) - pl.col('last_sold_date')).dt.total_days())
                      .alias('days_since_article_last_sold'))
        .select(['article_id', 'days_since_article_last_sold'])
    )

    articles_feat = articles.select(
        ['article_id', 'product_type_no', 'colour_group_code', 'department_no', 'perceived_colour_value_id']
    )
    return art_pop_7d, art_pop_14d, art_age_pop, art_last_sold, articles_feat

def compute_time_decay_and_trend(train_tx_clean):
<<<<<<< HEAD
=======
    """Tính toán trọng số thời gian (Time-decay) và Xu hướng bán hàng (Sales trend)"""
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
    decay_path = FEAT_DIR / 'user_item_decay.parquet'
    if not decay_path.exists():
        user_item_decay = (
            train_tx_clean
            .with_columns((pl.lit(LAST_TRAIN_DATE) - pl.col('t_dat')).dt.total_days().alias('days_ago'))
            .with_columns((pl.lit(0.95) ** pl.col('days_ago')).alias('decay_weight'))
            .group_by(['customer_id', 'article_id'])
            .agg(pl.col('decay_weight').sum().alias('user_item_decay_weight'))
        )
        user_item_decay.write_parquet(decay_path)
<<<<<<< HEAD
    
=======
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Time-decay saved')

>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
    trend_path = FEAT_DIR / 'art_sales_trend.parquet'
    if not trend_path.exists():
        art_alltime = train_tx_clean.group_by('article_id').len().rename({'len': 'art_alltime_sales'})
        art_recent_7d = train_tx_clean.filter(pl.col('t_dat') > DATE_7D).group_by('article_id').len().rename({'len': 'art_recent_7d_sales'})
        art_trend = (
            art_alltime.join(art_recent_7d, on='article_id', how='left')
            .with_columns(pl.col('art_recent_7d_sales').fill_null(0))
            .with_columns((pl.col('art_recent_7d_sales') / (pl.col('art_alltime_sales') + 1)).alias('sales_trend'))
            .select(['article_id', 'sales_trend'])
        )
        art_trend.write_parquet(trend_path)
<<<<<<< HEAD

def compute_interaction_tables(train_tx_clean, articles):
    if not (FEAT_DIR / 'user_bought.parquet').exists():
        user_bought = train_tx_clean.select(['customer_id', 'article_id']).unique().with_columns(pl.lit(1).cast(pl.Int8).alias('user_has_bought_before'))
        user_bought.write_parquet(FEAT_DIR / 'user_bought.parquet')
        del user_bought; gc.collect()

    if not (FEAT_DIR / 'user_prod_types.parquet').exists():
        user_prod_types = train_tx_clean.join(articles.select(['article_id', 'product_type_no']), on='article_id', how='left').select(['customer_id', 'product_type_no']).unique()
        user_prod_types.write_parquet(FEAT_DIR / 'user_prod_types.parquet')
        del user_prod_types; gc.collect()

    if not (FEAT_DIR / 'user_depts.parquet').exists():
        user_depts = train_tx_clean.join(articles.select(['article_id', 'department_no']), on='article_id', how='left').select(['customer_id', 'department_no']).unique()
        user_depts.write_parquet(FEAT_DIR / 'user_depts.parquet')
        del user_depts; gc.collect()

def compute_cross_features(train_tx_clean, articles):
    u_i_count_path = FEAT_DIR / 'user_item_count.parquet'
    if not u_i_count_path.exists():
        u_i_count = train_tx_clean.group_by(['customer_id', 'article_id']).len().rename({'len': 'user_item_purchase_count'})
        u_i_count.write_parquet(u_i_count_path)
        del u_i_count; gc.collect()

    u_dept_aff_path = FEAT_DIR / 'user_dept_affinity.parquet'
    if not u_dept_aff_path.exists():
        user_total = train_tx_clean.group_by('customer_id').len().rename({'len': 'total_purchases'})
        user_dept = train_tx_clean.join(articles.select(['article_id', 'department_no']), on='article_id', how='left')
        user_dept_counts = user_dept.group_by(['customer_id', 'department_no']).len().rename({'len': 'dept_purchases'})
        user_dept_affinity = user_dept_counts.join(user_total, on='customer_id')
        user_dept_affinity = user_dept_affinity.with_columns((pl.col('dept_purchases') / pl.col('total_purchases')).alias('user_dept_affinity')).drop(['dept_purchases', 'total_purchases'])
        user_dept_affinity.write_parquet(u_dept_aff_path)
        del user_total, user_dept, user_dept_counts, user_dept_affinity; gc.collect()

def compute_advanced_features(train_tx_clean):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tính toán Advanced Features...')
    
    age_path = FEAT_DIR / 'article_age.parquet'
    if not age_path.exists():
        art_age = (
            train_tx_clean.group_by('article_id')
            .agg(pl.col('t_dat').min().alias('first_sold_date'))
            .with_columns(((pl.lit(LAST_TRAIN_DATE) - pl.col('first_sold_date')).dt.total_days()).alias('article_age_days'))
            .select(['article_id', 'article_age_days'])
        )
        art_age.write_parquet(age_path)
        del art_age; gc.collect()

    cust_price_path = FEAT_DIR / 'cust_avg_price.parquet'
    art_price_path = FEAT_DIR / 'art_avg_price.parquet'
    if not cust_price_path.exists():
        cust_price = train_tx_clean.group_by('customer_id').agg(pl.col('price').mean().alias('cust_avg_price'))
        cust_price.write_parquet(cust_price_path)
        del cust_price; gc.collect()
        
    if not art_price_path.exists():
        art_price = train_tx_clean.group_by('article_id').agg(pl.col('price').mean().alias('art_avg_price'))
        art_price.write_parquet(art_price_path)
        del art_price; gc.collect()

    season_path = FEAT_DIR / 'autumn_seasonality.parquet'
    if not season_path.exists():
        autumn_start = date(2019, 8, 1)
        autumn_end = date(2019, 10, 31)
        autumn_sales = (
            train_tx_clean.filter((pl.col('t_dat') >= autumn_start) & (pl.col('t_dat') <= autumn_end))
            .group_by('article_id').len().rename({'len': 'autumn_2019_sales'})
        )
        autumn_sales.write_parquet(season_path)
        del autumn_sales; gc.collect()

def compute_source_pivot():
    src_pivot_v2_path = FEAT_DIR / 'source_pivot_v2.parquet'
    if not src_pivot_v2_path.exists():
        all_cands_v2 = pl.read_parquet(CAND_DIR / 'all_candidates_v2.parquet')
        all_cids = all_cands_v2['customer_id'].unique().to_list()
        CHUNK = 100_000
        pivot_paths = []
        for i in range(0, len(all_cids), CHUNK):
            cid_set = set(all_cids[i:i+CHUNK])
            chunk = all_cands_v2.filter(pl.col('customer_id').is_in(cid_set))
            pivot = chunk.with_columns(pl.lit(1).cast(pl.Int8).alias('flag')).pivot(index=['customer_id', 'article_id'], on='source', values='flag', aggregate_function='first').fill_null(0)
            for src in ['repurchase', 'global_popular', 'age_popular', 'als', 'embedding_sim', 'item_item']:
                col_name = f'candidate_source_{src}'
                if src in pivot.columns: pivot = pivot.rename({src: col_name})
                elif col_name not in pivot.columns: pivot = pivot.with_columns(pl.lit(0).cast(pl.Int8).alias(col_name))
            pp = V2_DIR / f'pivot_{i:07d}.parquet'
            pivot.write_parquet(pp)
            pivot_paths.append(pp)
            del chunk, pivot; gc.collect()
        pl.scan_parquet([str(p) for p in pivot_paths]).sink_parquet(src_pivot_v2_path)
        for p in pivot_paths: p.unlink()
        del all_cands_v2; gc.collect()

def build_features_v2(candidates_df, stats_dict, label_tx=None):
    cids = candidates_df['customer_id'].unique()

=======
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Sales trend saved')

# -------------------------------------------------------------------------
# Hàm build_features_v2: Gắn đặc trưng vào tập ứng viên (Candidates)
# (Bạn cần gọi hàm này trong vòng lặp chia chunk lúc train/test giống file gốc)
# -------------------------------------------------------------------------

def build_features_v2(candidates_df, stats_dict, label_tx=None):
    """
    stats_dict chứa các dataframe đã được precompute từ các hàm trên:
    { 'customers_feat': customers_feat, 'cust_7d': cust_7d, ... }
    """
    cids = candidates_df['customer_id'].unique()

    # Load các bảng từ ổ cứng để tránh RAM đầy
    _decay = pl.read_parquet(FEAT_DIR / 'user_item_decay.parquet').filter(pl.col('customer_id').is_in(cids))
    _trend = pl.read_parquet(FEAT_DIR / 'art_sales_trend.parquet')
    
    # Do giới hạn hiển thị, giả định các bảng user_bought, source_pivot, emb_sim đã được tạo
    # và đọc tương tự như _decay.
    
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
    feats = (
        candidates_df.select(['customer_id', 'article_id']).unique()
        .join(stats_dict['customers_feat'].select(['customer_id', 'age', 'is_club_member', 'fashion_news_freq', 'age_group']), on='customer_id', how='left')
        .join(stats_dict['cust_7d'], on='customer_id', how='left')
        .join(stats_dict['cust_14d'], on='customer_id', how='left')
        .join(stats_dict['cust_30d'].select(['customer_id', 'n_purchases_30d', 'n_unique_articles_30d', 'avg_price_30d']), on='customer_id', how='left')
        .join(stats_dict['last_purchase'], on='customer_id', how='left')
        .join(stats_dict['art_pop_7d'], on='article_id', how='left')
        .join(stats_dict['art_pop_14d'], on='article_id', how='left')
        .join(stats_dict['art_age_pop'], on=['age_group', 'article_id'], how='left')
        .join(stats_dict['art_last_sold'], on='article_id', how='left')
        .join(stats_dict['articles_feat'], on='article_id', how='left')
<<<<<<< HEAD
    )

    # Nối các file có sẵn
    if (FEAT_DIR / 'art_sales_trend.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'art_sales_trend.parquet'), on='article_id', how='left')
    if (FEAT_DIR / 'user_bought.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'user_bought.parquet').filter(pl.col('customer_id').is_in(cids)), on=['customer_id', 'article_id'], how='left')
    if (FEAT_DIR / 'user_prod_types.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'user_prod_types.parquet').filter(pl.col('customer_id').is_in(cids)).with_columns(pl.lit(1).cast(pl.Int8).alias('user_bought_same_product_type')), on=['customer_id', 'product_type_no'], how='left')
    if (FEAT_DIR / 'user_depts.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'user_depts.parquet').filter(pl.col('customer_id').is_in(cids)).with_columns(pl.lit(1).cast(pl.Int8).alias('user_bought_same_department')), on=['customer_id', 'department_no'], how='left')
    if (FEAT_DIR / 'user_item_decay.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'user_item_decay.parquet').filter(pl.col('customer_id').is_in(cids)), on=['customer_id', 'article_id'], how='left')
    if (FEAT_DIR / 'source_pivot_v2.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'source_pivot_v2.parquet').filter(pl.col('customer_id').is_in(cids)), on=['customer_id', 'article_id'], how='left')
    if (FEAT_DIR / 'user_item_count.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'user_item_count.parquet').filter(pl.col('customer_id').is_in(cids)), on=['customer_id', 'article_id'], how='left')
    if (FEAT_DIR / 'user_dept_affinity.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'user_dept_affinity.parquet').filter(pl.col('customer_id').is_in(cids)), on=['customer_id', 'department_no'], how='left')

    # Nối Advanced Features
    if (FEAT_DIR / 'article_age.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'article_age.parquet'), on='article_id', how='left')
    if (FEAT_DIR / 'autumn_seasonality.parquet').exists():
        feats = feats.join(pl.read_parquet(FEAT_DIR / 'autumn_seasonality.parquet'), on='article_id', how='left')
        
    # [TÍCH HỢP ĐA PHƯƠNG THỨC] Nối Đặc trưng Hình ảnh 32 chiều
    IMG_FEAT_PATH = "/content/drive/MyDrive/inputs2/data/image_features_32d.parquet"
    if os.path.exists(IMG_FEAT_PATH):
        img_feats = pl.read_parquet(IMG_FEAT_PATH)
        feats = feats.join(img_feats, on='article_id', how='left')

    # Tính Price Deviation
    if (FEAT_DIR / 'cust_avg_price.parquet').exists() and (FEAT_DIR / 'art_avg_price.parquet').exists():
        _c_price = pl.read_parquet(FEAT_DIR / 'cust_avg_price.parquet').filter(pl.col('customer_id').is_in(cids))
        _a_price = pl.read_parquet(FEAT_DIR / 'art_avg_price.parquet')
        feats = feats.join(_c_price, on='customer_id', how='left').join(_a_price, on='article_id', how='left')
        feats = feats.with_columns(
            (pl.col('art_avg_price') - pl.col('cust_avg_price')).abs().alias('price_deviation')
        ).drop(['cust_avg_price', 'art_avg_price'])

    # Xử lý missing values
    int_cols   = ['is_club_member', 'fashion_news_freq', 'user_has_bought_before',
                  'user_bought_same_product_type', 'user_bought_same_department',
                  'candidate_source_repurchase', 'candidate_source_global_popular',
                  'candidate_source_age_popular', 'candidate_source_als',
                  'candidate_source_embedding_sim', 'candidate_source_item_item']
    
    zero_cols  = ['n_purchases_7d', 'n_purchases_14d', 'n_purchases_30d',
                  'n_unique_articles_30d', 'article_popularity_7d', 'article_popularity_14d',
                  'article_popularity_age_group_14d', 'embedding_similarity',
                  'user_item_decay_weight', 'sales_trend', 'user_dept_affinity', 
                  'user_item_purchase_count', 'autumn_2019_sales']
                  
    # Tự động điền số 0.0 cho 32 cột ảnh nếu có sản phẩm nào bị mất ảnh gốc
    zero_cols.extend([f'img_feat_{i}' for i in range(32)]) 

    large_cols = ['days_since_last_purchase', 'days_since_article_last_sold', 'article_age_days', 'price_deviation']

    for c in int_cols:
        if c in feats.columns: feats = feats.with_columns(pl.col(c).fill_null(0).cast(pl.Int8))
    for c in zero_cols:
        if c in feats.columns: feats = feats.with_columns(pl.col(c).fill_null(0.0))
    for c in large_cols:
        if c in feats.columns: feats = feats.with_columns(pl.col(c).fill_null(9999.0))
=======
        .join(_trend, on='article_id', how='left')
        .join(_decay, on=['customer_id', 'article_id'], how='left')
    )

    # Điền giá trị Null
    zero_cols = ['n_purchases_7d', 'n_purchases_14d', 'n_purchases_30d', 'n_unique_articles_30d', 
                 'article_popularity_7d', 'article_popularity_14d', 'article_popularity_age_group_14d', 
                 'user_item_decay_weight', 'sales_trend']
    large_cols = ['days_since_last_purchase', 'days_since_article_last_sold']

    for c in zero_cols:
        if c in feats.columns: feats = feats.with_columns(pl.col(c).fill_null(0))
    for c in large_cols:
        if c in feats.columns: feats = feats.with_columns(pl.col(c).fill_null(999))
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
        
    feats = feats.with_columns([
        pl.col('avg_price_7d').fill_null(pl.col('avg_price_7d').mean()),
        pl.col('avg_price_14d').fill_null(pl.col('avg_price_14d').mean()),
        pl.col('avg_price_30d').fill_null(pl.col('avg_price_30d').mean()),
    ])

    if label_tx is not None:
        bought = (label_tx.select(['customer_id', 'article_id']).unique()
                  .with_columns(pl.lit(1).cast(pl.Int8).alias('label')))
        feats = feats.join(bought, on=['customer_id', 'article_id'], how='left')
        feats = feats.with_columns(pl.col('label').fill_null(0))

<<<<<<< HEAD
    return feats


def build_train_features(transactions, train_tx_clean, articles, customers):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang build train features...')
    
    compute_time_decay_and_trend(train_tx_clean)
    compute_interaction_tables(train_tx_clean, articles)
    compute_source_pivot()
    compute_cross_features(train_tx_clean, articles)
    compute_advanced_features(train_tx_clean)
    
    cust_7d, cust_14d, cust_30d, last_purchase, customers_feat = compute_customer_stats(train_tx_clean, customers)
    art_pop_7d, art_pop_14d, art_age_pop, art_last_sold, articles_feat = compute_article_stats(train_tx_clean, articles, customers_feat)
    
    stats_dict = {
        'customers_feat': customers_feat, 'cust_7d': cust_7d, 'cust_14d': cust_14d,
        'cust_30d': cust_30d, 'last_purchase': last_purchase,
        'art_pop_7d': art_pop_7d, 'art_pop_14d': art_pop_14d,
        'art_age_pop': art_age_pop, 'art_last_sold': art_last_sold,
        'articles_feat': articles_feat
    }

    VAL_START = date(2020, 9, 9)
    VAL_END   = date(2020, 9, 15)
    label_tx = transactions.filter((pl.col('t_dat') >= VAL_START) & (pl.col('t_dat') <= VAL_END))
    
    all_cands_v2 = pl.read_parquet(CAND_DIR / 'all_candidates_v2.parquet')
    all_cids = all_cands_v2['customer_id'].unique().to_list()
    CHUNK_SIZE = 100_000
    chunks = [all_cids[i:i+CHUNK_SIZE] for i in range(0, len(all_cids), CHUNK_SIZE)]
    
    for i, cid_chunk in enumerate(chunks):
        chunk_path = V2_DIR / f'train_feat_{i:04d}.parquet'
        if chunk_path.exists():
            continue
            
        cid_set = set(cid_chunk)
        cands_chunk = all_cands_v2.filter(pl.col('customer_id').is_in(cid_set))
        label_chunk = label_tx.filter(pl.col('customer_id').is_in(cid_set))
        
        feat_chunk = build_features_v2(cands_chunk, stats_dict, label_tx=label_chunk)
        feat_chunk.write_parquet(chunk_path)
        del cands_chunk, label_chunk, feat_chunk; gc.collect()
        print(f'  [{datetime.now().strftime("%H:%M:%S")}] Chunk {i+1}/{len(chunks)} hoàn thành')

    del all_cands_v2; gc.collect()
    
    # ĐÃ XÓA PHẦN GỘP FILE (sink_parquet) TẠI ĐÂY ĐỂ CHỐNG TRÀN RAM
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Hoàn thành build toàn bộ train features theo từng Chunk!')
=======
    del _decay, _trend
    return feats
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
