import gc
import polars as pl
from datetime import datetime
from src.config import (
    LAST_TRAIN_DATE, DATE_7D, DATE_14D, DATE_30D,
    VAL_START_DATE, VAL_END_DATE,
    FEAT_DIR, CAND_DIR, V2_DIR
)

def compute_customer_stats(train_tx_clean, customers):
    """Tính toán các chỉ số lịch sử mua hàng của khách hàng"""
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tính toán Customer stats...')
    
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
    """Tính toán các chỉ số thống kê của sản phẩm"""
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang tính toán Article stats...')
    
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
    """Tính toán trọng số thời gian (Time-decay) và Xu hướng bán hàng (Sales trend)"""
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
        print(f'[{datetime.now().strftime("%H:%M:%S")}] Time-decay saved')

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

    del _decay, _trend
    return feats