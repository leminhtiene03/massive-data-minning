import polars as pl
from datetime import datetime
from src.config import DATA_DIR, TEST_START_DATE, VAL_START_DATE

def load_data():
    """
    Đọc dữ liệu thô từ file CSV, ép kiểu và chia tập train/test.
    """
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đang nạp dữ liệu gốc...')
    
    # Đọc transactions và ép kiểu string cho id
    transactions = pl.read_csv(
        DATA_DIR / 'transactions_train.csv',
        dtypes={'article_id': pl.String, 'customer_id': pl.String}
    )
    # Xử lý định dạng ngày tháng
    transactions = transactions.with_columns(pl.col('t_dat').str.to_date('%Y-%m-%d'))
    
    # Tập train_tx: Lọc bỏ tuần test cuối cùng
    train_tx = transactions.filter(pl.col('t_dat') < TEST_START_DATE)
    
    # Tập train_tx_clean: Lọc bỏ thêm tuần validation để dùng cho việc sinh feature
    train_tx_clean = train_tx.filter(pl.col('t_dat') < VAL_START_DATE)
    
    # Đọc articles
    articles = pl.read_csv(DATA_DIR / 'articles.csv', dtypes={'article_id': pl.String})
    
    # Đọc customers
    customers = pl.read_csv(DATA_DIR / 'customers.csv', dtypes={'customer_id': pl.String})

    print(f'[{datetime.now().strftime("%H:%M:%S")}] Kích thước dữ liệu:')
    print(f' - All Transactions: {transactions.shape}')
    print(f' - Train TX Clean (dùng cho features): {train_tx_clean.shape}')
    print(f' - Articles: {articles.shape}')
    print(f' - Customers: {customers.shape}')
    
    return transactions, train_tx, train_tx_clean, articles, customers

if __name__ == "__main__":
    # Chạy thử file này độc lập để kiểm tra xem có đọc được data không
    load_data()