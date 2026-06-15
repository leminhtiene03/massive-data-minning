from datetime import datetime
from src.data_loader import load_data
from src.candidates import run_candidate_generation
<<<<<<< HEAD
from src.features import build_train_features  # <-- Đã thêm import hàm này
=======
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
from src.train import sample_negatives, build_lgb_dataset, train_lgbm_model
from src.predict import build_test_features, score_and_rank
from src.evaluate import evaluate_map, run_ablation_study

def main():
    print(f'[{datetime.now().strftime("%H:%M:%S")}] KHỞI ĐỘNG HỆ THỐNG GỢI Ý H&M V2')

    # BẬT/TẮT CÁC BƯỚC TẠI ĐÂY (Đổi thành False nếu muốn bỏ qua bước đó)
    RUN_CANDIDATES = True
    RUN_TRAINING   = True
    RUN_PREDICT    = True
    RUN_EVALUATE   = True

    # Bước 1: Nạp dữ liệu
    transactions, train_tx, train_tx_clean, articles, customers = load_data()

    # Bước 2: Sinh tập ứng viên
    if RUN_CANDIDATES:
        print("\n" + "="*50)
        print("PHẦN 1: SINH ỨNG VIÊN (CANDIDATE GENERATION)")
        print("="*50)
        run_candidate_generation(train_tx, customers)

    # Bước 3: Huấn luyện mô hình xếp hạng
    if RUN_TRAINING:
        print("\n" + "="*50)
        print("PHẦN 2: HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING)")
        print("="*50)
<<<<<<< HEAD
        
        build_train_features(transactions, train_tx_clean, articles, customers)

=======
>>>>>>> 16749963e5caa79fb5f645de6374aa10dce318ff
        sampled_paths = sample_negatives(ratio=10)
        lgb_train = build_lgb_dataset(sampled_paths)
        train_lgbm_model(lgb_train)

    # Bước 4: Dự đoán và xuất kết quả
    if RUN_PREDICT:
        print("\n" + "="*50)
        print("PHẦN 3: DỰ ĐOÁN & RANKING (PREDICTION)")
        print("="*50)
        build_test_features(train_tx_clean, articles, customers)
        score_and_rank(train_tx, customers)

    # Bước 5: Đánh giá và Phân tích
    if RUN_EVALUATE:
        print("\n" + "="*50)
        print("PHẦN 4: ĐÁNH GIÁ (EVALUATION & ABLATION)")
        print("="*50)
        ground_truth, baseline_map = evaluate_map(transactions)
        if baseline_map is not None:
            run_ablation_study(transactions, customers, ground_truth, baseline_map)

    print(f'\n[{datetime.now().strftime("%H:%M:%S")}] HOÀN THÀNH TOÀN BỘ PIPELINE!')

if __name__ == "__main__":
    main()