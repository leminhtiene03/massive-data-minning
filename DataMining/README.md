# H&M Personalized Fashion Recommendations

Dự án khai phá dữ liệu bán lẻ và xây dựng hệ thống gợi ý sản phẩm thời trang cá nhân hóa (Personalized Fashion Recommendations) áp dụng trên tập dữ liệu của H&M.

##  Tổng quan Kiến trúc (Two-Stage Recommender System)

Hệ thống được thiết kế theo cấu trúc 2 tầng (Two-stage) chuyên nghiệp để xử lý tập dữ liệu lớn (Big Data):
1. **Tầng 1 - Candidate Generation (Sinh ứng viên):
Thay vì xếp hạng toàn bộ kho hàng, hệ thống thu hẹp phạm vi bằng cách lọc ra danh sách ứng viên tiềm năng cho từng khách hàng thông qua các chiến lược:
   * Mua lại (Repurchase)
   * Xu hướng toàn cục và xu hướng theo độ tuổi (Global/Age Popularity)
   * Lọc cộng tác (Collaborative Filtering - ALS, Item-to-item)
   * Tương đồng Vector (Text & Image Embeddings bằng SentenceTransformers và CLIP)
2. **Tầng 2 - Learning to Rank (Học xếp hạng):
Sử dụng thuật toán LightGBM (Lambdarank) để đánh giá và xếp hạng danh sách ứng viên, từ đó chọn ra Top 12 sản phẩm phù hợp nhất cho mỗi khách hàng. Tích hợp các đặc trưng về trọng số giảm dần theo thời gian (Time-decay) và xu hướng bán hàng (Sales trend).

##  Cấu trúc mã nguồn

* `data/`: Chứa dữ liệu gốc (transactions, articles, customers).
* `outputs/`: Chứa các file parquet trung gian, models và kết quả (Đã được gitignore).
* `src/`: Thư mục mã nguồn chính.
  * `config.py`: Khai báo tham số, đường dẫn và danh sách features.
  * `data_loader.py`: Nạp và ép kiểu dữ liệu.
  * `candidates.py`: Các thuật toán sinh tập ứng viên.
  * `embeddings.py`: Trích xuất vector đặc trưng từ mô tả (Text) và hình ảnh (Image).
  * `features.py`: Trích xuất đặc trưng lịch sử mua hàng, time-decay.
  * `train.py`: Lấy mẫu âm (Negative Sampling) và huấn luyện mô hình LightGBM.
  * `predict.py`: Dự đoán, xếp hạng và xử lý Fallback cho tập test.
  * `evaluate.py`: Tính toán chỉ số MAP@12 và chạy kiểm định Ablation.
* `main.py`: File điều hướng chạy toàn bộ Pipeline.

##  Hướng dẫn cài đặt và chạy

**Bước 1: Cài đặt môi trường**
pip install -r requirements.txt

**Bước 2: Cấp phát dữ liệu**
* Đặt 3 file `articles.csv`, `customers.csv`, `transactions_train.csv` vào thư mục `data/`.

**Bước 3: Khởi chạy luồng xử lý**
Chạy trực tiếp file `main.py` để thực thi toàn bộ pipeline từ sinh ứng viên đến đánh giá mô hình:
python main.py
*(Bạn có thể bật/tắt các công đoạn trong file main.py bằng các cờ Boolean).*

##  Kết quả đánh giá
* Metric: **MAP@12**
* Kết quả mô hình V2: **0.0273**
* Có đính kèm file Ablation Study (`outputs/ablation_v2.csv`) để đối chiếu mức độ đóng góp của từng thuật toán sinh ứng viên vào kết quả tổng thể.