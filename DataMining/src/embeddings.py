import gc
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path

# Import thư mục cấu hình
from src.config import EMB_DIR, DATA_DIR
from src.data_loader import load_data

def generate_text_embeddings(articles):
    """
    Sử dụng SentenceTransformers để đọc hiểu mô tả sản phẩm 
    và biến chúng thành vector (Text Embeddings).
    """
    out_path = EMB_DIR / 'article_embeddings.npy'
    if out_path.exists():
        print('article_embeddings.npy đã tồn tại — Bỏ qua chạy lại.')
        return

    print(f'[{datetime.now().strftime("%H:%M:%S")}] Bắt đầu sinh Text Embeddings...')
    
    # Gom các trường text lại thành một câu miêu tả hoàn chỉnh
    articles_text = articles.with_columns(
        (pl.col('prod_name').fill_null('') + ' ' +
         pl.col('product_type_name').fill_null('') + ' ' +
         pl.col('colour_group_name').fill_null('') + ' ' +
         pl.col('department_name').fill_null('') + ' ' +
         pl.col('detail_desc').fill_null('')).alias('text')
    ).select(['article_id', 'text'])

    article_ids_list = articles_text['article_id'].to_list()
    texts = articles_text['text'].to_list()

    print(f'Số lượng sản phẩm cần nhúng: {len(article_ids_list):,}')

    # Khởi tạo mô hình ngôn ngữ
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Chạy encode (Quá trình này có thể tốn vài phút)
    text_embs = st_model.encode(
        texts, batch_size=512, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    )

    # Lưu ra đĩa
    np.save(EMB_DIR / 'article_embeddings.npy', text_embs)
    np.save(EMB_DIR / 'article_ids.npy', np.array(article_ids_list))
    
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Đã lưu text embeddings: {text_embs.shape}')
    del st_model, text_embs; gc.collect()

def generate_image_embeddings(article_ids_list, images_dir):
    """
    Sử dụng CLIP (OpenAI) để phân tích hình ảnh thực tế của quần áo.
    Lưu ý: Bạn cần tải thư mục ảnh của H&M về máy trước khi chạy hàm này.
    """
    import torch
    import open_clip
    from PIL import Image

    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"Lỗi: Không tìm thấy thư mục ảnh tại {images_dir}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Khởi động CLIP trên {device}')

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device).eval()

    IMG_DIM = 512
    img_embs = np.zeros((len(article_ids_list), IMG_DIM), dtype=np.float32)
    CLIP_BATCH = 512
    total = len(article_ids_list)
    missing = 0

    for start in range(0, total, CLIP_BATCH):
        batch_ids = article_ids_list[start:start + CLIP_BATCH]
        imgs = []
        for aid in batch_ids:
            img_path = images_path / aid[:3] / f'{aid}.jpg'
            try:
                imgs.append(clip_preprocess(Image.open(img_path).convert('RGB')))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
                missing += 1

        batch_tensor = torch.stack(imgs).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = clip_model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            feats = feats.cpu().numpy().astype(np.float32)

        img_embs[start:start + CLIP_BATCH] = feats

        if start % (CLIP_BATCH * 10) == 0:
            print(f'  CLIP tiến độ: {start}/{total} — VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB')

    np.save(EMB_DIR / 'image_embeddings.npy', img_embs)
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Hoàn thành Image Embeddings. Lỗi/Thiếu ảnh: {missing:,}')

if __name__ == "__main__":
    # Nạp dữ liệu articles để lấy ID và Text
    _, _, _, articles, _ = load_data()
    
    # 1. Chạy sinh Vector cho Text (Chỉ tốn CPU/RAM, chạy khá nhanh)
    generate_text_embeddings(articles)
    
    # 2. Chạy sinh Vector cho Ảnh (Cần GPU và tải file ảnh nặng hàng chục GB)
    # LƯU Ý: Ở file gốc của bạn, quá trình này bị sập (KeyboardInterrupt) do lỗi đường dẫn ảnh hoặc tràn VRAM.
    # Hãy đảm bảo bạn đã cấu hình thư mục chứa ảnh (vd: /content/images) trước khi mở comment dòng dưới.
    
    # article_ids = np.load(EMB_DIR / 'article_ids.npy', allow_pickle=True)
    # generate_image_embeddings(article_ids, images_dir='/đường/dẫn/đến/thư/mục/ảnh/H&M')