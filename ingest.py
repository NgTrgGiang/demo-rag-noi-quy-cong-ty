"""
ingest.py - Nạp tài liệu vào kho vector (ChromaDB).

Luồng xử lý:
  data/ (PDF, MD, TXT)  ->  đọc text  ->  cắt nhỏ (chunk)
                        ->  nhúng (embed)  ->  lưu vào ChromaDB (persist ra ổ đĩa)

Chạy lại file này mỗi khi bạn THÊM hoặc SỬA tài liệu trong thư mục data/.
    python ingest.py
"""

import glob
import os

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def doc_text(path: str) -> str:
    """Đọc nội dung 1 file thành chuỗi text. Hỗ trợ .pdf, .md, .txt."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        # Đọc PDF theo từng trang rồi ghép lại
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)

    # .md và .txt: đọc trực tiếp dạng văn bản
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_documents(data_dir: str) -> list[dict]:
    """Đọc tất cả file trong data/ -> danh sách {'source': tên_file, 'text': nội_dung}."""
    docs = []
    # Duyệt các định dạng được hỗ trợ
    for pattern in ("*.pdf", "*.md", "*.txt"):
        for path in glob.glob(os.path.join(data_dir, pattern)):
            text = doc_text(path).strip()
            if text:  # bỏ qua file rỗng
                docs.append({"source": os.path.basename(path), "text": text})
    return docs


def main():
    # 1) Đọc tài liệu nguồn
    docs = load_documents(config.DATA_DIR)
    if not docs:
        print(f"[!] Không tìm thấy tài liệu nào trong '{config.DATA_DIR}/'.")
        print("    Hãy đặt file .pdf / .md / .txt vào đó rồi chạy lại.")
        return
    print(f"Đã đọc {len(docs)} tài liệu: {[d['source'] for d in docs]}")

    # 2) Cắt nhỏ (chunk) từng tài liệu
    #    RecursiveCharacterTextSplitter cố cắt theo đoạn/câu trước, tránh cắt giữa từ.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks, metadatas, ids = [], [], []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(piece)
            # Lưu metadata để sau này TRÍCH NGUỒN được: tên file + số thứ tự đoạn
            metadatas.append({"source": doc["source"], "chunk": i})
            ids.append(f"{doc['source']}::chunk::{i}")
    print(
        f"Đã cắt thành {len(chunks)} đoạn (chunk_size={config.CHUNK_SIZE}, "
        f"overlap={config.CHUNK_OVERLAP})."
    )

    # 3) Nhúng (embed) tất cả các đoạn thành vector
    print(f"Đang nhúng {len(chunks)} đoạn bằng provider '{config.PROVIDER}'...")
    embeddings = config.embed_texts(chunks)

    # 4) Lưu vào ChromaDB (persist ra thư mục config.CHROMA_DIR)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    # Xoá collection cũ (nếu có) để không bị trùng dữ liệu khi ingest lại
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # chưa có thì bỏ qua

    collection = client.create_collection(config.COLLECTION_NAME)
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Xong! Đã lưu {collection.count()} đoạn vào Chroma tại '{config.CHROMA_DIR}/'.")
    print("   Giờ bạn có thể chạy:  streamlit run app.py")


if __name__ == "__main__":
    main()
