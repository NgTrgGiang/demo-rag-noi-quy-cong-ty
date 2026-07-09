"""
ingest.py - Nạp tài liệu vào kho vector (ChromaDB).

Chứa các hàm dùng chung cho CẢ HAI luồng:
  - CLI:  python ingest.py   -> đọc data/ trên đĩa, lưu vào Chroma persist.
  - App:  app.py             -> đọc file người dùng upload, lưu vào Chroma in-memory.

Luồng xử lý: đọc text -> cắt nhỏ (chunk) -> nhúng (embed) -> add vào collection.
"""

import glob
import io
import os

import config


def _make_splitter():
    """Tạo bộ cắt văn bản (import muộn để đọc file upload không cần cài langchain)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def _pdf_to_text(source) -> str:
    """Đọc PDF (source có thể là đường dẫn hoặc file-like/BytesIO) -> text ghép các trang."""
    from pypdf import PdfReader

    reader = PdfReader(source)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def doc_text(path: str) -> str:
    """Đọc nội dung 1 file trên đĩa thành chuỗi text. Hỗ trợ .pdf, .md, .txt."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _pdf_to_text(path)
    # .md và .txt: đọc trực tiếp dạng văn bản
    with open(path, encoding="utf-8") as f:
        return f.read()


def text_from_upload(name: str, data: bytes) -> str:
    """Đọc text từ file người dùng UPLOAD (dạng bytes). Dùng trong app.py."""
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        return _pdf_to_text(io.BytesIO(data))
    # .md và .txt: giải mã UTF-8, bỏ qua ký tự lỗi cho an toàn
    return data.decode("utf-8", errors="ignore")


def load_documents(data_dir: str) -> list[dict]:
    """Đọc tất cả file trong data/ -> danh sách {'source': tên_file, 'text': nội_dung}."""
    docs = []
    for pattern in ("*.pdf", "*.md", "*.txt"):
        for path in glob.glob(os.path.join(data_dir, pattern)):
            text = doc_text(path).strip()
            if text:  # bỏ qua file rỗng
                docs.append({"source": os.path.basename(path), "text": text})
    return docs


def count_chunks(docs: list[dict]) -> int:
    """Đếm trước số đoạn sẽ tạo (để kiểm tra giới hạn MAX_CHUNKS trước khi embed)."""
    splitter = _make_splitter()
    return sum(len(splitter.split_text(doc["text"])) for doc in docs)


def index_documents(collection, docs: list[dict], settings=None) -> int:
    """Chunk + embed + add danh sách tài liệu vào MỘT collection Chroma.

    Dùng chung cho CLI (collection persist) và app (collection in-memory).
    `settings`: config.Settings của phiên (None = dùng mặc định). Trả về số đoạn đã thêm.
    """
    splitter = _make_splitter()

    chunks, metadatas, ids = [], [], []
    for doc in docs:
        pieces = splitter.split_text(doc["text"])
        for i, piece in enumerate(pieces):
            chunks.append(piece)
            # Metadata để TRÍCH NGUỒN được: tên file + số thứ tự đoạn
            metadatas.append({"source": doc["source"], "chunk": i})
            ids.append(f"{doc['source']}::chunk::{i}")

    if not chunks:
        return 0

    embeddings = config.embed_texts(chunks, settings)
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def main():
    """Luồng CLI: đọc data/ -> lưu vào Chroma persist (chạy: python ingest.py)."""
    import chromadb

    docs = load_documents(config.DATA_DIR)
    if not docs:
        print(f"[!] Không tìm thấy tài liệu nào trong '{config.DATA_DIR}/'.")
        print("    Hãy đặt file .pdf / .md / .txt vào đó rồi chạy lại.")
        return
    print(f"Đã đọc {len(docs)} tài liệu: {[d['source'] for d in docs]}")

    # Tạo lại collection persistent (xoá cũ để không trùng dữ liệu)
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # chưa có thì bỏ qua
    collection = client.create_collection(config.COLLECTION_NAME)

    print(f"Đang cắt & nhúng bằng provider '{config.PROVIDER}'...")
    n = index_documents(collection, docs)
    print(
        f"Đã cắt thành {n} đoạn (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})."
    )
    print(f"Xong! Đã lưu {collection.count()} đoạn vào Chroma tại '{config.CHROMA_DIR}/'.")
    print("   Giờ bạn có thể chạy:  streamlit run app.py")


if __name__ == "__main__":
    main()
