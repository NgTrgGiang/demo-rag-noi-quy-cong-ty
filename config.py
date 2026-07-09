"""
config.py - Nơi DUY NHẤT để cấu hình dự án.

Muốn đổi nhà cung cấp (OpenAI / Gemini / Ollama) hay đổi tham số RAG
(chunk_size, top_k...) thì chỉ cần sửa ở file này.

Ba hàm quan trọng ở cuối file:
  - embed_texts(texts): nhúng NHIỀU đoạn văn bản -> danh sách vector (dùng khi ingest).
  - embed_query(text):  nhúng 1 câu hỏi -> 1 vector (dùng khi truy hồi).
  - chat(system, user): gọi LLM sinh câu trả lời.
ingest.py và rag.py chỉ gọi 3 hàm này, KHÔNG cần biết đang dùng provider nào.
"""

import os

from dotenv import load_dotenv

# Đọc các biến môi trường từ file .env (ví dụ OPENAI_API_KEY)
load_dotenv()

# ============================================================
# 1) CHỌN NHÀ CUNG CẤP (PROVIDER)
# ============================================================
# Đổi 1 dòng dưới đây để chuyển provider: "openai" | "gemini" | "ollama"
# Có thể ghi đè bằng biến môi trường PROVIDER trong file .env.
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# ============================================================
# 2) TÊN MODEL THEO TỪNG PROVIDER
# ============================================================
MODELS = {
    "openai": {
        "embedding": "text-embedding-3-small",  # rẻ, chất lượng tốt
        "chat": "gpt-4o-mini",
    },
    "gemini": {
        "embedding": "models/text-embedding-004",
        "chat": "gemini-1.5-flash",
    },
    "ollama": {
        # Nhớ chạy: ollama pull nomic-embed-text && ollama pull llama3.2
        "embedding": "nomic-embed-text",
        "chat": "llama3.2",
    },
}

# ============================================================
# 3) THAM SỐ RAG (bạn có thể thử nghiệm để cải thiện chất lượng)
# ============================================================
CHUNK_SIZE = 500  # độ dài mỗi đoạn (số ký tự)
CHUNK_OVERLAP = 50  # phần gối đầu giữa 2 đoạn liền nhau (giữ ngữ cảnh liền mạch)
TOP_K = 4  # số đoạn liên quan nhất lấy ra để làm ngữ cảnh

# ============================================================
# 4) NƠI LƯU VECTOR STORE (ChromaDB) - có persist để không phải nạp lại
# ============================================================
DATA_DIR = "data"  # thư mục chứa tài liệu nguồn
CHROMA_DIR = "chroma_db"  # thư mục Chroma lưu dữ liệu xuống ổ đĩa
COLLECTION_NAME = "faq_noi_quy"  # tên "bảng" trong Chroma

# ============================================================
# 5) HÀM CẦU NỐI - nhúng văn bản & gọi LLM (tự chọn theo PROVIDER)
# ============================================================
# Lưu ý: import SDK theo kiểu "lazy" (chỉ import khi cần) để bạn không phải
# cài đủ cả 3 thư viện - chỉ cài thư viện của provider bạn đang dùng.


def _model(kind: str) -> str:
    """Lấy tên model theo provider hiện tại. kind = 'embedding' hoặc 'chat'."""
    if PROVIDER not in MODELS:
        raise ValueError(f"PROVIDER không hợp lệ: {PROVIDER}. Chọn openai | gemini | ollama.")
    return MODELS[PROVIDER][kind]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Nhúng một DANH SÁCH đoạn văn bản thành danh sách vector."""
    if PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()  # tự đọc OPENAI_API_KEY từ môi trường
        resp = client.embeddings.create(model=_model("embedding"), input=texts)
        return [item.embedding for item in resp.data]

    elif PROVIDER == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        vectors = []
        for t in texts:
            r = genai.embed_content(model=_model("embedding"), content=t)
            vectors.append(r["embedding"])
        return vectors

    elif PROVIDER == "ollama":
        import ollama

        vectors = []
        for t in texts:
            r = ollama.embeddings(model=_model("embedding"), prompt=t)
            vectors.append(r["embedding"])
        return vectors

    raise ValueError(f"PROVIDER không hợp lệ: {PROVIDER}")


def embed_query(text: str) -> list[float]:
    """Nhúng 1 câu hỏi thành 1 vector (tái dùng embed_texts cho gọn)."""
    return embed_texts([text])[0]


def chat(system_prompt: str, user_prompt: str) -> str:
    """Gọi LLM sinh câu trả lời. Nhận prompt hệ thống + prompt người dùng."""
    if PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=_model("chat"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # 0 = trả lời ổn định, ít "bịa"
        )
        return resp.choices[0].message.content

    elif PROVIDER == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(_model("chat"), system_instruction=system_prompt)
        resp = model.generate_content(user_prompt)
        return resp.text

    elif PROVIDER == "ollama":
        import ollama

        resp = ollama.chat(
            model=_model("chat"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.0},
        )
        return resp["message"]["content"]

    raise ValueError(f"PROVIDER không hợp lệ: {PROVIDER}")
