"""
config.py - Nơi cấu hình dự án.

- Các HẰNG mặc định (provider, model, chunk_size, top_k...) dùng cho luồng CLI.
- Lớp `Settings`: gói cấu hình cho MỘT phiên (provider + key + tên model). App truyền
  Settings này xuống để mỗi người dùng tự chọn provider/model/key mà không đụng biến
  toàn cục (an toàn khi nhiều người dùng chung server).

Ba hàm cầu nối: embed_texts / embed_query / chat - nhận `settings` (None = dùng mặc định).
ingest.py và rag.py chỉ gọi 3 hàm này, không cần biết chi tiết provider.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Đọc các biến môi trường từ file .env (ví dụ OPENAI_API_KEY)
load_dotenv()

# ============================================================
# 1) PROVIDER + MODEL MẶC ĐỊNH (dùng cho luồng CLI khi không truyền Settings)
# ============================================================
PROVIDER = os.getenv("PROVIDER", "openai").lower()  # openai | gemini | ollama
PROVIDERS = ["openai", "gemini", "ollama"]

# Model mặc định của mỗi provider.
MODELS = {
    "openai": {"embedding": "text-embedding-3-small", "chat": "gpt-4o-mini"},
    "gemini": {"embedding": "models/text-embedding-004", "chat": "gemini-1.5-flash"},
    "ollama": {"embedding": "nomic-embed-text", "chat": "llama3.2"},
}

# Danh sách model cho người dùng CHỌN trên app (phần tử đầu là mặc định).
MODEL_CHOICES = {
    "openai": {
        "chat": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        "embedding": ["text-embedding-3-small", "text-embedding-3-large"],
    },
    "gemini": {
        "chat": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "embedding": ["models/text-embedding-004"],
    },
    "ollama": {
        "chat": ["llama3.2", "llama3.1", "qwen2.5"],
        "embedding": ["nomic-embed-text", "mxbai-embed-large"],
    },
}

# ============================================================
# 2) THAM SỐ RAG + VECTOR STORE
# ============================================================
CHUNK_SIZE = 500  # độ dài mỗi đoạn (số ký tự)
CHUNK_OVERLAP = 50  # phần gối đầu giữa 2 đoạn liền nhau (giữ ngữ cảnh liền mạch)
TOP_K = 4  # số đoạn liên quan nhất lấy ra để làm ngữ cảnh

DATA_DIR = "data"  # thư mục chứa tài liệu nguồn (luồng CLI)
CHROMA_DIR = "chroma_db"  # thư mục Chroma lưu dữ liệu xuống ổ đĩa (luồng CLI)
COLLECTION_NAME = "faq_noi_quy"  # tên "bảng" trong Chroma

# Giới hạn khi người dùng tự upload tài liệu trên app (tránh nạp quá lớn, chậm/tốn kém).
MAX_UPLOAD_MB = 5  # tổng dung lượng file upload tối đa (MB)
MAX_CHUNKS = 400  # số đoạn tối đa sau khi cắt (chặn embed quá nhiều)


# ============================================================
# 3) GÓI CẤU HÌNH THEO PHIÊN
# ============================================================
@dataclass
class Settings:
    """Cấu hình cho 1 phiên: provider + key + tên model.

    Để None ở các trường model -> tự lấy model mặc định của provider.
    Để None ở api_key -> đọc từ biến môi trường (.env) - dùng cho luồng CLI.
    """

    provider: str = PROVIDER
    api_key: str | None = None
    embedding_model: str | None = None
    chat_model: str | None = None

    def _check(self):
        if self.provider not in MODELS:
            raise ValueError(f"Provider không hợp lệ: {self.provider}. Chọn {PROVIDERS}.")

    def emb_model(self) -> str:
        self._check()
        return self.embedding_model or MODELS[self.provider]["embedding"]

    def chat_model_name(self) -> str:
        self._check()
        return self.chat_model or MODELS[self.provider]["chat"]


# ============================================================
# 4) HÀM CẦU NỐI - nhúng văn bản & gọi LLM (import SDK theo kiểu lazy)
# ============================================================
def embed_texts(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Nhúng một DANH SÁCH đoạn văn bản thành danh sách vector."""
    s = settings or Settings()
    if s.provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=s.api_key)  # api_key=None -> đọc OPENAI_API_KEY từ môi trường
        resp = client.embeddings.create(model=s.emb_model(), input=texts)
        return [item.embedding for item in resp.data]

    elif s.provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=s.api_key or os.getenv("GOOGLE_API_KEY"))
        return [genai.embed_content(model=s.emb_model(), content=t)["embedding"] for t in texts]

    elif s.provider == "ollama":
        import ollama

        return [ollama.embeddings(model=s.emb_model(), prompt=t)["embedding"] for t in texts]

    raise ValueError(f"Provider không hợp lệ: {s.provider}")


def embed_query(text: str, settings: Settings | None = None) -> list[float]:
    """Nhúng 1 câu hỏi thành 1 vector (tái dùng embed_texts cho gọn)."""
    return embed_texts([text], settings)[0]


def chat(system_prompt: str, user_prompt: str, settings: Settings | None = None) -> str:
    """Gọi LLM sinh câu trả lời. Nhận prompt hệ thống + prompt người dùng."""
    s = settings or Settings()
    if s.provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=s.api_key)
        resp = client.chat.completions.create(
            model=s.chat_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # 0 = trả lời ổn định, ít "bịa"
        )
        return resp.choices[0].message.content

    elif s.provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=s.api_key or os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(s.chat_model_name(), system_instruction=system_prompt)
        return model.generate_content(user_prompt).text

    elif s.provider == "ollama":
        import ollama

        resp = ollama.chat(
            model=s.chat_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.0},
        )
        return resp["message"]["content"]

    raise ValueError(f"Provider không hợp lệ: {s.provider}")
