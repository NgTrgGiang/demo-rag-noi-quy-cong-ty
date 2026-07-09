"""Test phần lõi rag.py (không gọi API, không cần ChromaDB)."""

import rag


def test_build_context_co_nhan_nguon_va_ten_file():
    """Ngữ cảnh ghép ra phải kèm nhãn nguồn + tên file để LLM trích dẫn được."""
    chunks = [
        {"label": "Nguồn 1", "source": "noi_quy.md", "chunk": 0, "text": "Giờ làm 8h30"},
        {"label": "Nguồn 2", "source": "noi_quy.md", "chunk": 1, "text": "Nghỉ phép 12 ngày"},
    ]
    ctx = rag._build_context(chunks)
    assert "Nguồn 1" in ctx
    assert "Nguồn 2" in ctx
    assert "noi_quy.md" in ctx
    assert "8h30" in ctx


def test_system_prompt_co_cau_tu_choi_chuan():
    """Câu từ chối phải cố định đúng nguyên văn (bất biến quan trọng của sản phẩm)."""
    assert "Mình không tìm thấy thông tin này trong tài liệu." in rag.SYSTEM_PROMPT


def test_answer_tra_ve_tu_choi_khi_khong_co_doan(monkeypatch):
    """Nếu retrieve() không trả về đoạn nào -> answer() phải từ chối, không gọi LLM."""
    monkeypatch.setattr(rag, "retrieve", lambda question: [])
    out = rag.answer("Câu hỏi bất kỳ")
    assert out["answer"] == "Mình không tìm thấy thông tin này trong tài liệu."
    assert out["sources"] == []
