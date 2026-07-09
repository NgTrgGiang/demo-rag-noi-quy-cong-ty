"""Test đọc file upload trong ingest.py (không cần chromadb / langchain / API)."""

import ingest


def test_text_from_upload_txt_giu_nguyen_noi_dung():
    data = "Xin chào\nĐây là nội quy".encode()
    assert ingest.text_from_upload("noi_quy.txt", data) == "Xin chào\nĐây là nội quy"


def test_text_from_upload_md_doc_duoc_tieng_viet():
    data = "# Tiêu đề\nGiờ làm việc 8h30".encode()
    text = ingest.text_from_upload("faq.md", data)
    assert "Tiêu đề" in text
    assert "8h30" in text


def test_text_from_upload_bo_qua_byte_loi():
    # Byte không hợp lệ UTF-8 phải bị bỏ qua thay vì gây lỗi
    data = b"hello \xff\xfe world"
    text = ingest.text_from_upload("a.txt", data)
    assert "hello" in text and "world" in text
