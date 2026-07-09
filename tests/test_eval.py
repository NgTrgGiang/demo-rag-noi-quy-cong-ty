"""Test logic chấm điểm trong eval.py (không gọi API)."""

import importlib

# 'eval' trùng tên hàm builtin nên import qua importlib cho rõ ràng.
eval_mod = importlib.import_module("eval")


def test_normalize_ve_chu_thuong():
    assert eval_mod.normalize("Giờ LÀM Việc") == "giờ làm việc"


def test_contains_all_dung_khi_du_tu_khoa():
    answer = "Giờ hành chính từ 8h30 đến 17h30."
    assert eval_mod.contains_all(answer, ["8h30", "17h30"]) is True


def test_contains_all_sai_khi_thieu_tu_khoa():
    answer = "Chỉ nhắc tới 8h30 thôi."
    assert eval_mod.contains_all(answer, ["8h30", "17h30"]) is False


def test_retrieval_rank_tim_thay_o_vi_tri_2():
    """Đoạn chứa cụm đặc trưng nằm ở vị trí thứ 2 -> rank = 2."""
    sources = [
        {"text": "nội dung không liên quan"},
        {"text": "Nhân viên có 12 ngày phép năm"},
        {"text": "đoạn khác"},
    ]
    assert eval_mod.retrieval_rank(sources, "12 ngày phép") == 2


def test_retrieval_rank_khong_tim_thay_tra_ve_0():
    sources = [{"text": "abc"}, {"text": "xyz"}]
    assert eval_mod.retrieval_rank(sources, "không có cụm này") == 0


def test_retrieval_rank_cau_ngoai_tai_lieu_tra_ve_none():
    """phrase = None (câu ngoài tài liệu) -> không tính retrieval."""
    assert eval_mod.retrieval_rank([{"text": "abc"}], None) is None
