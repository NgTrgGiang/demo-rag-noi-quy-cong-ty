"""
eval.py — Chấm điểm chất lượng bot theo 3 TẦNG (chuẩn đánh giá RAG).

Vì sao tách 3 tầng? Để khi điểm thấp, bạn biết NÊN SỬA KHÂU NÀO:
  TẦNG 1 - RETRIEVAL  : Kho vector có lấy ĐÚNG đoạn chứa đáp án không?
      • Hit Rate@k : tỉ lệ câu mà trong top-k có ít nhất 1 đoạn đúng.
      • MRR@k      : trung bình của 1/thứ_hạng đoạn đúng đầu tiên
                     (đoạn đúng đứng càng trên → điểm càng cao; hạng 1 = 1.0).
      → Điểm thấp = lỗi ở khâu TÌM (chunking/embedding/top_k), KHÔNG phải LLM.

  TẦNG 2 - KEYWORD    : Câu trả lời có chứa từ khóa bắt buộc không?
      → Nhanh, miễn phí, deterministic. Dùng làm "baseline" kiểm tra thô.

  TẦNG 3 - LLM-JUDGE  : Dùng LLM chấm câu trả lời có ĐÚNG Ý không (kể cả diễn đạt khác).
      → Sát với đánh giá của con người nhất. Tốn 1 lượt gọi LLM mỗi câu.
      → Tắt bằng cờ:  python eval.py --no-judge

Chạy:  python eval.py            (chạy đủ 3 tầng)
       python eval.py --no-judge (bỏ tầng LLM-judge để khỏi tốn tiền)
"""

import argparse
import json
import time
import unicodedata

import config
import rag

# ============================================================
# PROMPT cho LLM-as-judge (tầng 3)
# ============================================================
JUDGE_SYSTEM = """Bạn là giám khảo chấm câu trả lời của một chatbot hỏi-đáp tài liệu.
Bạn nhận: CÂU HỎI, ĐÁP ÁN MONG ĐỢI (ý đúng), và CÂU TRẢ LỜI của bot.
Nhiệm vụ: quyết định câu trả lời của bot có ĐÚNG Ý so với đáp án mong đợi không.

Nguyên tắc:
- Bỏ qua khác biệt về cách diễn đạt/độ dài, chỉ xét ĐÚNG hay SAI về nội dung.
- Nếu đáp án mong đợi là bot PHẢI TỪ CHỐI (không có thông tin trong tài liệu),
  thì bot chỉ ĐẠT khi nó thực sự từ chối/nói không có thông tin, KHÔNG bịa.

Định dạng trả lời BẮT BUỘC: dòng đầu tiên CHỈ gồm một từ PASS hoặc FAIL.
Dòng sau (tuỳ chọn) ghi lý do ngắn gọn."""

JUDGE_USER = """CÂU HỎI: {question}
ĐÁP ÁN MONG ĐỢI: {expected}
CÂU TRẢ LỜI CỦA BOT: {answer}"""


def normalize(text: str) -> str:
    """Chuẩn hoá chuỗi để so khớp: chuẩn hoá Unicode + đưa về chữ thường."""
    return unicodedata.normalize("NFC", text or "").lower()


def contains_all(answer: str, keywords: list[str]) -> bool:
    """TẦNG 2: True nếu câu trả lời chứa TẤT CẢ từ khoá mong đợi."""
    a = normalize(answer)
    return all(normalize(k) in a for k in keywords)


def retrieval_rank(sources: list[dict], phrase: str | None):
    """TẦNG 1: tìm THỨ HẠNG (1-based) của đoạn liên quan đầu tiên trong kết quả truy hồi.

    - Trả về None nếu câu hỏi nằm NGOÀI tài liệu (phrase = null) -> không tính retrieval.
    - Trả về 0 nếu không có đoạn nào chứa cụm đặc trưng (miss).
    - Ngược lại trả về thứ hạng (1 = đoạn đúng nằm đầu tiên).
    """
    if not phrase:
        return None
    p = normalize(phrase)
    for i, s in enumerate(sources, start=1):
        if p in normalize(s["text"]):
            return i
    return 0


def llm_judge(question: str, expected: str, answer: str) -> bool:
    """TẦNG 3: gọi LLM chấm PASS/FAIL."""
    resp = config.chat(
        JUDGE_SYSTEM,
        JUDGE_USER.format(question=question, expected=expected, answer=answer),
    )
    # Lấy dòng đầu, chuẩn hoá để đọc verdict
    first_line = (resp or "").strip().splitlines()[0].upper() if (resp or "").strip() else ""
    return "PASS" in first_line and "FAIL" not in first_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-judge", action="store_true",
                        help="Bỏ qua tầng LLM-as-judge (không tốn tiền gọi LLM chấm điểm).")
    args = parser.parse_args()
    use_judge = not args.no_judge

    with open("eval_questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Bộ đếm cho từng tầng
    kw_correct = 0        # tầng 2
    judge_correct = 0     # tầng 3
    hit_count = 0         # tầng 1: số câu có "hit"
    reciprocal_sum = 0.0  # tầng 1: tổng 1/rank để tính MRR
    retrieval_n = 0       # số câu có tính retrieval (loại câu ngoài tài liệu)
    total_time = 0.0

    print("=" * 78)
    print(f"BẮT ĐẦU CHẤM ĐIỂM  (provider={config.PROVIDER}, top_k={config.TOP_K}, "
          f"LLM-judge={'BẬT' if use_judge else 'TẮT'})")
    print("=" * 78)
    # Tiêu đề bảng chi tiết
    print(f"{'#':>2}  {'Keyword':<8} {'Judge':<7} {'R-rank':<7} {'Time':<7} Câu hỏi")
    print("-" * 78)

    for i, item in enumerate(questions, start=1):
        q = item["question"]

        # Gọi RAG (đo thời gian trả lời thực tế)
        start = time.time()
        result = rag.answer(q)
        elapsed = time.time() - start
        total_time += elapsed

        answer_text = result["answer"]
        sources = result["sources"]

        # --- Tầng 1: Retrieval ---
        rank = retrieval_rank(sources, item.get("relevant_contains"))
        if rank is not None:  # câu có trong tài liệu -> mới tính retrieval
            retrieval_n += 1
            if rank > 0:
                hit_count += 1
                reciprocal_sum += 1.0 / rank
            rank_str = str(rank) if rank > 0 else "miss"
        else:
            rank_str = "n/a"  # câu ngoài tài liệu

        # --- Tầng 2: Keyword ---
        kw_ok = contains_all(answer_text, item["keywords"])
        if kw_ok:
            kw_correct += 1

        # --- Tầng 3: LLM-judge ---
        if use_judge:
            judge_ok = llm_judge(q, item["expected"], answer_text)
            if judge_ok:
                judge_correct += 1
            judge_str = "PASS" if judge_ok else "FAIL"
        else:
            judge_str = "-"

        kw_str = "PASS" if kw_ok else "FAIL"
        print(f"{i:>2}  {kw_str:<8} {judge_str:<7} {rank_str:<7} "
              f"{elapsed:>5.2f}s  {q[:34]}")

    # ============================================================
    # BẢNG TỔNG HỢP
    # ============================================================
    n = len(questions)
    hit_rate = hit_count / retrieval_n if retrieval_n else 0.0
    mrr = reciprocal_sum / retrieval_n if retrieval_n else 0.0

    print("\n" + "=" * 78)
    print("KẾT QUẢ TỔNG HỢP")
    print("=" * 78)
    print("TẦNG 1 - RETRIEVAL (chất lượng khâu tìm đoạn):")
    print(f"    Hit Rate@{config.TOP_K}      : {hit_count}/{retrieval_n}  ({hit_rate * 100:.1f}%)")
    print(f"    MRR@{config.TOP_K}           : {mrr:.3f}   (1.0 = đoạn đúng luôn ở vị trí số 1)")
    print("TẦNG 2 - KEYWORD (baseline nhanh):")
    print(f"    Số câu đúng        : {kw_correct}/{n}  ({kw_correct / n * 100:.1f}%)")
    if use_judge:
        print("TẦNG 3 - LLM-JUDGE (sát đánh giá con người nhất):")
        print(f"    Số câu đúng        : {judge_correct}/{n}  ({judge_correct / n * 100:.1f}%)")
    else:
        print("TẦNG 3 - LLM-JUDGE : (đã tắt bằng --no-judge)")
    print("-" * 78)
    print(f"Thời gian TB mỗi câu   : {total_time / n:.2f}s")
    print("=" * 78)
    print("\nGợi ý đọc kết quả:")
    print("  • Hit Rate/MRR thấp  -> sửa khâu TÌM: tăng top_k, chỉnh chunk_size, đổi embedding.")
    print("  • Retrieval cao nhưng Judge thấp -> sửa khâu TRẢ LỜI: chỉnh SYSTEM_PROMPT trong rag.py.")
    print("  • Keyword và Judge lệch nhau -> keyword đang chấm quá thô, tin Judge hơn.")


if __name__ == "__main__":
    main()
