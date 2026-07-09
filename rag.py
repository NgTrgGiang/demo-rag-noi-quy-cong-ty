"""
rag.py - Phần lõi RAG (dùng lại cho cả app.py và eval.py).

Luồng của hàm answer(question):
  câu hỏi -> nhúng thành vector -> tìm top_k đoạn gần nghĩa nhất trong Chroma
          -> ghép các đoạn thành "ngữ cảnh" -> đưa vào prompt có luật nghiêm ngặt
          -> LLM sinh câu trả lời (chỉ dựa trên ngữ cảnh) + trích nguồn.
"""

import config

# ============================================================
# PROMPT HỆ THỐNG - ép LLM tuân thủ luật để KHÔNG bịa
# ============================================================
SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp nội bộ, chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.

QUY TẮC BẮT BUỘC:
1. CHỈ dùng thông tin trong phần NGỮ CẢNH bên dưới để trả lời. Tuyệt đối không dùng kiến thức ngoài.
2. Nếu NGỮ CẢNH không chứa đủ thông tin để trả lời, hãy trả lời ĐÚNG NGUYÊN VĂN câu này và không thêm gì khác:
   "Mình không tìm thấy thông tin này trong tài liệu."
3. Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm, không suy diễn.
4. Nếu (và chỉ khi) bạn có trả lời được từ ngữ cảnh, hãy KẾT THÚC bằng một mục:
   "Nguồn tham khảo:" rồi liệt kê các nhãn nguồn đã dùng, ví dụ: [Nguồn 1], [Nguồn 2].
"""

# Khung prompt người dùng: nhồi ngữ cảnh + câu hỏi vào
USER_TEMPLATE = """NGỮ CẢNH:
{context}

CÂU HỎI: {question}

Hãy trả lời theo đúng các QUY TẮC BẮT BUỘC ở trên."""


def _get_collection():
    """Mở collection Chroma đã persist. Báo lỗi rõ ràng nếu chưa ingest."""
    # Import muộn (lazy) để việc chỉ dùng các hàm khác không cần cài sẵn chromadb.
    import chromadb

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        return client.get_collection(config.COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            "Chưa có dữ liệu trong Chroma. Hãy chạy 'python ingest.py' trước."
        ) from None


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Truy hồi top_k đoạn liên quan nhất tới câu hỏi.

    Trả về danh sách dict: {label, source, chunk, text, distance}.
    - distance càng NHỎ nghĩa là càng gần nghĩa (liên quan hơn).
    """
    top_k = top_k or config.TOP_K
    collection = _get_collection()

    # Nhúng câu hỏi rồi truy vấn Chroma
    q_vector = config.embed_query(question)
    res = collection.query(
        query_embeddings=[q_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists, strict=False), start=1):
        results.append(
            {
                "label": f"Nguồn {i}",  # nhãn để LLM trích dẫn
                "source": meta.get("source", "?"),  # tên file
                "chunk": meta.get("chunk", "?"),  # vị trí đoạn trong file
                "text": doc,  # nội dung đoạn
                "distance": dist,
            }
        )
    return results


def _build_context(chunks: list[dict]) -> str:
    """Ghép các đoạn thành 1 khối ngữ cảnh có đánh nhãn để LLM trích dẫn."""
    parts = []
    for c in chunks:
        header = f"[{c['label']}] (tài liệu: {c['source']}, đoạn {c['chunk']})"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n".join(parts)


def answer(question: str) -> dict:
    """Trả lời 1 câu hỏi bằng RAG.

    Trả về dict: {'answer': <chuỗi trả lời>, 'sources': <danh sách đoạn đã dùng>}.
    """
    chunks = retrieve(question)

    # Nếu kho rỗng (không có đoạn nào) -> trả lời từ chối luôn cho an toàn
    if not chunks:
        return {
            "answer": "Mình không tìm thấy thông tin này trong tài liệu.",
            "sources": [],
        }

    context = _build_context(chunks)
    user_prompt = USER_TEMPLATE.format(context=context, question=question)

    # Gọi LLM (hàm chat tự chọn provider theo config.PROVIDER)
    reply = config.chat(SYSTEM_PROMPT, user_prompt).strip()

    return {"answer": reply, "sources": chunks}


# Cho phép test nhanh từ dòng lệnh: python rag.py "câu hỏi của bạn"
if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Giờ làm việc hành chính là mấy giờ?"
    out = answer(q)
    print("Câu hỏi:", q)
    print("\nTrả lời:", out["answer"])
    print("\nCác đoạn đã dùng:")
    for s in out["sources"]:
        print(f"  - [{s['label']}] {s['source']} (đoạn {s['chunk']}, distance={s['distance']:.3f})")
