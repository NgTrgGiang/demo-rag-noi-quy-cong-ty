"""
app.py - Giao diện chat bằng Streamlit.

Chạy:  streamlit run app.py
(Nhớ chạy 'python ingest.py' trước để tạo kho vector.)
"""

import streamlit as st

import config
import rag

# ---- Cấu hình trang ----
st.set_page_config(page_title="Bot hỏi đáp Nội quy công ty")
st.title("Bot hỏi đáp Nội quy công ty (RAG)")
st.caption(
    f"Trả lời dựa trên tài liệu trong thư mục `data/` | Provider: "
    f"**{config.PROVIDER}** | top_k = {config.TOP_K}"
)

# ---- Lịch sử hội thoại lưu trong phiên làm việc (session_state) ----
# Mỗi phần tử: {"role": "user"/"assistant", "content": ..., "sources": [...]}
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Thanh bên: nút xóa hội thoại + gợi ý câu hỏi ----
with st.sidebar:
    st.header("Tùy chọn")
    if st.button("Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Câu hỏi mẫu để thử:**")
    st.markdown(
        "- Giờ làm việc hành chính là mấy giờ?\n"
        "- Được nghỉ phép năm bao nhiêu ngày?\n"
        "- Xin nghỉ việc phải báo trước bao lâu?\n"
        "- Công ty có bán bảo hiểm xe máy không? *(câu ngoài tài liệu)*"
    )


def render_sources(sources: list[dict]):
    """Hiển thị phần 'Nguồn tham khảo' dạng mở rộng được."""
    if not sources:
        return
    with st.expander("Nguồn tham khảo"):
        for s in sources:
            st.markdown(
                f"**[{s['label']}]** | `{s['source']}` | đoạn {s['chunk']} "
                f"| độ gần: {s['distance']:.3f}"
            )
            st.caption(s["text"])


# ---- Vẽ lại toàn bộ lịch sử hội thoại ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

# ---- Ô nhập câu hỏi ----
if question := st.chat_input("Nhập câu hỏi về nội quy công ty..."):
    # 1) Hiển thị + lưu câu hỏi của người dùng
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2) Gọi RAG để lấy câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu tài liệu..."):
            try:
                result = rag.answer(question)
                answer_text = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer_text = f"Lỗi: {e}"
                sources = []

        st.markdown(answer_text)
        render_sources(sources)

    # 3) Lưu câu trả lời vào lịch sử
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "sources": sources}
    )
