"""
app.py - Giao diện chat self-serve bằng Streamlit.

Người dùng tự chọn provider + model, nhập API key, và tải tài liệu của họ lên rồi hỏi đáp.
Tài liệu nạp vào kho vector IN-MEMORY theo từng phiên (không lưu đĩa, không lẫn dữ liệu
giữa người dùng). Cấu hình (provider/model/key) được "khoá" theo phiên lúc nạp tài liệu.

Chạy:  streamlit run app.py
"""

import streamlit as st

import config
import ingest
import rag

st.set_page_config(page_title="Bot hỏi đáp tài liệu (RAG)")
st.title("Bot hỏi đáp tài liệu của bạn (RAG)")
st.caption(
    "Chọn provider/model, nhập API key, tải tài liệu (PDF / Markdown / TXT) và hỏi đáp có "
    "trích nguồn. Dữ liệu xử lý trong bộ nhớ theo phiên, không lưu trữ."
)

# ---- Khởi tạo trạng thái phiên ----
st.session_state.setdefault("messages", [])
st.session_state.setdefault("collection", None)  # None = chưa nạp tài liệu
st.session_state.setdefault("doc_info", None)
st.session_state.setdefault("settings", None)  # config.Settings đã khoá lúc nạp


# ============================================================
# Hàm phụ
# ============================================================
def build_session_index(docs: list[dict], settings: config.Settings):
    """Tạo kho vector in-memory từ danh sách tài liệu, lưu vào session_state."""
    if settings.provider in ("openai", "gemini") and not settings.api_key:
        st.error(f"Hãy nhập API key cho provider '{settings.provider}' ở thanh bên.")
        return
    if not docs:
        st.warning("Chưa có tài liệu nào để nạp.")
        return

    # Kiểm tra giới hạn số đoạn TRƯỚC khi embed (tránh nạp quá lớn)
    n_chunks = ingest.count_chunks(docs)
    if n_chunks == 0:
        st.warning("Không đọc được nội dung văn bản (file rỗng, hoặc PDF là ảnh scan?).")
        return
    if n_chunks > config.MAX_CHUNKS:
        st.error(
            f"Tài liệu quá lớn: {n_chunks} đoạn (giới hạn {config.MAX_CHUNKS}). "
            "Hãy tách nhỏ tài liệu rồi thử lại."
        )
        return

    with st.spinner(f"Đang nhúng {n_chunks} đoạn vào kho vector..."):
        import chromadb

        client = chromadb.EphemeralClient()  # kho in-memory, mất khi đóng phiên
        collection = client.create_collection("session_docs")
        try:
            n = ingest.index_documents(collection, docs, settings=settings)
        except Exception as e:
            st.error(f"Lỗi khi nạp tài liệu (kiểm tra lại API key / model?): {e}")
            return

    # Lưu vào phiên (giữ cả client để kho không bị giải phóng)
    st.session_state.chroma_client = client
    st.session_state.collection = collection
    st.session_state.settings = settings
    st.session_state.doc_info = {"sources": [d["source"] for d in docs], "chunks": n}
    st.session_state.messages = []
    st.rerun()


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


# ============================================================
# Thanh bên: cấu hình (khi chưa nạp) hoặc điều khiển (khi đã nạp)
# ============================================================
with st.sidebar:
    st.header("Cấu hình")

    if st.session_state.collection is None:
        # Chọn provider -> model -> key (áp dụng cho lần "Nạp tài liệu" tiếp theo)
        provider = st.selectbox("Provider", config.PROVIDERS, key="provider")
        chat_model = st.selectbox("Model trả lời (LLM)", config.MODEL_CHOICES[provider]["chat"])
        emb_model = st.selectbox("Model embedding", config.MODEL_CHOICES[provider]["embedding"])

        if provider == "ollama":
            api_key = ""
            st.caption("Ollama chạy local - không cần API key.")
        else:
            api_key = st.text_input(
                f"{provider} API key",
                type="password",
                key="api_key",
                help="Key chỉ dùng trong phiên của bạn, không được lưu trữ.",
            )
        if provider == "openai":
            st.markdown("[Lấy OpenAI key ->](https://platform.openai.com/api-keys)")
        elif provider in ("gemini",):
            st.caption("Gemini/Ollama cần cài thêm thư viện (xem README).")
    else:
        # Đã nạp: hiển thị cấu hình đang dùng + nút điều khiển
        s = st.session_state.settings
        st.caption(
            f"Provider: **{s.provider}**\n\n"
            f"LLM: `{s.chat_model_name()}`\n\n"
            f"Embedding: `{s.emb_model()}`"
        )
        if st.button("Nạp tài liệu khác", use_container_width=True):
            st.session_state.collection = None
            st.session_state.doc_info = None
            st.session_state.settings = None
            st.session_state.messages = []
            st.rerun()
        if st.button("Xóa hội thoại", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# Màn hình chính
# ============================================================
if st.session_state.collection is None:
    # --- Trạng thái 1: chưa nạp tài liệu -> khu upload ---
    st.subheader("1) Tải tài liệu lên")
    uploaded = st.file_uploader(
        "Chọn file (PDF, Markdown, TXT) - có thể chọn nhiều file",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    st.caption(f"Giới hạn: tổng {config.MAX_UPLOAD_MB} MB, tối đa {config.MAX_CHUNKS} đoạn.")

    # Gói cấu hình từ các lựa chọn ở thanh bên
    settings = config.Settings(
        provider=provider,
        api_key=(api_key or None),
        embedding_model=emb_model,
        chat_model=chat_model,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Nạp tài liệu", type="primary", use_container_width=True):
            docs, total = [], 0
            for f in uploaded or []:
                data = f.getvalue()
                total += len(data)
                text = ingest.text_from_upload(f.name, data).strip()
                if text:
                    docs.append({"source": f.name, "text": text})
            if total > config.MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"Tổng dung lượng vượt {config.MAX_UPLOAD_MB} MB.")
            else:
                build_session_index(docs, settings)
    with col2:
        if st.button("Dùng tài liệu mẫu", use_container_width=True):
            build_session_index(ingest.load_documents(config.DATA_DIR), settings)

else:
    # --- Trạng thái 2: đã nạp tài liệu -> chat ---
    info = st.session_state.doc_info
    st.success(
        f"Đã nạp {len(info['sources'])} tài liệu · {info['chunks']} đoạn: "
        + ", ".join(info["sources"])
    )

    # Vẽ lại lịch sử hội thoại
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg.get("sources", []))

    if question := st.chat_input("Nhập câu hỏi về tài liệu của bạn..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Đang tra cứu tài liệu..."):
                try:
                    result = rag.answer(
                        question,
                        st.session_state.collection,
                        settings=st.session_state.settings,
                    )
                    answer_text = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer_text = f"Lỗi: {e}"
                    sources = []
            st.markdown(answer_text)
            render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": sources}
        )
