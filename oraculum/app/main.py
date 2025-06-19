import torch
import streamlit as st
from views import upload_page, chat_page, faiss_page, qa_page
from faiss_db import init_faiss_index

torch.classes.__path__ = []  # add this line to manually set it to empty.

# Configuração inicial
st.set_page_config(
    page_title="MAP Turismo",
    page_icon="🌎",
    layout="wide"
)

def main():
    init_faiss_index()
    st.sidebar.title("Menu Principal")
    page = st.sidebar.radio("Navegação:",
                            ["💬 Chat com RAG", "📤 Upload e Processamento", "🧠 Gerador QA", "📂 FAISS Manager"])

    if page == "📤 Upload e Processamento":
        upload_page.show()
    elif page == "📂 FAISS Manager":
        faiss_page.show_faiss_manager()
    elif page == "🧠 Gerador QA":
        qa_page.show_qa_generator()
    else:
        chat_page.show()

if __name__ == "__main__":
    main()