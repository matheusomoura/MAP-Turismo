import streamlit as st
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from utils import get_by_session_id
from faiss_db import search_documents
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_CHAT = os.getenv("MODEL_CHAT")


def clear_session_id():
    if "session_id_chat" in st.session_state:
        get_by_session_id(st.session_state.session_id_chat).clear()
        st.session_state.session_id_chat = None


def load_llm():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é um guia turístico simpático, detalhista e especializado em Brasília. Use o conteúdo abaixo para responder perguntas dos visitantes com clareza e entusiasmo:

{context}

Se o contexto não for relevante para a pergunta, diga que não há informações disponíveis nos documentos. Sempre cite a fonte usando [NOME_DO_ARQUIVO] ao final da frase relevante."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    return prompt | ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0.5,
        model=MODEL_CHAT,
        streaming=True
    )


def show():
    st.markdown("# 🗺️ Guia Interativo de Turismo em Brasília")
    st.caption("Descubra os encantos da capital com IA e aproveite roteiros turísticos criados a partir de fontes confiáveis.")

    with st.sidebar:
        st.header("Opções do Chat")
        st.button("🧹 Limpar Conversa", on_click=clear_session_id)

    if not st.session_state.get("session_id_chat"):
        st.session_state.session_id_chat = str(uuid4())

    chain = load_llm()
    history = get_by_session_id(st.session_state.session_id_chat)

    for msg in history.messages:
        st.chat_message(msg.type).markdown(msg.content)

    if prompt := st.chat_input("Pergunte sobre turismo, roteiros, pontos turísticos..."):
        human_message = HumanMessage(content=prompt)
        history.add_messages([human_message])
        st.chat_message("human").markdown(prompt)

        try:
            docs = search_documents(prompt, k=10)
            context = ""
            for i, (doc, score) in enumerate(docs):
                source = doc.metadata.get('source', 'Fonte desconhecida')
                context += f"**Fonte {i + 1} ({source})**: {doc.page_content}\n\n"
        except Exception as e:
            st.error(f"Erro na busca de contexto: {str(e)}")
            context = "Nenhum contexto encontrado."

        chat_history = history.messages[:-1]

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in chain.stream({
                    "question": prompt,
                    "history": chat_history,
                    "context": context
                }):
                    if content := getattr(chunk, 'content', ''):
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                history.add_messages([AIMessage(content=full_response)])

            except Exception as e:
                st.error(f"Erro na geração da resposta: {str(e)}")
                history.add_messages([AIMessage(content="Desculpe, ocorreu um erro interno.")])
