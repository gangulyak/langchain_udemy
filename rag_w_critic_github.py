# rag_w_critic_github.py

import os
import sys
import tempfile
import uuid
import streamlit as st

# ------------------------------------------------------------------
# SQLite fix for Streamlit Cloud (Chroma requirement)
# ------------------------------------------------------------------
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# ------------------------------------------------------------------
# LangChain imports
# ------------------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from chromadb.config import Settings

# ------------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------------
st.set_page_config(page_title="RAG", layout="wide")
st.title("📄 RAG (Local Embeddings, OpenRouter LLM)")

# ------------------------------------------------------------------
# OpenRouter API Key
# ------------------------------------------------------------------
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in Streamlit secrets.")
    st.stop()

# ------------------------------------------------------------------
# LLM (OpenRouter – OpenAI compatible)
# ------------------------------------------------------------------
llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-7b-instruct",
    temperature=0.2,
)

# ------------------------------------------------------------------
# Upload document
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a document (PDF or DOCX)",
    type=["pdf", "docx"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ---------------- Load document ----------------
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = Docx2txtLoader(tmp_path)

    docs = loader.load()

    # ---------------- Split ----------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # ✅ Empty-chunk guard
    if not chunks:
        st.error("❌ No readable text found in this document.")
        st.stop()

    # ---------------- Local embeddings ----------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------- Vector store ----------------
    # ✅ HARD FIX for Streamlit Cloud Chroma crashes
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"rag_{uuid.uuid4().hex}",
        persist_directory=None,
        client_settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    # ------------------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------------------
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer ONLY using the provided context. "
                "If the answer is not in the context, say 'I don't know'."
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{input}")
        ]
    )

    # ------------------------------------------------------------------
    # RAG CHAIN
    # ------------------------------------------------------------------
    rag_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough()
        }
        | rag_prompt
        | llm
    )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    query = st.text_input("Ask a question about the document")

    if query:
        with st.spinner("Thinking..."):
            answer_msg = rag_chain.invoke(query)
            answer = answer_msg.content

        st.subheader("🧠 Answer")
        st.write(answer)
