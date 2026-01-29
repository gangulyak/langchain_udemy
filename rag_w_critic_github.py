# rag_w_critic_github.py

import os
import tempfile
import streamlit as st

# --- SQLite fix for Streamlit Cloud ---
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# --- LangChain imports ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="RAG with Critic", layout="wide")
st.title("📄 RAG + Critic (Local Embeddings, OpenRouter LLM)")

# ------------------------------------------------------------------
# API KEY (OpenRouter)
# ------------------------------------------------------------------
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", None)

if not OPENROUTER_API_KEY:
    st.warning("⚠️ OPENROUTER_API_KEY not found in Streamlit secrets.")
    st.stop()

# ------------------------------------------------------------------
# LLM (OpenRouter)
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

    # --- Load document ---
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = Docx2txtLoader(tmp_path)

    docs = loader.load()

    # --- Split ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    # --- Local embeddings ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Vector store ---
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    # ------------------------------------------------------------------
    # PROMPTS
    # ------------------------------------------------------------------
    rag_prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{input}
"""
    )

    critic_prompt = ChatPromptTemplate.from_template(
        """
You are a critical reviewer.
Check if the answer below is fully supported by the context.
If not, explain what is missing or incorrect.

Context:
{context}

Answer:
{answer}
"""
    )

    # ------------------------------------------------------------------
    # CHAINS
    # ------------------------------------------------------------------
    rag_chain = (
        {"context": retriever, "input": RunnablePassthrough()}
        | rag_prompt
        | llm
    )

    critic_chain = (
        {
            "context": retriever,
            "answer": RunnablePassthrough()
        }
        | critic_prompt
        | llm
    )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    query = st.text_input("Ask a question about the document")

    if query:
        with st.spinner("Thinking..."):
            
            result = rag_chain.invoke({"input": query})
            answer = result["answer"]
            critique = critic_chain.invoke(answer).content

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 Answer")
            st.write(answer)

        with col2:
            st.subheader("🔍 Critic Review")
            st.write(critique)
