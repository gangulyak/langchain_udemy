import os
import sys
import tempfile
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ------------------------------------------------------------------
# OPENAI API KEY (TEMPORARY – replace with your real key)
# ------------------------------------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ------------------------------------------------------------------
# Force Chroma to use newer sqlite (Streamlit Cloud fix)
# ------------------------------------------------------------------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📚 RAG Agent with Critic")

with st.sidebar:
    st.title("Ask questions about policies")
    st.success("Local embeddings enabled")
    st.info("OpenAI used only for answers & critique")

# ------------------------------------------------------------------
# LLM (OpenAI only for generation)
# ------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY
)

# ------------------------------------------------------------------
# File Upload
# ------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF or Word document",
    type=["pdf", "docx"]
)

if uploaded_file:
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{uploaded_file.name.split('.')[-1]}"
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # ------------------------------------------------------------------
    # Document Loader (SAFE – no unstructured)
    # ------------------------------------------------------------------
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(tmp_path)
    else:
        st.error("Unsupported file type")
        st.stop()

    docs = loader.load()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    # ------------------------------------------------------------------
    # Local Embeddings (cached)
    # ------------------------------------------------------------------
    @st.cache_resource
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    # ------------------------------------------------------------------
    # Retriever (MMR)
    # ------------------------------------------------------------------
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.5}
    )

    # ------------------------------------------------------------------
    # QA Prompt
    # ------------------------------------------------------------------
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant answering user queries using the provided context.
If the answer cannot be found in the context, say so clearly.
Keep answers concise (max 3 paragraphs).

Context:
{context}
"""
            ),
            ("human", "{input}")
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # ------------------------------------------------------------------
    # Critic Prompt
    # ------------------------------------------------------------------
    critic_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a critic that evaluates answers for clarity, factual grounding, and completeness.
- If the answer is correct and clear, respond with: "APPROVED".
- If incomplete or vague, provide actionable feedback.
- If hallucinated, explain what is wrong and advise rechecking the context."""
            ),
            (
                "human",
                "Evaluate this answer for the question: '{question}'\nAnswer: {answer}"
            )
        ]
    )

    critic_chain = critic_prompt | llm

    # ------------------------------------------------------------------
    # User Query
    # ------------------------------------------------------------------
    query = st.text_input("Ask a question about the document")

    if query:
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]

        critic_response = critic_chain.invoke(
            {"question": query, "answer": answer}
        )
        critic_feedback = critic_response.content.strip()

        if "APPROVED" in critic_feedback:
            st.success("✅ Final Answer")
            st.write(answer)
        else:
            st.warning("⚠️ Critic suggested improvements")
            st.write(f"**Original Answer:** {answer}")
            st.write(f"**Critic Feedback:** {critic_feedback}")

            improved_response = rag_chain.invoke(
                {"input": f"{query}\nCritic feedback: {critic_feedback}"}
            )
            st.info("🔄 Improved Answer (based on critic)")
            st.write(improved_response["answer"])
