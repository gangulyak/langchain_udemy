import os
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import sys

# Force Chroma to use the newer sqlite from pysqlite3
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

st.title("📚 RAG Agent with Critic")
with st.sidebar:
    st.title("Ask questions about policies")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
if not OPENAI_API_KEY:
    st.info("Please provide OpenAI API key")
    st.stop()

# ---- Setup ----
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

# Upload a PDF or Word doc
uploaded_file = st.file_uploader("Upload a PDF or Word document", type=["pdf", "docx"])

if uploaded_file:
    # Save uploaded file to a temp location for loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load document
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = UnstructuredWordDocumentLoader(tmp_path)

    docs = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings & vectorstore
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = Chroma.from_documents(chunks, embeddings)

    # Retriever with MMR
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.5}
    )

    # QA prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a helpful assistant answering user queries using the provided context. 
            If questions are direct and factual and if the answer cannot be found in the context, say so clearly. 
            Keep answers concise (max 3 paragraphs). If questions are more general - Example - tell me about what 
            the author thinks? or similar then you are allowed to make suggestions keeping the context in mind.
            Context: {context}"""),
            ("human", "{input}")
        ]
    )

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # Critic prompt
    critic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are a critic that evaluates answers for clarity, factual grounding, and completeness.
            - If the answer is correct and clear, respond with: "APPROVED".
            - If the answer is incomplete, vague, or misses key context, provide actionable feedback 
              suggesting improvements. You may infer details if context strongly suggests them.
            - If the answer is hallucinated or not grounded in context, state clearly what is wrong 
              and advise the LLM to recheck the retrieved documents."""),
            ("human", "Evaluate this answer for the question: '{question}'\nAnswer: {answer}")
        ]
    )

    critic_chain = critic_prompt | llm

    # User query
    query = st.text_input("Ask a question about the document")

    if query:
        # Run RAG
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]

        # Critic evaluation
        critic_response = critic_chain.invoke({"question": query, "answer": answer})
        critic_feedback = critic_response.content.strip()

        # Show results
        if "APPROVED" in critic_feedback:
            st.success("✅ Final Answer")
            st.write(answer)
        else:
            st.warning("⚠️ Critic suggested improvements")
            st.write(f"**Original Answer:** {answer}")
            st.write(f"**Critic Feedback:** {critic_feedback}")

            # Optional: retry with critic feedback appended
            improved_response = rag_chain.invoke(
                {"input": f"{query}\nCritic feedback: {critic_feedback}"}
            )
            st.info("🔄 Improved Answer (based on critic)")
            st.write(improved_response["answer"])
