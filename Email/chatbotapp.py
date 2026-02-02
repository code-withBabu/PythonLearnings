import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import tempfile
import os

st.set_page_config(page_title="PDF Chatbot", layout="centered")

st.title("📄🤖 PDF Chatbot (Hugging Face)")
st.write("Upload a PDF and ask questions based on its content.")

# --- Load Hugging Face models ---
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return embeddings, llm

embeddings, llm = load_models()

# --- PDF Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        os.remove(tmp_path)

    st.success("PDF processed successfully!")

    # --- User Question ---
    query = st.text_input("Ask a question about the PDF")

    if query:
        with st.spinner("Searching for answer..."):
            docs = vectorstore.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in docs])

            prompt = f"""
            Answer the question using ONLY the context below.
            If the answer is not present, say "I don't know."

            Context:
            {context}

            Question:
            {query}
            """

            response = llm(prompt)
            answer = response[0]["generated_text"]

        st.subheader("📌 Answer")
        st.write(answer)
