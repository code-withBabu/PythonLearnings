import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
from langchain_classic.chains.question_answering import load_qa_chain

# --- UI Setup ---
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.header("Chat with your PDF 📄")

api_key = st.sidebar.text_input("Enter Your Google API Key", type="password")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# --- Main App Logic ---
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

if uploaded_files and api_key:
    if st.button("Process PDF"):
        with st.spinner("Analyzing document..."):
            raw_text = get_pdf_text(uploaded_files)
            print(raw_text)
            st.session_state.vector_store = get_vector_store(raw_text)
            st.success("Done!")

user_question = st.text_input("Ask a question about your PDF:")

if user_question and "vector_store" in st.session_state:
    docs = st.session_state.vector_store.similarity_search(user_question)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    response = chain.run(input_documents=docs, question=user_question)
    st.write("### Answer:", response)