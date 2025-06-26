import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ API key is missing! Please check your .env file.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    Make sure to give complete and easy to understand output
    <context>
    {context}
    </context>
    Question: {input}
    """
)

def create_vector_embedding(directory_path):
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

            if not os.path.exists(directory_path):
                st.error(f"Directory `{directory_path}` does not exist.")
                return
            
            loader = PyPDFDirectoryLoader(directory_path)
            st.session_state.docs = loader.load()
            
            if not st.session_state.docs:
                st.error("No PDF files found in the directory.")
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
            st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs[:50])

            if not st.session_state.final_documents:
                st.error("Failed to split documents.")
                return
            
            doc_texts = [doc.page_content for doc in st.session_state.final_documents]
            embeddings = st.session_state.embeddings.embed_documents(doc_texts)

            if not embeddings or any(len(e) == 0 for e in embeddings):
                st.error("Embeddings were not generated correctly.")
                return

            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Vector database initialized successfully.")

        except Exception as e:
            st.error(f"Error during vector initialization: {e}")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

directory_path = "/tmp/research_papers"

if uploaded_files:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(directory_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")

if st.button("Document Embedding"):
    create_vector_embedding(directory_path)

user_prompt = st.text_input("Enter your query from the research paper")

if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.time()
        response = retrieval_chain.invoke({'input': user_prompt})
        end_time = time.time()

        st.write(f"Response time: {end_time - start_time:.2f} seconds")
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-------------------')
    else:
        st.warning("Vector database is not initialized. Please click 'Document Embedding' first.")
