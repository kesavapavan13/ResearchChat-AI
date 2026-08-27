import streamlit as st
import os
import time
import faiss
import numpy as np

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from langchain_docling.loader import DoclingLoader


load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ API key is missing! Please check your .env file.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key


llm = ChatGroq(
    api_key=groq_api_key,
    model_name="openai/gpt-oss-20b"
)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.

    Please provide the most accurate response based on the question.
    Make sure to give a complete and easy-to-understand answer.

    <context>
    {context}
    </context>

    Question: {input}
    """
)


def create_vector_embedding(directory_path):

    try:

        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en"
        )


        if not os.path.exists(directory_path):
            st.error(
                f"Directory `{directory_path}` does not exist."
            )
            return


        pdf_files = [
            os.path.join(directory_path, file)
            for file in os.listdir(directory_path)
            if file.lower().endswith(".pdf")
        ]


        if not pdf_files:
            st.error("No PDF files found.")
            return


        loader = DoclingLoader(
            file_path=pdf_files
        )


        st.session_state.docs = loader.load()


        if not st.session_state.docs:
            st.error("No documents were loaded.")
            return


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )


        st.session_state.final_documents = (
            text_splitter.split_documents(
                st.session_state.docs[:50]
            )
        )


        if not st.session_state.final_documents:
            st.error("Failed to split documents.")
            return


        doc_texts = [
            doc.page_content
            for doc in st.session_state.final_documents
        ]


        document_embeddings = (
            st.session_state.embeddings.embed_documents(
                doc_texts
            )
        )


        if not document_embeddings:
            st.error("Embeddings were not generated.")
            return


        document_embeddings = np.array(
            document_embeddings,
            dtype="float32"
        )


        dimension = document_embeddings.shape[1]


        index = faiss.IndexFlatL2(dimension)


        index.add(document_embeddings)


        st.session_state.vectors = index

        st.session_state.vector_documents = (
            st.session_state.final_documents
        )


        st.success(
            f"Vector database initialized successfully. "
            f"{index.ntotal} vectors indexed."
        )


    except Exception as e:

        st.error(
            f"Error during vector initialization: {e}"
        )


def retrieve_documents(query, k=4):

    query_embedding = (
        st.session_state.embeddings.embed_query(
            query
        )
    )


    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )


    distances, indices = (
        st.session_state.vectors.search(
            query_embedding,
            k
        )
    )


    documents = []


    for index in indices[0]:

        if index == -1:
            continue

        documents.append(
            st.session_state.vector_documents[index]
        )


    return documents


uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


directory_path = os.path.join(
    os.getcwd(),
    "research_papers"
)


if uploaded_files:

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


    for uploaded_file in uploaded_files:

        file_path = os.path.join(
            directory_path,
            uploaded_file.name
        )


        with open(file_path, "wb") as f:
            f.write(
                uploaded_file.getbuffer()
            )


    st.success(
        f"Uploaded {len(uploaded_files)} file(s) successfully!"
    )


if st.button("Document Embedding"):

    create_vector_embedding(
        directory_path
    )


user_prompt = st.text_input(
    "Enter your query from the research paper"
)


if user_prompt:

    if "vectors" in st.session_state:

        start_time = time.time()


        retrieved_documents = retrieve_documents(
            user_prompt,
            k=4
        )


        context = "\n\n".join(
            doc.page_content
            for doc in retrieved_documents
        )


        response = llm.invoke(
            prompt.format(
                context=context,
                input=user_prompt
            )
        )


        end_time = time.time()


        st.write(
            f"Response time: "
            f"{end_time - start_time:.2f} seconds"
        )


        st.write(
            response.content
        )


        with st.expander(
            "Document similarity search"
        ):

            for i, doc in enumerate(
                retrieved_documents
            ):

                st.write(
                    f"### Document {i + 1}"
                )

                st.write(
                    doc.page_content
                )

                st.write(
                    "-------------------"
                )


    else:

        st.warning(
            "Vector database is not initialized. "
            "Please click 'Document Embedding' first."
        )