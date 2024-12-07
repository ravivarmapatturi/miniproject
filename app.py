__import__('pysqlite3')  # Import the pysqlite3 module
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import yaml 
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import pandas as pd
import tempfile
import time
import tabula
import pdfplumber
from langchain.chains import LLMChain

# Function to extract text from PDF using pdfplumber
def extracting_text_data(path):
    extracted_text = []
    with pdfplumber.open(path) as pdf:
        for i in range(len(pdf.pages)):
            page = pdf.pages[i]
            extracted_text.append(page.extract_text())
    return extracted_text

# Function to extract tables from PDF using tabula
def extracting_tabular_data(path):
    extracted_tabular = []
    dfs = tabula.read_pdf(path)
    for df in dfs:
        extracted_tabular.append(df.to_csv(index=False))
    return extracted_tabular

# Load API keys
with open("./credentials/api_keys.yaml") as file:
    config = yaml.safe_load(file)
api_keys = config['api_keys']["chatgpt"]
os.environ["OPENAI_API_KEY"] = api_keys

# Streamlit UI setup
st.set_page_config(page_title="File QA Chatbot", page_icon="🤖")
st.title("Welcome to File QA RAG Chatbot 🤖")
placeholder = st.empty()

# File uploader
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"],
    accept_multiple_files=True
)

# Check if any files are uploaded
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
else:
    # Check if documents have already been processed
    if 'docs' not in st.session_state:
        # Step 1: Extract text and tabular data only once
        all_data = []

        placeholder.info(f"Extracting text data...")
        text_data = extracting_text_data(uploaded_files[0])
        all_text = "\n\n".join(text_data)
        st.sidebar.download_button("Download Extracted Text (Word)", all_text, file_name="extracted_text.docx")

        placeholder.info(f"Extracting tabular data...")
        csv_list = extracting_tabular_data(uploaded_files[0])
        for idx, csv in enumerate(csv_list):
            st.sidebar.download_button(f"Download Extracted Table_{idx}", csv, "file.csv", "text/csv", key=f'download-csv_{idx}')

        all_data.extend(text_data)
        all_data.extend(csv_list)

        # Convert text and tables into LangChain Documents
        docs = [Document(page_content=info, metadata={"source": idx}) for idx, info in enumerate(all_data)]

        # Step 2: Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=20, length_function=len, is_separator_regex=False)
        doc_chunks = text_splitter.split_documents(docs)

        # Step 3: Convert chunks into embeddings
        embeddings = OpenAIEmbeddings()
        vector_db = Chroma.from_documents(doc_chunks, embeddings)

        # Step 4: Store documents and vector DB in session state for future use
        st.session_state.docs = docs
        st.session_state.vector_db = vector_db
        st.session_state.embeddings = embeddings
        st.session_state.text_splitter = text_splitter

        placeholder.info(f"Total documents split into {len(doc_chunks)} chunks.")
        placeholder.info(f"Stored vector DB for future use.")


    similarity_retriever = st.session_state.vector_db.as_retriever(search_type="similarity",
                                                search_kwargs={"k": 10})
    chatgpt = ChatOpenAI(model_name='gpt-4', temperature=0.7)

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=similarity_retriever, llm=chatgpt
    )
    

    # qa_template = """
    # Use only the following pieces of context to answer the question at the end.
    # If you don't know the answer, just say that you don't know,
    # don't try to make up an answer. Keep the answer as concise as possible.

    # {context}

    # Question: {question}
    # """

    qa_template = """
    your an document expert generator . your task is to provide detail document or answer . based on the user question take info of context 
    answer the question    

    {context}

    Question: {question}
    """
    qa_prompt = ChatPromptTemplate.from_template(qa_template)
    rag_chain = LLMChain(prompt=qa_prompt, llm=chatgpt)

    # Input for asking questions
    question = st.text_input("Ask a question:")

    if question:
        placeholder.info("Searching for context and generating the answer...")
        
        # Retrieve relevant context from the retriever
        context = mq_retriever.get_relevant_documents(question)
        
        # Join relevant context pieces into a single string
        context_text = "\n".join([doc.page_content for doc in context])
        
        # Run the QA chain with the question and context
        if context_text:
            result = rag_chain.run({"question": question, "context": context_text})
            st.write("Answer:", result)

            st.sidebar.write("I retreived the data from this source...",context)
        else:
            st.write("Sorry, I couldn't find any relevant context.")

