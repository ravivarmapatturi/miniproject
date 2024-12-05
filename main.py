import os
import yaml 
from langchain_core.documents import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
import streamlit as st
import os
import pandas as pd
import tempfile
import time
# from docx import Document
import tabula
import pdfplumber

from langchain.chains import LLMChain






def extracting_text_data(path):
   extracted_text=[]
   with pdfplumber.open(path) as pdf:
      for i in range(len(pdf.pages)):
         page=pdf.pages[i]
         extracted_text.append(page.extract_text())

   return extracted_text

def extracting_tabular_data(path):
   extracted_tabular=[]
   dfs=tabula.read_pdf(uploaded_files[0],pages="all",stream=True)
   for df in dfs:
      extracted_tabular.append(df.to_csv(index=False))
   return extracted_tabular


with open("/home/ravivarma/Downloads/preplaced/session_5_tasks/miniproject/credentials/api_keys.yaml") as file:
    config=yaml.safe_load(file)

api_keys=config['api_keys']["chatgpt"]
os.environ["OPENAI_API_KEY"]=api_keys


st.set_page_config(page_title="File QA Chatbot", page_icon="🤖")
st.title("Welcome to File QA RAG Chatbot 🤖")


placeholder=st.empty()

uploaded_files = st.sidebar.file_uploader(
  label="Upload PDF files", type=["pdf"],
  accept_multiple_files=True
)
if not uploaded_files:
  st.info("Please upload PDF documents to continue.")

else:
   
   all_data=[]
   
   
   placeholder.info(f"extracting_text_data...........")
   text_data=extracting_text_data(uploaded_files[0])
   all_text = "\n\n".join(text_data)
   st.sidebar.download_button("Download Extracted Text (Word)", all_text, file_name="extracted_text.docx")

   placeholder.info("extractig_tabular_data...........")

   csv_list=extracting_tabular_data(uploaded_files[0])

   for idx,csv in enumerate(csv_list):
      st.sidebar.download_button(f"Download Extracted Table_{idx}",csv,"file.csv","text/csv",key=f'download-csv_{idx}')

   all_data.extend(text_data)
   all_data.extend(csv_list)


   docs=[Document(
    page_content=info,
    metadata={"source": idx})for idx,info in enumerate(all_data)]
   

#    for doc in docs:
#       print(doc)








      
   


#    docs=[]
#    text_data = []
#    table_data = []
    
#    temp_dir = tempfile.TemporaryDirectory()
#    placeholder.info("loading your pdf file..........")
#    for file in uploaded_files:
#     temp_filepath=os.path.join(temp_dir.name,file.name)
#     with open(temp_filepath,"wb") as f:
#        f.write(file.getvalue())
#     pdf_reader=PyMuPDFLoader(temp_filepath)
#     doc=pdf_reader.load()    
#     for page in doc:
#         # print(page,type(page))
#         text_data.append(page.page_content)
        
#         # Extract tables (if any) from the page
#         # print(page.keys())
#         # for table in page.tables:
#         #     table_data.append(pd.DataFrame(table))

#     docs.extend(doc) 


#    all_text = "\n\n".join(text_data)

    # Save text to .docx file
#    docx_file_path = os.path.join(temp_dir.name, "extracted_text.txt")
#    save_text_to_doc(all_text, docx_file_path)

#     # Save tables to either Excel or CSV
#     if file_type == 'excel':
#         excel_file_path = os.path.join(temp_dir.name, "extracted_tables.xlsx")
#         save_tables_to_excel_or_csv(all_tables, excel_file_path, 'excel')
#     elif file_type == 'csv':
#         save_tables_to_excel_or_csv(all_tables, temp_dir.name, 'csv')

    # Provide download links

   

#     if file_type == 'excel':
#         st.download_button("Download Extracted Tables (Excel)", excel_file_path, file_name="extracted_tables.xlsx")
#     elif file_type == 'csv':
#         # Provide CSV download option for each table
#         for i, table in enumerate(all_tables):
#             csv_file_path = os.path.join(temp_dir.name, f"table_{i+1}.csv")
#             table.to_csv(csv_file_path, index=False)
#             st.download_button(f"Download Table {i+1} (CSV)", csv_file_path, file_name=f"table_{i+1}.csv")
   

#    placeholder.info("1.loading completed...")
#    time.sleep(1)
#    placeholder.info(f"{docs[0]}")
#    time.sleep(1)
   placeholder.info("2.Breaking  docs into smaller chunks")

   text_splitter=RecursiveCharacterTextSplitter( chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False)

   doc_chunks=text_splitter.split_documents(docs) 
   

   placeholder.info(f"total no of documets {len(docs)} are converted to {len(doc_chunks)} chunks")

   time.sleep(1)

   placeholder.info("3.converting chunks to embeddings")

   time.sleep(1)

   embeddings=OpenAIEmbeddings()

   vector_db=Chroma.from_documents(doc_chunks,embeddings)

   placeholder.info("4.storing them into chroma vector db")

   time.sleep(1)

   placeholder.info("5.defining the retriever")

   retriever = vector_db.as_retriever()




   chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1,
                      streaming=True)

  # Create a prompt template for QA RAG System
   qa_template = """
                Use only the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know,
                don't try to make up an answer. Keep the answer as concise as possible.

                {context}

                Question: {question}
                """
   qa_prompt = ChatPromptTemplate.from_template(qa_template)

   rag_chain = LLMChain(prompt=qa_prompt, llm=chatgpt)


# Step 8: Set up Streamlit user interface
   placeholder = st.empty()



   question = st.text_input("Ask a question:")

   if question:
    # Call the chain to process the question
    placeholder.info("Searching for context and generating the answer...")
    
    # Retrieve relevant context from the retriever
    context = retriever.get_relevant_documents(question)
    
    # Join all relevant context pieces into a single string
    context_text = "\n".join([doc.page_content for doc in context])
    
    # Ensure context is available, then run the LLMChain
    if context_text:
        result = rag_chain.run({"question": question, "context": context_text})
        # Display the answer to the user
        st.write("Answer:", result)
    else:
        st.write("Sorry, I couldn't find any relevant context.")
   
   






