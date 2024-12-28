__import__('pysqlite3')  # Import the pysqlite3 module
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import yaml 

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_community.vectorstores.chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
import pandas as pd
import tempfile
import time
import tabula
import pdfplumber
from langchain.chains import LLMChain

from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
import ast
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from pdfminer.high_level import extract_text
from langchain_core.messages import HumanMessage, AIMessage
import re

import datetime
from query_translation import rag_chain,rag_chain_multi_query,generate_queries,get_unique_union,reciprocal_rank_fusion,generate_queries_decomposition,decomposition_prompt,format_qa_pair,generate_queries_step_back,response_prompt,generate_docs_for_retrieval,prompt
from chunking_strategies import CHUNKING_STRATEGY
from parser import PARSING_PDF
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Streamlit UI setup
st.set_page_config(page_title="File QA Chatbot", page_icon="🤖")
st.title("Welcome to File QA RAG Chatbot")
placeholder = st.empty()

# File uploader
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"],
    accept_multiple_files=True
)


chunking_strategy = st.sidebar.selectbox(
    "Chunking Strategy for RAG:",
    [   
        "RecursiveCharacterTextSplitter",
        "CharacterTextSplitter",
        "titoken",
        "semantic"
    ]
    )

parsing_strategy = st.sidebar.selectbox(
    "Parsing Strategy for RAG:",
    [   
        "pdfium",
        "PyMuPDFLoader",
        "PyPDFLoader",
        "PDFMinerLoader"
    ]
    )


prompting_method = st.sidebar.selectbox(
    "Select the type of prompting for RAG:",
    [
        "Default (Based on User Query)",
        "Multi-Query",
        "RAG Fusion",
        "Decomposition",
        "Step Back",
        "HyDE"
    ]
    )

# Check if any files are uploaded
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
else:
    # Check if documents have already been processed
    if 'docs' not in st.session_state:
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
        
        
        # step 1: parsing the pdf
        docs=PARSING_PDF(parsing_strategy,temp_filepath)
        
     
    
        # Step 2: Split documents into chunks
        
        text_splitter=CHUNKING_STRATEGY(chunking_strategy)
        print(text_splitter)
        
        doc_chunks = text_splitter.split_documents(docs)

        # Step 3: Convert chunks into embeddings
        embeddings = OpenAIEmbeddings()
        vector_db = Chroma.from_documents(doc_chunks, embeddings,persist_directory="./vector_db")

        # Step 4: Store documents and vector DB in session state for future use
        st.session_state.docs = docs
        st.session_state.vector_db = vector_db
        st.session_state.embeddings = embeddings
        st.session_state.text_splitter = text_splitter

        placeholder.info(f"Total documents split into {len(doc_chunks)} chunks.")
        placeholder.info(f"Stored vector DB for future use.")


    # similarity_retriever = st.session_state.vector_db.as_retriever(search_type="similarity",
    #                                             search_kwargs={"k": 5})

    retriever = st.session_state.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})


    chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7)

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=chatgpt
    )
    






    # Input for asking questions
    if "chat_history"  not in st.session_state:
        st.session_state.chat_history=[]
        
    for message in st.session_state.chat_history:
        if isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                # st.markdown(message.content)
                st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="font-size: 16px;">{message.content}</p>
                    </div>
                    """, unsafe_allow_html=True)
        elif isinstance(message,AIMessage):
            with st.chat_message("AI"):
                st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="font-size: 16px;">{message.content}</p>
                    </div>
                    """, unsafe_allow_html=True)


    question = st.chat_input("Message me :")


    if question is not None and question !="":
        st.session_state.chat_history.append(HumanMessage(question))
        
        
        with st.chat_message("Human"):
            st.markdown(question)
            
        

    


    if question:
        placeholder.info("Searching for context and generating the answer...")

        if prompting_method == "Default (Based on User Query)" or prompting_method == None :
            placeholder.info("generating answer based on User Query Prompting method..")

                # # Retrieve relevant context from the retriever
            context = mq_retriever.get_relevant_documents(question)
            
            # # Join relevant context pieces into a single string
            context_text = "\n".join([doc.page_content for doc in context])
            if context_text:
                result =rag_chain.run({"question": question,"context":context_text})
                
                with st.chat_message("AI"):
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="color: #777; font-size: 12px;">{timestamp}</p>
                        <p style="font-size: 16px;">{result}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.session_state.chat_history.append(AIMessage(result))
                st.sidebar.info(f"I retrieved the data from this source: [View Source]")
                
                st.sidebar.markdown(context_text)

            else:
                st.write("Sorry, I couldn't find any relevant context.")


        elif prompting_method=="Multi-Query": 
            placeholder.info("generating answer based on Multi-Query Prompting method..")
            # Retrieve
            # question = "What is task decomposition for LLM agents?"
            retrieval_chain = generate_queries | mq_retriever.map() | get_unique_union
            context = retrieval_chain.invoke({"question": question})
            context_text = "\n".join([doc.page_content for doc in context])
            
            if context_text:
                result = rag_chain_multi_query.run({"question": question})
                result2=rag_chain.invoke({"question":question,"context":context_text})
                # print(result2)
                with st.chat_message("AI"):
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="color: #777; font-size: 12px;">{timestamp}</p>
                        <p style="font-size: 16px;">{result2['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(result2['text']))
                st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                st.sidebar.markdown(f"questions.. ,{result}")
            else:
                st.write("Sorry, I couldn't find any relevant context.")

        elif prompting_method=="RAG Fusion":
            placeholder.info("generating answer based on Multi-Query RAG Fusion Prompting method..")
            retrieval_chain = generate_queries | mq_retriever.map() | reciprocal_rank_fusion
            context_text = retrieval_chain.invoke({"question": question})



            if context_text:
                result = rag_chain_multi_query.run({"question": question})
                result2=rag_chain.invoke({"question":question,"context":context_text})
                with st.chat_message("AI"):
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <p style="color: #777; font-size: 12px;">{timestamp}</p>
                        <p style="font-size: 16px;">{result2['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(result2['text']))
                st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                st.sidebar.markdown(f"questions.. ,{result}")
            else:
                st.write("Sorry, I couldn't find any relevant context.")


        elif prompting_method == "Decomposition":
            questions = generate_queries_decomposition.invoke({"question":question})

            q_a_pairs = ""
            for q in questions:
                rag_chain = (
                {"context": itemgetter("question") | mq_retriever, 
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs")} 
                | decomposition_prompt
                | chatgpt
                | StrOutputParser())

                answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
                q_a_pair = format_qa_pair(q,answer)
                q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

            with st.chat_message("AI"):
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #777; font-size: 12px;">{timestamp}</p>
                    <p style="font-size: 16px;">{answer}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(answer))
                # st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                # st.sidebar.markdown(f"questions.. ,{result}")

        elif prompting_method=="Step Back":

            # question = "What is task decomposition for LLM agents?"
            generate_queries_step_back.invoke({"question": question})

            chain = (
                {
                    # Retrieve context using the normal question
                    "normal_context": RunnableLambda(lambda x: x["question"]) | mq_retriever,
                    # Retrieve context using the step-back question
                    "step_back_context": generate_queries_step_back | mq_retriever,
                    # Pass on the question
                    "question": lambda x: x["question"],
                }
                | response_prompt
                | chatgpt
                | StrOutputParser()
            )

            answer=chain.invoke({"question": question})

            with st.chat_message("AI"):
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #777; font-size: 12px;">{timestamp}</p>
                    <p style="font-size: 16px;">{answer}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(answer))
                # st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                # st.sidebar.markdown(f"questions.. ,{result}")

        elif prompting_method == "HyDE":
            

            # Run
            # question = "What is task decomposition for LLM agents?"
            generate_docs_for_retrieval.invoke({"question":question})


            retrieval_chain = generate_docs_for_retrieval | mq_retriever 
            retireved_docs = retrieval_chain.invoke({"question":question})

            # final_rag_chain = (
            #     prompt
            #     | chatgpt
            #     | StrOutputParser()
            # )

            answer=rag_chain.invoke({"context":retireved_docs,"question":question})

            with st.chat_message("AI"):
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                    <p style="color: #777; font-size: 12px;">{timestamp}</p>
                    <p style="font-size: 16px;">{answer['text']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.chat_history.append(AIMessage(answer['text']))
                # st.sidebar.markdown(f"I retrieved the data from this source: {context_text}")
                # st.sidebar.markdown(f"questions.. ,{result}")
                
        else:
            # Default (Based on User Query)
            pass