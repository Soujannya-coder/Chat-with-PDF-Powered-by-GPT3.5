import streamlit as st
import os
import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import openai
from dotenv import load_dotenv
from pathlib import Path
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc

def retrieve_query(query,index,k=2):
    matching_result=index.similarity_search(query,k=k)
    return matching_result
def retireve_answers(query,chain,index):
    doc_search=retrieve_query(query,index)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response
def file_selector(folder_path='.'):
    filenames = os.listdir()
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
    
def delete_index(index):
     index.delete(delete_all=True)
