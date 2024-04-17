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
import functions as f

load_dotenv(Path(".env"))
embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)
index_name =os.getenv('PINECONE_INDEX_NAME')

filename = f.file_selector()
st.write('You selected `%s`' % filename)
doc = f.read_doc(filename)
documents = f.chunk_data(docs=doc)

index=Pinecone.from_documents(doc,embeddings,index_name=index_name)

llm=OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=1)
chain=load_qa_chain(llm,chain_type="stuff")

title = st.text_input('Question', 'type your question here')
answer = f.retireve_answers(title,chain,index)
st.write(answer)
if st.button("Reset Index"):
    f.delete_index(index)
    st.write("Index Deleted")
