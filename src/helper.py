from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.prompt import *
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
from src.logger import *
import requests
import io
from PIL import Image
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import asyncio
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings

from langchain_cohere import (
    ChatCohere,
    CohereEmbeddings,
    CohereRerank,
    CohereRagRetriever,
)


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# GoogleGenerativeAIEmbeddings.api_key = api_key
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
HuggingFace_API_KEY = os.getenv('HuggingFace_API_KEY')

def save_uploaded_file(uploaded_file):
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
    
    return tmp_file.name  # return path


def read_single_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def read_documents_for_multiple_pdfs():
    logging.info('read data directory all pdfs')
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()
    return docs


def get_text_chunks_for_multiple_pdfs(text):
    logging.info('chunking process started for the extracted text')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    logging.info('chucking process done successfully')
    split_text = text_splitter.split_documents(text)
    st.write(len(split_text))
    return split_text


def get_vector_store_for_multiple_pdfs(text_chunks):
    logging.info("start storing embeeding into vector db")
    embeddings = CohereEmbeddings(model="embed-v4.0")
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    # vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logging.info("storing embeeding into vector db done successfully in local folder")
    return vector_store


def create_embedding_for_multiple_pdfs():
    logging.info('embedding process started')
    text = read_documents_for_multiple_pdfs()
    text_chunks = get_text_chunks_for_multiple_pdfs(text)
    db = get_vector_store_for_multiple_pdfs(text_chunks=text_chunks)
    logging.info('embedding process done successfully')
    return db


def get_file_text():
    logging.info('read the pdf dat from the data folder')
    text = ""
    pdf_reader = PdfReader('data/Bhagavad-Gita As It Is.pdf')
    for page in pdf_reader.pages:
        text += page.extract_text()
    logging.info('extract text from pdf successfully')
    return text

# Function to split text into chunks
def get_text_chunks(text):
    logging.info('chunking process started for the extracted text')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    logging.info('chucking process done successfully')
    return text_splitter.split_text(text)


def create_embedding():
    logging.info('embedding process started')
    text = get_file_text()
    text_chunks = get_text_chunks(text)
    db = get_vector_store(text_chunks=text_chunks)
    logging.info('embedding process done successfully')
    return db

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
embeddings = CohereEmbeddings(model="embed-v4.0")
# Function to load or create a vector store from the text chunks
def get_vector_store(text_chunks):
    logging.info("start storing embeeding into vector db")
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    logging.info("storing embeeding into vector db done successfully in local folder")
    return vector_store


def get_conversational_chain():
    logging.info("import the base prompt from promt template")
    prompt_template = base_prompt
    
    # Define the language model
    model = ChatCohere(
    cohere_api_key="COHERE_API_KEY",
    model="command-a-03-2025",
    temperature=0,
)
    # model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    logging.info('import successfully the  llm model')

    # Create a prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    logging.info('prompt load successfully')
    
    # Create the conversational chain
    chain = prompt | model | StrOutputParser()
    logging.info('conversational chain load successfully')
    return chain


def user_input(user_question):
    print(user_question)
    logging.info('user input question received')
    # Load the vector store
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    logging.info('load vector database for successfully')
    # Perform a similarity search and retrieve relevant documents as context
    docs = new_db.similarity_search(user_question)
    logging.info('similarity search done successfully')
    # Get the conversational chain
    chain = get_conversational_chain()
    logging.info('conversational chain initialize')
    # Run the chain with the retrieved context and the user's question
    response = chain.invoke({"context": docs, "question": user_question})
    logging.info('llm model provide the response successfully')
    print(response)
    
    
    return response
