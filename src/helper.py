import os

from langchain.document_loaders import PyPDFLoader


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "data/Understanding_Climate_Change.pdf"



def load_pdf(path: str):
    """
    Load a PDF document using LangChain's PyPDFLoader.
    """
    pass


def load_pdf(path: str):
    """
    Load a PDF document using LangChain's PyPDFLoader.
    """
    pass


def split_documents(docs):
    """
    Split loaded documents into manageable chunks.
    """
    pass


def create_vectorstore(chunks):
    """
    Create a FAISS vector store or any other supported index.
    """
    pass


def get_retriever(vectorstore):
    """
    Return a retriever for querying relevant document chunks.
    """
    pass


def generate_prompt():
    """
    Create a LangChain PromptTemplate.
    """
    pass


def run_qa_chain(prompt, retriever):
    """
    Run the question-answering chain using LLM + retriever.
    """
    pass
