import os

from langchain.document_loaders import PyPDFLoader


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "data/your_pdf.pdf"
PDF_FILES_PATH = "data/"

def load_pdf_files(path: str):
    """
    Load a PDF document using LangChain's PyPDFLoader.
    """
    # Get list of pdf files in the directory
    pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError("No PDF files found in the specified directory.")

    # Load all pdf files using PyPDFLoader
    pdf_docs = [PyPDFLoader(os.path.join(path, pdf_file)).load_and_split() for pdf_file in pdf_files]

    return pdf_docs

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
