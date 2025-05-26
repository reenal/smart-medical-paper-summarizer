import streamlit as st
import tempfile
from langchain_cohere import (
    ChatCohere,
    CohereEmbeddings,
    CohereRagRetriever,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# --- API Key Setup ---
COHERE_API_KEY = "0HSdZxcNd2xy3ZccJNYt8t2kDmHjQmbvMJNypRpZ"

# --- LLM and Embeddings ---
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-a-03-2025",
    temperature=0,
)

embeddings = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-v4.0"
)

# --- Streamlit UI ---
st.set_page_config(page_title="Cohere PDF Summarizer", layout="centered")
st.title("📄 PDF Summarizer using Cohere RAG")
st.subheader("Upload your PDF and click 'Summarize PDF' to generate a summary.")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    summarize_button = st.button("Summarize PDF")

    if summarize_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("🔍 Reading and processing your PDF..."):
            # Load and split the PDF
            loader = PyPDFLoader(tmp_path)
            raw_documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            documents = text_splitter.split_documents(raw_documents)

            # Create vector store and retriever
            db = Chroma.from_documents(documents, embeddings)
            retriever = db.as_retriever()

            # RAG summarization
            user_query = "Can you summarise the whole document for me?"
            input_docs = retriever.invoke(user_query)

            rag = CohereRagRetriever(llm=llm)
            docs = rag.invoke(user_query, documents=input_docs)

        st.success("✅ Summary generated!")

        # Display summary
        st.subheader("🧠 Summary:")
        st.write(docs[-1].page_content)

        # Show citations if available
        citations = docs[-1].metadata.get("citations", None)
        if citations:
            st.subheader("🔗 Citations:")
            st.write(citations)

        # Show intermediate documents (optional)
        with st.expander("📚 Retrieved Passages"):
            for doc in docs[:-1]:
                st.markdown(f"**Source**: {doc.metadata}")
                st.write(doc.page_content)
                st.markdown("---")
