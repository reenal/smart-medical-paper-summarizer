import streamlit as st
import tempfile
import os
import json
from datetime import datetime
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from src.helper import *

# Initialize session state variables
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_explanation' not in st.session_state:
    st.session_state.best_model_explanation = None

# --- CSS Styling ---
st.markdown("""
    <style>
    .summary-box {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-table {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 PDF Summarizer with Multiple LLMs")
st.subheader("Upload a PDF and compare summaries from various models")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Summarize PDF"):
        st.session_state.processed = False
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            with st.spinner("🔍 Processing PDF..."):
                loader = PyPDFLoader(tmp_path)
                raw_documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
                documents = text_splitter.split_documents(raw_documents)

                # Check if embeddings is defined in helper.py
                try:
                    db = FAISS.from_documents(documents, embeddings)
                    retriever = db.as_retriever()
                    query = "Can you summarise the whole document for me?"
                    input_docs = retriever.invoke(query)

                    results = {}
                    for name, model in llms.items():
                        results[name] = run_summarization_with_chain(model, input_docs, query)

                    st.session_state.results = results

                    if "GPT-4" in results:
                        gpt_summary = results["GPT-4"]
                        evaluation_results = {
                            name: evaluate_summary(summary, gpt_summary)
                            for name, summary in results.items() if name != "GPT-4"
                        }

                        st.session_state.evaluation_results = evaluation_results
                        best_model_explanation = interpret_metrics_with_llm(evaluation_results, llms["LLaMA"])
                        st.session_state.best_model_explanation = best_model_explanation

                        # Compute best model (highest avg score)
                        def average(metrics):
                            rouge_score = metrics["ROUGE"]["rouge1"]
                            bleu_score = metrics["BLEU"]["bleu"]
                            meteor_score = metrics["METEOR"]["meteor"]
                            return (rouge_score + bleu_score + meteor_score) / 3.0

                        best_model = max(evaluation_results.items(), key=lambda item: average(item[1]))[0]
                        st.session_state.best_model = best_model
                        st.session_state.processed = True

                        # Save summaries to JSON
                        os.makedirs("summaries", exist_ok=True)
                        filename = f"summaries/summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(filename, "w") as f:
                            json.dump({
                                "timestamp": datetime.now().isoformat(),
                                "query": query,
                                "summaries": results
                            }, f, indent=4)

                        st.info(f"✅ Summaries saved to: {filename}")
                    else:
                        st.error("GPT-4 model not found in the available models.")
                
                except NameError as e:
                    st.error(f"Configuration error: {str(e)}. Please check your helper.py file.")
                
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {str(e)}")

# --- Results ---
if st.session_state.processed and st.session_state.results:
    if st.session_state.best_model:
        st.success(f"🏆 Best Performing Model: {st.session_state.best_model}")
    if st.session_state.best_model_explanation:
        st.markdown("### 🧠 Best Model Explanation (LLaMA)")
        st.info(st.session_state.best_model_explanation)

    st.markdown("## 📊 Evaluation Metrics (vs GPT-4)")
    if st.session_state.evaluation_results:
        st.dataframe(format_metrics_for_display(st.session_state.evaluation_results).style.format(precision=3), use_container_width=True)

    st.markdown("## 📚 Summaries")
    for name, summary in st.session_state.results.items():
        with st.expander(f"📝 Summary by {name}", expanded=(name == st.session_state.best_model)):
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
