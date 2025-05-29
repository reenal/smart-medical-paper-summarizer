import streamlit as st
import tempfile
import json
import os
import logging
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from evaluate import load as load_metric

# --- Load environment variables ---
load_dotenv()

# --- Initialize session state ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# --- Define helper functions ---
def get_conversational_chain(llm):
    prompt_template = '''You are a highly skilled academic summarizer with extensive experience in distilling complex research papers into clear, concise summaries that highlight key findings and implications. 

Your task is to summarize a research paper effectively. Please provide the following details about the paper you would like summarized:  
- Title of the research paper: __________  
- Authors: __________  
- Abstract: __________  
- Key findings: __________  
- Any specific sections to focus on (e.g., introduction, conclusion): __________  

---

The summary should be structured in a professional format, including a brief introduction of the paper, followed by the main findings and their significance. Aim for a length of approximately 150-300 words, ensuring clarity and coherence throughout.

---

Keep in mind that the summary should be accessible to a general audience, avoiding overly technical jargon, while still accurately representing the content of the research paper. 

---

Be cautious to retain the original context and meaning of the research, ensuring that no critical information is omitted. Avoid personal opinions or interpretations; the summary should strictly reflect the paper's findings.

---

Example format for the summary:

Title: [Title of the research paper]  
Authors: [List of authors]  

Summary:  
[Your concise summary here...]


Document Context:
{context}'''

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | llm | StrOutputParser()
    return chain

def run_summarization_with_chain(llm, docs, query):
    chain = get_conversational_chain(llm)
    return chain.invoke({"context": docs, "question": query})

def evaluate_summary(prediction, reference):
    rouge = load_metric("rouge")
    bleu = load_metric("bleu")
    meteor = load_metric("meteor")

    rouge_result = rouge.compute(predictions=[prediction], references=[reference])
    bleu_result = bleu.compute(predictions=[prediction], references=[reference])
    meteor_result = meteor.compute(predictions=[prediction], references=[reference])

    return {
        "ROUGE": rouge_result,
        "BLEU": bleu_result,
        "METEOR": meteor_result
    }

# --- API Key Setup ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- LLMs Setup ---
llms = {
    "Cohere": ChatCohere(cohere_api_key=COHERE_API_KEY, model="command-a-03-2025", temperature=0),
    "LLaMA": ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY, temperature=0),
    "GPT-4": ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")
}

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-v4.0")

# --- Streamlit UI ---
st.set_page_config(page_title="Multi-LLM PDF Summarizer", layout="centered")
st.title("📄 PDF Summarizer with Multiple LLMs")
st.subheader("Upload your PDF and compare summaries from different LLMs")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    summarize_button = st.button("Summarize PDF")

    if summarize_button:
        # Reset session state for new processing
        st.session_state.processed = False
        st.session_state.results = None
        st.session_state.evaluation_results = None
        st.session_state.best_model = None
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("🔍 Reading and processing your PDF..."):
            loader = PyPDFLoader(tmp_path)
            raw_documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            documents = text_splitter.split_documents(raw_documents)

            db = Chroma.from_documents(documents, embeddings)
            retriever = db.as_retriever()
            user_query = "Can you summarise the whole document for me?"
            input_docs = retriever.invoke(user_query)

            results = {}
            for name, model in llms.items():
                summary = run_summarization_with_chain(model, input_docs, user_query)
                results[name] = summary

            # Store results in session state
            st.session_state.results = results

        st.success("✅ Summaries generated!")

        gpt_summary = st.session_state.results.get("GPT-4")
        reference_summary = gpt_summary  # use GPT-4 summary as reference

        evaluation_results = {}
        for name, summary in st.session_state.results.items():
            if name != "GPT-4":
                evaluation_results[name] = evaluate_summary(summary, reference_summary)

        # Store evaluation results in session state
        st.session_state.evaluation_results = evaluation_results

        if evaluation_results:
            def average_score(metrics):
                rouge = metrics["ROUGE"]["rouge1"]
                bleu = metrics["BLEU"]["bleu"]
                meteor = metrics["METEOR"]["meteor"]
                return (rouge + bleu + meteor) / 3

            best_model = max(evaluation_results.items(), key=lambda item: average_score(item[1]))[0]
            st.session_state.best_model = best_model
            
        st.session_state.processed = True

        # Save results to file
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "summaries": st.session_state.results
        }

        os.makedirs("summaries", exist_ok=True)
        output_file = f"summaries/summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

        st.info(f"Summaries saved to: {output_file}")

    # Display results if they exist in session state
    if st.session_state.processed and st.session_state.results:
        if st.session_state.best_model:
            st.success(f"🏆 **Best Performing Model (compared to GPT-4): {st.session_state.best_model}**")

        st.markdown("### Click to see individual summary metrics")
        if st.button("Show Evaluation Metrics"):
            if st.session_state.evaluation_results:
                for name, metrics in st.session_state.evaluation_results.items():
                    st.subheader(f"🔍 Metrics for {name} (vs GPT-4)")
                    st.write("**ROUGE**:", metrics["ROUGE"])
                    st.write("**BLEU**:", metrics["BLEU"])
                    st.write("**METEOR**:", metrics["METEOR"])

        # Display summaries
        for name, summary in st.session_state.results.items():
            st.subheader(f"🧠 Summary by {name}:")
            st.write(summary)