# 📄 Smart Medical Paper Summarizer

This is a Streamlit app that allows users to upload medical research papers in PDF format and generate a smart, AI-powered summary using **Cohere's large language models** via LangChain and **retrieval-augmented generation (RAG)**.

---

## 🚀 Features

- Upload medical research papers in PDF format
- Generate comprehensive document summaries with one click
- Powered by Cohere's `command-a-03-2025` model
- Uses `CohereEmbeddings` and `Chroma` for accurate document retrieval
- RAG (Retrieval-Augmented Generation) for context-aware summarization
- View source citations and retrieved content
- Specialized for medical and scientific papers

---

## 🧠 Tech Stack

- [Streamlit](https://streamlit.io/) - Web interface
- [LangChain](https://www.langchain.com/) - LLM framework
- [Cohere LLM & Embeddings](https://cohere.com/) - AI model
- [Chroma Vector DB](https://www.trychroma.com/) - Vector storage
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) - PDF parsing

---

## 🛠️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/smart-medical-paper-summarizer.git
cd smart-medical-paper-summarizer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
COHERE_API_KEY=your_cohere_api_key
```

4. **Run the application:**
```bash
streamlit run app.py
```

---

## 📄 Usage Guide

1. Launch the application
2. Upload a medical research paper (PDF format)
3. Click **"Summarize Paper"**
4. View the generated summary with:
   - Key findings
   - Methodology
   - Results
   - Citations
   - Relevant passages

---

## 📁 Project Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Project dependencies
├── .env                # Environment variables (not tracked)
├── utils/              # Helper functions
│   └── pdf_processor.py
└── README.md           # Documentation
```

---

## 🖼️ Screenshots

![App Screenshot](screenshots/main.png)
*Add screenshots of your application here*

---

## 🤖 Default Prompts

The system uses these default prompts for summarization:

```text
"Please provide a comprehensive summary of this medical research paper, including:
1. Main objectives
2. Methodology
3. Key findings
4. Clinical implications"
```

---

## 🎯 Roadmap

- [ ] Custom query support for specific paper sections
- [ ] Batch processing for multiple papers
- [ ] Export summaries in various formats (PDF, DOCX, etc.)
- [ ] Enhanced medical terminology recognition
- [ ] Citation formatting options
- [ ] Advanced search functionality

---

## 🔒 Security & Best Practices

- Store API keys securely using environment variables
- Implement rate limiting for API calls
- Validate PDF files before processing
- Handle user data according to HIPAA guidelines
- Regular security updates for dependencies

---

## 📝 License

[MIT License](LICENSE)

---

## 👥 Contributing

Contributions are welcome! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details.