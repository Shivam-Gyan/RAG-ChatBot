# 📄 Chat with your PDF – GenAI-Powered PDF Q&A App

An AI-powered tool that lets you upload one or more PDFs and **interact with their content** using natural language. Built using a **Retrieval-Augmented Generation (RAG)** pipeline powered by **LangChain**, **Groq**, **FAISS**, and **HuggingFace Embeddings**.

---

## 🚀 Project Highlights

- 🔍 Ask context-aware questions about PDFs.
- 🧠 Built on RAG for retrieval + LLM reasoning.
- 💬 Maintains session-wise chat history.
- 🧾 Summarize legal docs, papers, technical manuals, etc.

---

## 🧰 Tech Stack

| Component       | Tool / Library                        |
|-----------------|----------------------------------------|
| Frontend UI     | Streamlit                              |
| LLM Inference   | [Groq API](https://console.groq.com/) using `Gemma2-9b-It` |
| Embeddings      | HuggingFace (`all-MiniLM-L6-v2`)       |
| Vector DB       | FAISS                                  |
| PDF Loader      | PyPDFLoader (LangChain Community)      |
| Chunking        | RecursiveCharacterTextSplitter         |
| RAG Orchestration | LangChain (`retrieval_chain`, `RunnableWithMessageHistory`) |
| Session Memory  | LangChain `ChatMessageHistory`         |

---

## 🧩 Features

- ✅ Upload and parse multiple PDF files
- 🤖 Ask any question in natural language
- 🧠 Memory-aware answers with contextual history
- ⚡ Fast and lightweight interface using Streamlit

---

## 📁 Folder Structure



chat-with-pdf/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── .env # API keys and environment variables
└── README.md # You're reading it :)



---

## ⚙️ Getting Started

### 🔗 Prerequisites

- Python 3.9 or higher
- API keys from:
  - [HuggingFace](https://huggingface.co/settings/tokens)
  - [Groq](https://console.groq.com/)

---

### 🧪 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/pdf-chat-genai.git
cd pdf-chat-genai


python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


pip install -r requirements.txt


HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key

![Chat UI Screenshot](https://example.com/images/chat-ui.png)



