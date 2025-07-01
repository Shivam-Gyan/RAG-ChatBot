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


```bash
chat-with-pdf/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── .env # API keys and environment variables
└── README.md # You're reading it :)

```


### ScreenShot of Chat-Bot

![Chat UI Screenshot](https://res.cloudinary.com/dglwzejwk/image/upload/v1751390932/75a09bce-0271-4767-a708-e6c33be01577.png)



