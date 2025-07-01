import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import tempfile

# Load environment variables (optional)
load_dotenv()

# Set HuggingFace & Groq API keys
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

llm = ChatGroq(
    model="Gemma2-9b-It",  # Use a valid Groq model name
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# Streamlit layout
st.set_page_config(layout="wide")
st.sidebar.title("📄 Upload PDF")
st.sidebar.write("Upload one or more PDF files to chat with their content.")
st.title("🤖 Chat with your PDF")

session_id = st.text_input("Session ID", value="default_session")

if not session_id:
    st.warning("Please enter a session ID to start chatting.")
    st.stop()

# File uploader
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Load and split documents
if uploaded_files:
    documents = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            docs = loader.load()
            documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    split_docs = splitter.split_documents(documents)

    vector_faiss = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    retriever = vector_faiss.as_retriever()
else:
    st.warning("Upload at least one PDF to proceed.")
    st.stop()

# Prompts
contextual_q_system_prompt = (
    "given chat history and latest user question, "
    "retrieve relevant context from the knowledge base to answer the question.\n"
    "If the context does not contain relevant information, you should respond with 'I don't know'.\n"
    "Make the answer as short as possible, maximum 3 sentences.\n"
)

contextual_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextual_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

system_prompt =system_prompt = '''

You are a helpful AI assistant that answers questions based on the provided context.
Your responses should be concise and directly related to the context given.
If the context does not contain relevant information, you should respond with "I don't know".
make answer as short as possible maximum 3 sentences .

context: {context}


'''

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Chains
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextual_q_prompt,
)

document_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)

rag_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=document_chain
)

# Chat history session store
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

runnable_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Chat Input UI
user_input = st.chat_input("Ask a question about the PDF content...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process RAG
    response = runnable_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    # Show assistant reply
    with st.chat_message("assistant"):
        st.markdown(response["answer"])


print(st.session_state.store)