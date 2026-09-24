"""
StudyGen AI 🧠 — Local AI Study Assistant
A Streamlit app that lets a student upload a PDF, builds a local FAISS
vector store from it using Google Gemini embeddings, and then answers
questions about the document using Gemini 1.5 Flash with streaming
responses and retrieval-augmented generation (RAG).

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# --------------------------------------------------------------------------- #
# Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="StudyGen AI 🧠",
    page_icon="🧠",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Session state initialization
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF file object using PyPDF2."""
    reader = PdfReader(uploaded_file)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def build_vector_store(raw_text: str, api_key: str) -> FAISS:
    """Split raw text into chunks, embed them with Gemini embeddings,
    and build a local FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)

    if not chunks:
        raise ValueError("No text could be extracted from this PDF. It may be a scanned/image-only document.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )

    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def get_rag_chain(vector_store: FAISS, api_key: str):
    """Build a retrieval-augmented generation chain using LCEL that
    supports streaming output."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are StudyGen AI, a helpful and knowledgeable study assistant.
Use ONLY the following context extracted from the student's uploaded document
to answer the question as clearly and helpfully as possible. If the answer
cannot be found in the context, say that the document does not contain
enough information to answer, and do not make up facts.

Context:
{context}

Question:
{question}

Answer in a clear, well-structured, student-friendly way:"""
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("StudyGen AI 🧠")
    st.caption("Your local AI-powered study assistant")

    st.markdown("---")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="Enter your Google Gemini API key",
        help="Get a free API key at https://aistudio.google.com/app/apikey",
    )

    st.markdown("---")

    uploaded_pdf = st.file_uploader(
        "📄 Upload your study material (PDF)",
        type=["pdf"],
        accept_multiple_files=False,
    )

    process_clicked = st.button("⚙️ Process Document", use_container_width=True)

    if process_clicked:
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key before processing.")
        elif not uploaded_pdf:
            st.warning("⚠️ Please upload a PDF file before processing.")
        else:
            try:
                with st.spinner("Reading PDF and building your knowledge base... 📚"):
                    raw_text = extract_text_from_pdf(uploaded_pdf)

                    if not raw_text.strip():
                        st.error(
                            "❌ No extractable text was found in this PDF. "
                            "It may be a scanned image — try a text-based PDF instead."
                        )
                    else:
                        vector_store = build_vector_store(raw_text, api_key)
                        st.session_state.vector_store = vector_store
                        st.session_state.document_name = uploaded_pdf.name
                        st.session_state.messages = []  # reset chat for the new document
                        st.success(f"✅ '{uploaded_pdf.name}' processed successfully!")
            except Exception as e:
                st.error(f"❌ Something went wrong while processing the document:\n\n{e}")

    st.markdown("---")

    if st.session_state.vector_store is not None:
        st.info(f"📘 Active document: **{st.session_state.document_name}**")
        if st.button("🗑️ Clear Document & Chat", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.document_name = None
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("No document processed yet.")


# --------------------------------------------------------------------------- #
# Main chat interface
# --------------------------------------------------------------------------- #
st.header("💬 Chat with your Study Material")

if st.session_state.document_name:
    st.caption(f"Currently studying: **{st.session_state.document_name}**")
else:
    st.caption("Upload and process a PDF from the sidebar to get started.")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input("Ask a question about your document...")

if user_question:
    # Validate prerequisites before doing anything else
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar first.")
    elif st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and process a PDF document first.")
    else:
        # Display and store the user's message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and stream the assistant's response
        with st.chat_message("assistant"):
            try:
                chain = get_rag_chain(st.session_state.vector_store, api_key)
                response_stream = chain.stream(user_question)
                full_response = st.write_stream(response_stream)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            except Exception as e:
                error_message = f"❌ An error occurred while generating the answer:\n\n{e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
