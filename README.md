# 📄💬 RAG-Powered PDF Chatbot

A powerful **Retrieval-Augmented Generation (RAG)** chatbot that lets you **upload PDFs** and **ask questions conversationally**.  
This project uses **LangChain**, **Hugging Face models**, and **Streamlit** to build an intelligent document assistant capable of extracting relevant information and generating human-like responses.

---

## 🚀 Features

- 📚 **PDF Ingestion & Text Splitting** — Process any PDF into searchable text chunks.  
- 🧠 **Embeddings & Vector Search** — Create FAISS vector stores for efficient semantic retrieval.  
- 🤖 **Hugging Face LLM Integration** — Generate natural, context-aware answers.  
- 📝 **Session Management** — Persistent chat history across multiple sessions.  
- ⚡ **Evaluation Pipeline** — Automatically evaluate chatbot responses using LangSmith or custom evaluators.  
- 🌐 **Streamlit UI** — Clean and interactive user interface for chatting with your PDFs.  
- 🔐 **Environment Variable Management** — Secure handling of API keys using `.env` files.

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)  
- **LLM & Embeddings**: [Hugging Face Hub](https://huggingface.co/)  
- **RAG Framework**: [LangChain](https://www.langchain.com/)  
- **Vector Store**: FAISS  
- **Evaluation**: LangSmith / Hugging Face based scorers  
- **Language**: Python 3.10+

---

## 📂 Project Structure

