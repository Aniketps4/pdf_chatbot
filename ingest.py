# ingest.py
import os
import requests
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# PDF download info
PDF_URL = "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/106MDNBFCS1910202343073E3EF57A4916AA5042911CD8D562.PDF"
PDF_PATH = "rbi_notification.pdf"
VECTORSTORE_PATH = "faiss_index"

def download_pdf():
    """Download PDF if it does not exist."""
    if os.path.exists(PDF_PATH):
        print(f"[INFO] PDF already exists: {PDF_PATH}")
        return

    print("[INFO] Downloading RBI PDF...")
    r = requests.get(PDF_URL)
    r.raise_for_status()
    with open(PDF_PATH, "wb") as f:
        f.write(r.content)
    print(f"[INFO] PDF downloaded successfully: {PDF_PATH}")

def build_vectorstore():
    """Load PDF, split, embed using HuggingFace, and save FAISS index."""
    # Ensure PDF exists
    download_pdf()

    # Load PDF
    print("[INFO] Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} pages")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Split PDF into {len(chunks)} chunks")

    # Initialize HuggingFace embeddings (free)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS vector store
    print("[INFO] Building FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"[✅] Vector store saved at: {VECTORSTORE_PATH}")

if __name__ == "__main__":
    build_vectorstore()
