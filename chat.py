# chat.py
# chat.py
import sys

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from callbacks import StreamlitCallbackHandler
from langchain_core.vectorstores import InMemoryVectorStore

import os

load_dotenv()

PERSIST_DIR = "faiss_index"

def load_qa_chain():
    print("Loading persistent vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Load or initialize FAISS persistence
    if os.path.exists(PERSIST_DIR):
        vectorstore = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        raise RuntimeError("No FAISS index found. Run ingest.py first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash",  # or "gemini-pro"
        temperature=0
    )

    # ✅ Persistent conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="query",
        output_key="answer",
        return_messages=True  # ✅ This line solves the ValueError
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"  # ✅ Tell chain to treat 'answer' as the final output
    )

    return qa_chain

if __name__ == "__main__":
    qa = load_qa_chain()
    while True:
        query = input("\n Your question: ")
        if query.lower() in ["quit", "exit"]:
            break
        res = qa.invoke({"query": query})
        print(f"\n Answer:\n{res['result']}")
        print("\nSources:")
        for doc in res['source_documents']:
            print(f"- Page: {doc.metadata.get('page_number')} | {doc.page_content[:100]}...")
