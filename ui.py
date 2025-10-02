import streamlit as st
import json
import os
from datetime import datetime
from chat import load_qa_chain

# --------------------------
# Constants
# --------------------------
SESSIONS_FILE = "chat_sessions.json"

# --------------------------
# Utility functions
# --------------------------
def save_sessions():
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chat_sessions, f, ensure_ascii=False, indent=2)

def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def format_session_label(session):
    if not session:
        return "Empty Session"
    timestamp = session[0].get("timestamp", "")
    if timestamp:
        ts_str = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
    else:
        ts_str = "Unknown time"
    # Take first 5 words of first message as preview
    first_msg = session[0]["content"].split()
    preview = " ".join(first_msg[:5]) + ("..." if len(first_msg) > 5 else "")
    return f"{ts_str} | {preview}"

# --------------------------
# Streamlit page setup
# --------------------------
st.set_page_config(page_title="RAG PDF Chatbot", page_icon="🤖", layout="wide")
st.title("SmartDoc Chat – intelligent document Q&A")
st.caption("Ask questions from your PDF knowledge base")

# --------------------------
# Initialize session state
# --------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_qa_chain()

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_sessions()

if "selected_session" not in st.session_state:
    st.session_state.selected_session = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------
# Sidebar: session list
# --------------------------
st.sidebar.title("Chat History")

if st.sidebar.button("➕ New Chat"):
    st.session_state.messages = []
    st.session_state.selected_session = None

for idx, session in enumerate(st.session_state.chat_sessions):
    label = format_session_label(session)
    if st.sidebar.button(label):
        st.session_state.selected_session = idx
        st.session_state.messages = st.session_state.chat_sessions[idx].copy()

# --------------------------
# Display chat messages
# --------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------
# User input
# --------------------------
prompt = st.chat_input("Ask me something...")

if prompt:
    # Add timestamp to the message
    timestamp = datetime.now().isoformat()
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Streaming assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Stream response from QA chain
        response = st.session_state.qa_chain.stream({"query": prompt})
        for chunk in response:
            if "answer" in chunk:
                full_response += chunk["answer"]
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant message with timestamp
    timestamp = datetime.now().isoformat()
    st.session_state.messages.append({"role": "assistant", "content": full_response, "timestamp": timestamp})

    # Update or save session
    if st.session_state.selected_session is not None:
        st.session_state.chat_sessions[st.session_state.selected_session] = st.session_state.messages.copy()
    else:
        st.session_state.chat_sessions.append(st.session_state.messages.copy())
        st.session_state.selected_session = len(st.session_state.chat_sessions) - 1

    save_sessions()
