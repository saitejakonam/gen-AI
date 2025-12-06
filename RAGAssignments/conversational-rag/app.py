import streamlit as st
import uuid
import os

from conversational_rag import ConversationalRAG
from chat_history import ChatHistory


# ---------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Conversational RAG Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Conversational RAG Chatbot (Assignment 2)")


# ---------------------------------------------------------------
# Sidebar - Chat Session Management
# ---------------------------------------------------------------
st.sidebar.header("📂 Chat Sessions")

# Track active chat session
active_session = st.session_state.get("active_session", None)

# Start New Chat Section
st.sidebar.subheader("➕ Start New Chat")

new_chat_name = st.sidebar.text_input(
    "Chat name",
    placeholder="Enter a chat name…"
)

if st.sidebar.button("Start Chat"):
    if not new_chat_name.strip():
        st.sidebar.error("Please enter a valid chat name.")
    else:
        st.session_state.active_session = new_chat_name.strip()
        st.session_state.rag = None  # force reload
        ChatHistory(new_chat_name).create_if_not_exists()
        st.rerun()


# ---------------------------------------------------------------
# Saved Chat Sessions List
# ---------------------------------------------------------------
st.sidebar.subheader("💾 Saved Chats")

sessions = ChatHistory.list_sessions()

for sid in sessions:
    col1, col2 = st.sidebar.columns([4, 1])

    # Highlight active session
    label = f"🟩 {sid}" if sid == active_session else sid

    with col1:
        if st.button(label, key=f"load_{sid}"):
            st.session_state.active_session = sid
            st.session_state.rag = None
            st.rerun()

    with col2:
        if st.button("❌", key=f"delete_{sid}", help="Delete this chat"):
            ChatHistory.delete_session(sid)
            if active_session == sid:
                st.session_state.active_session = None
            st.rerun()



# ---------------------------------------------------------------
# If NO session is selected → Stop here & show message
# ---------------------------------------------------------------
if not active_session:
    st.info("👈 Start a new chat from the sidebar or select a saved session.")
    st.stop()


# ---------------------------------------------------------------
# Right Side - RAG Model Configuration
# ---------------------------------------------------------------
st.sidebar.header("⚙️ Model Configuration")

vector_store_type = st.sidebar.selectbox(
    "Vector Store",
    ["faiss", "chromadb"]
)

embedding_model = st.sidebar.selectbox(
    "Embedding Model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L3-v2"
    ]
)

if st.sidebar.button("♻️ Reset Chat"):
    ChatHistory(active_session).clear()
    st.session_state.rag = None
    st.rerun()



# ---------------------------------------------------------------
# Initialize RAG Only After Chat Session is Selected
# ---------------------------------------------------------------
if (
    "rag" not in st.session_state
    or st.session_state.get("store") != vector_store_type
    or st.session_state.get("embed") != embedding_model
):
    st.session_state.rag = ConversationalRAG(
        session_id=active_session,
        vector_store_type=vector_store_type,
        embedding_model=embedding_model
    )
    st.session_state.store = vector_store_type
    st.session_state.embed = embedding_model

rag = st.session_state.rag
history = ChatHistory(active_session)


# ---------------------------------------------------------------
# Chat Display
# ---------------------------------------------------------------
st.subheader(f"💬 Chat: **{active_session}**")

messages = history.get_history()
for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ---------------------------------------------------------------
# Chat Input & Streaming Response
# ---------------------------------------------------------------
user_input = st.chat_input("Type your message…")

if user_input:
    history.add_message("user", user_input)

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        final_text = ""

        for token in rag.stream_answer(user_input):
            final_text += token
            placeholder.markdown(final_text)

        history.add_message("assistant", final_text)

    st.rerun()


# ---------------------------------------------------------------
# Retrieved Document Viewer
# ---------------------------------------------------------------
with st.expander("📄 Retrieved Documents (Last Query)"):

    user_messages = [m for m in messages if m["role"] == "user"]

    if not user_messages:
        st.write("No queries yet.")
    else:
        last_query = user_messages[-1]["content"]
        docs = rag.retrieve_docs(last_query)

        for idx, doc in enumerate(docs):
            st.markdown(f"### Chunk {idx}")
            st.write(doc.page_content)
            st.caption(doc.metadata)
