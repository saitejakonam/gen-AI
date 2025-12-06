"""
Conversational RAG engine (Assignment 2)
---------------------------------------

✔ Uses FAISS / Chroma (Persistent)
✔ Uses HuggingFace embeddings
✔ Uses DIAL for LLM responses
✔ Provides token-by-token streaming
✔ Chat history handled ONLY by Streamlit (NOT here!)
"""

import os
from typing import List, Dict

from utils.dial_client import DIALClient
from chat_history import ChatHistory
from message_trimming import MessageTrimmer
from vector_store import PersistentVectorStore

from langchain.docstore.document import Document


class ConversationalRAG:

    def __init__(
        self,
        session_id: str,
        vector_store_type: str = "faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        content_dir: str = "data/extracted_content",
    ):
        self.session_id = session_id
        self.vector_store_type = vector_store_type
        self.embedding_model = embedding_model
        self.content_dir = content_dir

        # Chat history + trimming
        self.history = ChatHistory(session_id)
        self.trimmer = MessageTrimmer(
            max_messages=10,
            max_tokens=2000,
            summarize_threshold=7,
        )

        # LLM Client
        self.llm = DIALClient()

        # Persistent vector store loader
        self.vector_store = PersistentVectorStore(
            vector_store_type=self.vector_store_type,
            embedding_model=self.embedding_model,
            content_dir=self.content_dir,
        ).load_or_create()


    # ---------------------------------------------------------
    # Retrieve documents
    # ---------------------------------------------------------
    def retrieve_docs(self, query: str, k: int = 3) -> List[Document]:
        return self.vector_store.similarity_search(query, k=k)


    # ---------------------------------------------------------
    # Build prompt for LLM
    # ---------------------------------------------------------
    def build_prompt(self, query: str, retrieved_docs: List[Document], trimmed_history: List[Dict]):
        context = "\n\n".join(
            [f"[Chunk {i}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)]
        )

        history_text = "\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in trimmed_history]
        )

        return f"""
You are a helpful conversational assistant using Retrieval Augmented Generation.

Conversation so far:
{history_text}

Relevant retrieved context:
{context}

User question:
{query}

Respond clearly. Use context when relevant.
"""


    # ---------------------------------------------------------
    # Non-streaming answer
    # ---------------------------------------------------------
    def ask(self, query: str) -> str:
        full_history = self.history.get_history()
        trimmed = self.trimmer.trim(full_history)

        docs = self.retrieve_docs(query)
        prompt = self.build_prompt(query, docs, trimmed)

        response = self.llm.get_completion([
            {"role": "system", "content": "You are a helpful RAG assistant."},
            {"role": "user", "content": prompt},
        ])

        return response


    # ---------------------------------------------------------
    # STREAMING answer generator
    # ---------------------------------------------------------
    def stream_answer(self, query: str):
        """
        Yields tokens one-by-one for real-time UI streaming.
        """
        # Prepare history and context
        full_history = self.history.get_history()
        trimmed = self.trimmer.trim(full_history)

        docs = self.retrieve_docs(query)
        prompt = self.build_prompt(query, docs, trimmed)

        # NEW — uses stream()
        for token in self.llm.stream(prompt):
            yield token

# ---------------------------------------------------------
# CLI TEST MODE (works without Streamlit)
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Conversational RAG via CLI")
    parser.add_argument("--session-id", required=True, help="Chat session ID")
    parser.add_argument("--question", type=str, default=None,
                        help="Ask one question (non-streaming mode)")
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming mode instead of full answer")

    args = parser.parse_args()

    rag = ConversationalRAG(
        session_id=args.session_id,
        vector_store_type="faiss",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )

    print("\n🔧 Conversational RAG CLI Test")
    print(f"Session: {args.session_id}")
    print("----------------------------------------")

    if args.question is None:
        args.question = input("❓ Enter your question: ")

    # Streaming mode
    if args.stream:
        print("\n🤖 AI (streaming): ", end="", flush=True)
        for token in rag.stream_answer(args.question):
            print(token, end="", flush=True)
        print("\n")
    else:
        # Non-streaming
        answer = rag.ask(args.question)
        print("\n🤖 AI Response:\n")
        print(answer)

