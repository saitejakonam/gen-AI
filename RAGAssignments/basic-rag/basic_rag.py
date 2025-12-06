"""
Basic RAG System using Vector Stores and DIAL API

Implements:
1. Load processed content from extract_content.py
2. Create embeddings and populate vector store
3. Perform similarity search
4. Build context-aware prompt
5. Generate answer using EPAM DIAL API
"""

import os
import json
import argparse
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from utils.dial_client import DIALClient



load_dotenv()


class BasicRAG:
    def __init__(self, vector_store_type="faiss"):
        """
        Initialize embedding model + vector store type.
        """
        self.vector_store_type = vector_store_type.lower()
        self.dial_client = DIALClient()

        # Using SentenceTransformers for embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vector_store = None
        self.documents = []

        print(f"🔧 RAG initialized with vector store: {self.vector_store_type}")

    def load_documents(self, content_dir: str = "data/extracted_content"):
        """
        Load extracted chunks saved by extract_content.py
        """

        filepath = os.path.join(content_dir, "chunks.json")

        if not os.path.exists(filepath):
            raise FileNotFoundError("❌ chunks.json not found. Run extract_content.py first.")

        with open(filepath, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        print(f"📄 Loaded {len(self.documents)} documents.")

    def create_vector_store(self, documents: list):
        """
        Convert documents to vector embeddings and store them.
        """

        texts = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]

        if self.vector_store_type == "faiss":
            self.vector_store = FAISS.from_texts(texts, self.embedding_model, metadatas=metadatas)
            print("📦 FAISS vector store created.")

        elif self.vector_store_type == "chromadb":
            self.vector_store = Chroma.from_texts(texts, self.embedding_model, metadatas=metadatas)
            print("📦 ChromaDB vector store created.")

        else:
            raise ValueError("❌ Invalid vector store type. Choose 'faiss' or 'chromadb'.")

    def retrieve_relevant_docs(self, query: str, k: int = 3):
        """
        Retrieve top-k similar documents.
        """

        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")

        results = self.vector_store.similarity_search(query, k=k)
        print(f"🔍 Retrieved {len(results)} relevant documents.")
        return results

    def generate_response(self, query: str, retrieved_docs: list):
        """
        Generate final answer using DIAL API with retrieved context.
        """

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
You are a helpful AI assistant. Use ONLY the context below to answer the question.

--- Retrieved Context ---
{context}
-------------------------

User Question: {query}

Answer in a clear and concise manner.
"""

        response = self.dial_client.generate(prompt)
        return response

    def query(self, query: str):
        """
        Full RAG pipeline:
        1. Retrieve relevant chunks
        2. Generate answer
        """

        if not self.vector_store:
            self.load_documents()
            self.create_vector_store(self.documents)

        retrieved_docs = self.retrieve_relevant_docs(query)
        answer = self.generate_response(query, retrieved_docs)
        return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic RAG Question Answering")
    parser.add_argument("--query", required=True, help="Query to ask")
    parser.add_argument("--vector-store", default="faiss", choices=["faiss", "chromadb"], help="Vector store type")

    args = parser.parse_args()

    rag = BasicRAG(vector_store_type=args.vector_store)

    print(f"🧠 Asking: {args.query}\n")
    answer = rag.query(args.query)

    print("\n🤖 Answer:")
    print(answer)
