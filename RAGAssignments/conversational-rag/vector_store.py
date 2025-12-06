import os
import json
import hashlib
from typing import List, Tuple

from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class PersistentVectorStore:
    """
    Handles persistent FAISS/Chroma DB with automatic rebuild
    when chunks.json changes.
    """

    def __init__(
        self,
        vector_store_type="faiss",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        content_dir="data/extracted_content",
        persist_dir="data/vector_stores"
    ):
        self.vector_store_type = vector_store_type
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.content_dir = content_dir

        # Store FAISS/Chroma files based on embedding model + store type
        self.store_dir = os.path.join(
            persist_dir,
            f"{vector_store_type}_{embedding_model.replace('/', '_')}"
        )
        os.makedirs(self.store_dir, exist_ok=True)

        self.index_path = os.path.join(self.store_dir, "index")
        self.meta_path = os.path.join(self.store_dir, "metadata.json")
        self.hash_path = os.path.join(self.store_dir, "chunk_hash.txt")

    # -----------------------------------------------------------
    # Load chunks.json
    # -----------------------------------------------------------
    def load_chunks(self) -> Tuple[List[str], List[dict]]:
        chunks_file = os.path.join(self.content_dir, "chunks.json")

        if not os.path.exists(chunks_file):
            raise FileNotFoundError("❌ chunks.json not found! Run extract_content.py")

        with open(chunks_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = [d["content"] for d in data]
        metadata = [d.get("metadata", {}) for d in data]

        return texts, metadata

    # -----------------------------------------------------------
    # Compute hash of all chunk text → detects content changes
    # -----------------------------------------------------------
    def compute_hash(self, texts: List[str]) -> str:
        combined = "".join(texts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def load_saved_hash(self) -> str:
        if not os.path.exists(self.hash_path):
            return ""
        with open(self.hash_path, "r") as f:
            return f.read().strip()

    def save_hash(self, h: str):
        with open(self.hash_path, "w") as f:
            f.write(h)

    # -----------------------------------------------------------
    # Load persistent FAISS
    # -----------------------------------------------------------
    def load_faiss(self):
        if not os.path.exists(self.index_path):
            return None
        try:
            return FAISS.load_local(self.index_path, self.embedding, allow_dangerous_deserialization=True)
        except:
            return None

    # -----------------------------------------------------------
    # Build/rebuild FAISS
    # -----------------------------------------------------------
    def build_faiss(self, texts, metadata):
        print("⚙️ Rebuilding FAISS index...")

        store = FAISS.from_texts(texts, self.embedding, metadatas=metadata)
        store.save_local(self.index_path)

        # Save metadata + hash
        with open(self.meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        new_hash = self.compute_hash(texts)
        self.save_hash(new_hash)

        print("✅ FAISS index rebuilt & saved.")
        return store

    # -----------------------------------------------------------
    # Load persistent Chroma
    # -----------------------------------------------------------
    def load_chroma(self):
        try:
            return Chroma(
                embedding_function=self.embedding,
                persist_directory=self.index_path
            )
        except:
            return None

    # -----------------------------------------------------------
    # Build/rebuild Chroma
    # -----------------------------------------------------------
    def build_chroma(self, texts, metadata):
        print("⚙️ Rebuilding ChromaDB...")

        store = Chroma.from_texts(
            texts,
            self.embedding,
            metadatas=metadata,
            persist_directory=self.index_path
        )
        store.persist()

        with open(self.meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        new_hash = self.compute_hash(texts)
        self.save_hash(new_hash)

        print("✅ ChromaDB rebuilt & persisted.")
        return store

    # -----------------------------------------------------------
    # MAIN: Load persistent DB or rebuild if needed
    # -----------------------------------------------------------
    def load_or_create(self):
        texts, metadata = self.load_chunks()
        new_hash = self.compute_hash(texts)
        old_hash = self.load_saved_hash()

        # Decide whether to rebuild DB
        must_rebuild = (
            not os.path.exists(self.meta_path) or
            new_hash != old_hash
        )

        if must_rebuild:
            print("🔄 Changes detected → Rebuilding vector DB...")
            if self.vector_store_type == "faiss":
                return self.build_faiss(texts, metadata)
            else:
                return self.build_chroma(texts, metadata)

        # Load existing DB
        print("✅ No changes in chunks.json, loading existing vector DB...")

        if self.vector_store_type == "faiss":
            store = self.load_faiss()
        else:
            store = self.load_chroma()

        if store:
            print("📦 Loaded persistent vector store successfully.")
            return store

        # Fallback: rebuild if load failed
        print("⚠️ Index missing or corrupt → rebuilding...")
        if self.vector_store_type == "faiss":
            return self.build_faiss(texts, metadata)
        else:
            return self.build_chroma(texts, metadata)
