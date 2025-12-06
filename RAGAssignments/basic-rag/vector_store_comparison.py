"""
Compare FAISS vs ChromaDB vector stores.

Measures:
1. Indexing performance (time to create index)
2. Retrieval performance (query latency)
3. Storage requirements (disk size)
4. Trade-offs and recommendations
"""

import os
import time
import json
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreComparison:
    def __init__(self):
        """Initialize embedding model and placeholder stores."""
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.faiss_store = None
        self.chroma_store = None

        self.content_dir = "data/extracted_content"
        self.chroma_dir = "vector_comparison/chromadb_store"
        os.makedirs(self.chroma_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # Setup Stores
    # -------------------------------------------------------------------
    def setup_faiss_store(self, documents: List[str]):
        start = time.time()
        self.faiss_store = FAISS.from_texts(documents, self.embedding_model)
        end = time.time()
        return end - start  # index creation time

    def setup_chromadb_store(self, documents: List[str]):
        start = time.time()
        self.chroma_store = Chroma.from_texts(
            documents,
            self.embedding_model,
            persist_directory=self.chroma_dir
        )
        self.chroma_store.persist()
        end = time.time()
        return end - start  # index creation time

    # -------------------------------------------------------------------
    # Query Performance
    # -------------------------------------------------------------------
    def measure_query_performance(self, query: str, num_queries: int = 50):
        faiss_times = []
        chroma_times = []

        # FAISS performance
        for _ in range(num_queries):
            start = time.time()
            self.faiss_store.similarity_search(query, k=3)
            faiss_times.append(time.time() - start)

        # CHROMA performance
        for _ in range(num_queries):
            start = time.time()
            self.chroma_store.similarity_search(query, k=3)
            chroma_times.append(time.time() - start)

        return {
            "faiss_avg_query_time": sum(faiss_times) / len(faiss_times),
            "chroma_avg_query_time": sum(chroma_times) / len(chroma_times),
        }

    # -------------------------------------------------------------------
    # Storage Comparison
    # -------------------------------------------------------------------
    def compare_storage_requirements(self):
        storage = {}

        # FAISS is in-memory, but we can save it to measure file size
        faiss_path = "vector_comparison/faiss_index"
        os.makedirs("vector_comparison", exist_ok=True)
        self.faiss_store.save_local(faiss_path)

        faiss_size = sum(
            os.path.getsize(os.path.join(faiss_path, f))
            for f in os.listdir(faiss_path)
        )
        storage["faiss_size_bytes"] = faiss_size

        # Chroma persists automatically
        chroma_size = sum(
            os.path.getsize(os.path.join(self.chroma_dir, f))
            for f in os.listdir(self.chroma_dir)
        )
        storage["chroma_size_bytes"] = chroma_size

        return storage

    # -------------------------------------------------------------------
    # Run Full Comparison
    # -------------------------------------------------------------------
    def run_comparison(self):
        print("\n=== 📊 Vector Store Comparison Report (FAISS vs ChromaDB) ===\n")

        # Load documents
        chunks_file = os.path.join(self.content_dir, "chunks.json")
        if not os.path.exists(chunks_file):
            print("❌ chunks.json not found. Run extract_content.py first.")
            return

        with open(chunks_file, "r", encoding="utf-8") as f:
            docs_json = json.load(f)

        documents = [d["content"] for d in docs_json]

        print(f"📄 Loaded {len(documents)} chunks for comparison.")

        # ---------------------------------------------------------------
        # 1. Indexing Performance
        # ---------------------------------------------------------------
        print("\n⏱ Measuring Index Creation Time...")
        faiss_time = self.setup_faiss_store(documents)
        chroma_time = self.setup_chromadb_store(documents)

        print(f"FAISS Indexing Time:  {faiss_time:.4f} sec")
        print(f"Chroma Indexing Time: {chroma_time:.4f} sec")

        # ---------------------------------------------------------------
        # 2. Query Performance
        # ---------------------------------------------------------------
        print("\n🚀 Measuring Query Performance (avg latency)...")
        query = "Explain the topic of the document."
        query_results = self.measure_query_performance(query)

        print(f"FAISS Avg Query Time:  {query_results['faiss_avg_query_time']:.6f} sec")
        print(f"Chroma Avg Query Time: {query_results['chroma_avg_query_time']:.6f} sec")

        # ---------------------------------------------------------------
        # 3. Storage Size
        # ---------------------------------------------------------------
        print("\n💾 Measuring Storage Requirements...")
        storage = self.compare_storage_requirements()

        print(f"FAISS Storage:  {storage['faiss_size_bytes']} bytes")
        print(f"Chroma Storage: {storage['chroma_size_bytes']} bytes")

                # ---------------------------------------------------------------
        # 4. Dynamic Conclusion (based on results)
        # ---------------------------------------------------------------
        print("\n=== 🧠 Data-Driven Summary & Recommendations ===\n")

        results = {
            "faiss": {
                "index_time": faiss_time,
                "query_time": query_results["faiss_avg_query_time"],
                "storage": storage["faiss_size_bytes"],
            },
            "chroma": {
                "index_time": chroma_time,
                "query_time": query_results["chroma_avg_query_time"],
                "storage": storage["chroma_size_bytes"],
            }
        }

        # Determine best performers
        best_index = min(results, key=lambda x: results[x]["index_time"])
        best_query = min(results, key=lambda x: results[x]["query_time"])
        best_storage = min(results, key=lambda x: results[x]["storage"])

        print("🏃 Indexing Speed (lower is better):")
        print(f"  → Fastest: {best_index} ({results[best_index]['index_time']:.4f} sec)\n")

        print("⚡ Query Latency (lower is better):")
        print(f"  → Fastest: {best_query} ({results[best_query]['query_time']:.6f} sec)\n")

        print("💾 Storage Efficiency (lower is better):")
        print(f"  → Most efficient: {best_storage} ({results[best_storage]['storage']} bytes)\n")

        # -----------------------------------------------------------
        # Automatically generate explanation based on patterns
        # -----------------------------------------------------------
        print("📌 Overall Recommendations (Generated Dynamically):\n")

        # Case 1 → One store wins most categories
        scores = {
            "faiss": 0,
            "chroma": 0
        }
        for metric in ["index_time", "query_time", "storage"]:
            if results["faiss"][metric] < results["chroma"][metric]:
                scores["faiss"] += 1
            else:
                scores["chroma"] += 1

        if scores["faiss"] == 3:
            print("✔ FAISS clearly outperforms ChromaDB in ALL metrics measured.\n"
                  "→ Best choice for high-performance RAG systems requiring speed + efficiency.\n")

        elif scores["chroma"] == 3:
            print("✔ ChromaDB outperforms FAISS in ALL metrics measured.\n"
                  "→ Best choice for scalable, persistent vector search workloads.\n")

        else:
            # Mixed results → Explain strengths per metric
            print("✔ Mixed performance detected — each store has strengths.")
            if best_index == "faiss":
                print("  - FAISS indexes documents faster.")
            else:
                print("  - ChromaDB indexes documents faster.")

            if best_query == "faiss":
                print("  - FAISS returns results faster (lower latency).")
            else:
                print("  - ChromaDB returns results faster (lower latency).")

            if best_storage == "faiss":
                print("  - FAISS uses less disk space.")
            else:
                print("  - ChromaDB uses less disk space.")

            print("\n📌 Choose FAISS when:")
            print("  - You need maximum speed for RAG queries")
            print("  - Memory usage is manageable")
            print("  - Persistence is not mandatory\n")

            print("📌 Choose ChromaDB when:")
            print("  - You need persistent on-disk storage")
            print("  - You want multi-user or scalable deployments")
            print("  - Disk space usage is not a problem\n")

        print("=== End of Report ===\n")


if __name__ == "__main__":
    comparison = VectorStoreComparison()
    comparison.run_comparison()
