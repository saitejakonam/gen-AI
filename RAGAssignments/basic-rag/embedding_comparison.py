"""
Embedding Model Comparison

Compares:
1. Embedding speed
2. Embedding dimensions
3. Semantic similarity quality
4. Computational trade-offs
"""

import time
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingComparison:
    def __init__(self):
        """Initialize multiple embedding models for comparison."""
        self.models = {
            "all-MiniLM-L6-v2": None,
            "all-mpnet-base-v2": None,
            "paraphrase-MiniLM-L3-v2": None,
        }

        self.test_texts = [
            "Artificial intelligence is transforming the world.",
            "Machine learning is a key part of AI.",
            "The cat is sleeping on the sofa.",
            "Quantum computing will change encryption.",
        ]

        # Pairs used to test similarity quality
        self.semantic_pairs = [
            ("AI is changing industries.", "Artificial intelligence transforms businesses."),
            ("The dog is barking loudly.", "A dog makes a loud noise."),
            ("The stock market crashed today.", "There was a major drop in financial markets."),
        ]

        self.unrelated_pairs = [
            ("I love playing football.", "Quantum computing improves cryptography."),
            ("The sun is bright today.", "Economics studies financial systems."),
        ]

    # -------------------------------------------------------------
    # Loading models
    # -------------------------------------------------------------
    def load_sentence_transformers_model(self, model_name: str):
        model = SentenceTransformer(model_name)
        return model

    def load_all_models(self):
        for model_name in self.models:
            print(f"🔄 Loading model: {model_name}")
            self.models[model_name] = self.load_sentence_transformers_model(model_name)
        print("✅ All models loaded!\n")

    # -------------------------------------------------------------
    # Embedding Speed
    # -------------------------------------------------------------
    def measure_embedding_speed(self, texts: List[str], model_name: str) -> float:
        model = self.models[model_name]

        start = time.time()
        model.encode(texts, convert_to_numpy=True)
        end = time.time()

        avg_time = (end - start) / len(texts)
        return avg_time

    # -------------------------------------------------------------
    # Embedding Quality
    # -------------------------------------------------------------
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def evaluate_embedding_quality(self, model_name: str) -> Dict[str, float]:
        model = self.models[model_name]

        semantic_scores = []
        unrelated_scores = []

        # Similar pairs
        for a, b in self.semantic_pairs:
            emb1 = model.encode(a, convert_to_numpy=True)
            emb2 = model.encode(b, convert_to_numpy=True)
            semantic_scores.append(self.cosine_similarity(emb1, emb2))

        # Unrelated pairs
        for a, b in self.unrelated_pairs:
            emb1 = model.encode(a, convert_to_numpy=True)
            emb2 = model.encode(b, convert_to_numpy=True)
            unrelated_scores.append(self.cosine_similarity(emb1, emb2))

        return {
            "semantic_avg_similarity": float(np.mean(semantic_scores)),
            "unrelated_avg_similarity": float(np.mean(unrelated_scores)),
            "separation_score": float(np.mean(semantic_scores) - np.mean(unrelated_scores)),
        }

    # -------------------------------------------------------------
    # Embedding Dimensions
    # -------------------------------------------------------------
    def compare_embedding_dimensions(self) -> Dict[str, int]:
        dims = {}
        for model_name, model in self.models.items():
            dims[model_name] = model.get_sentence_embedding_dimension()
        return dims

    # -------------------------------------------------------------
    # Run full comparison
    # -------------------------------------------------------------
    def run_comparison(self):
        print("\n=== 🧠 Embedding Model Comparison Report ===\n")

        self.load_all_models()

        # ---------------------------------------------------------
        # 1. Dimensions
        # ---------------------------------------------------------
        print("📏 Embedding Dimensions:")
        dims = self.compare_embedding_dimensions()
        for model, dim in dims.items():
            print(f"  - {model}: {dim} dimensions")
        print()

        # ---------------------------------------------------------
        # 2. Speed
        # ---------------------------------------------------------
        print("⏱ Measuring Embedding Speed (avg per text):")
        speeds = {}
        for model_name in self.models:
            avg_time = self.measure_embedding_speed(self.test_texts, model_name)
            speeds[model_name] = avg_time
            print(f"  - {model_name}: {avg_time:.6f} sec per text")
        print()

        # ---------------------------------------------------------
        # 3. Similarity Quality
        # ---------------------------------------------------------
        print("🎯 Evaluating Embedding Quality:")
        quality_scores = {}
        for model_name in self.models:
            quality = self.evaluate_embedding_quality(model_name)
            quality_scores[model_name] = quality
            print(f"\nModel: {model_name}")
            print(f"  - Semantic Similarity:   {quality['semantic_avg_similarity']:.4f}")
            print(f"  - Unrelated Similarity: {quality['unrelated_avg_similarity']:.4f}")
            print(f"  - Separation Score:      {quality['separation_score']:.4f}")

        # ---------------------------------------------------------
        # 4. Dynamic Recommendations
        # ---------------------------------------------------------
        print("\n=== 📌 Dynamic Model Recommendations (Based on Results) ===")

        # Best quality model
        best_quality_model = max(quality_scores.items(), key=lambda x: x[1]["separation_score"])[0]

        # Fastest model
        fastest_model = min(speeds.items(), key=lambda x: x[1])[0]

        # Highest dimensional model
        highest_dim_model = max(dims.items(), key=lambda x: x[1])[0]

        # Lowest dimensional model
        lowest_dim_model = min(dims.items(), key=lambda x: x[1])[0]

        print(f"✔ Highest embedding quality: {best_quality_model} "
                f"(Score: {quality_scores[best_quality_model]['separation_score']:.4f})")

        print(f"✔ Fastest model: {fastest_model} "
                f"({speeds[fastest_model]:.6f} sec/text)")

        print(f"✔ Most expressive (highest dimension): {highest_dim_model} "
                f"({dims[highest_dim_model]} dims)")

        print(f"✔ Most efficient (lowest dimension): {lowest_dim_model} "
                f"({dims[lowest_dim_model]} dims)")

        # Decision based on use-case
        print("\n📌 Suggested Use Cases:")

        for model_name in self.models:
            print(f"\n🔹 {model_name}:")
            print(f"   - Dimension: {dims[model_name]}")
            print(f"   - Speed: {speeds[model_name]:.6f} sec")
            print(f"   - Quality: {quality_scores[model_name]['separation_score']:.4f}")

            if model_name == best_quality_model:
                print("   → Best choice for accuracy-focused applications (semantic search, summarization).")

            if model_name == fastest_model:
                print("   → Ideal for real-time RAG, chatbots, high throughput systems.")

            if model_name == lowest_dim_model:
                print("   → Most memory-efficient (good for edge devices or large-scale vector DBs).")

        print("\n=== End of Comparison ===\n")



if __name__ == "__main__":
    comparison = EmbeddingComparison()
    comparison.run_comparison()
