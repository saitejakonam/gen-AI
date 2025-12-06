# 📘 RAG Concepts & Technology Stack – Assignment 1

This document explains the core concepts, techniques, and tools used in Retrieval-Augmented Generation (RAG), including embeddings, vector stores, chunking, indexing, retrieval, and an introduction to Graph-RAG.

It focuses only on concepts, not your project structure.

## 🧠 1. Retrieval-Augmented Generation (RAG)

RAG = LLM + External Knowledge Retrieval

Instead of relying only on an LLM’s internal knowledge, RAG retrieves information from a knowledge base and injects it into the prompt before generating an answer.

Why RAG?

- Reduces hallucinations
- Enables answering domain-specific & private data questions
- Keeps LLM responses grounded
- Allows continuous knowledge updates without re-training

RAG Pipeline  
User Query → Query Embedding → Vector Search → Top-K Chunks → Prompt Construction → LLM Response → Return Answer

## 📚 2. Document Preprocessing Concepts

To use documents in RAG, we must preprocess them.

### 2.1 Text Extraction

Extract content from:

- PDFs
- Text files
- Websites
- Markdown / HTML

### 2.2 Text Chunking

Large documents are split into smaller pieces called chunks.

Why chunk?

- Improves retrieval accuracy
- Prevents LLM token overflow
- Makes search more fine-grained

Chunking Strategies

- Fixed chunk size (e.g., 300–500 tokens)
- Overlapping chunks
- Semantic chunking (optional, advanced)

## 🔢 3. Embeddings

Embeddings convert text into numerical representations (vectors).

You used Sentence-Transformers models such as:

- all-MiniLM-L6-v2 (fast, 384-dim)
- all-mpnet-base-v2 (high quality, 768-dim)
- paraphrase-MiniLM-L3-v2 (lightweight)

What embeddings do:

- Capture semantic meaning
- Enable similarity search
- Map text to a vector space where similar texts are closer

Similarity Metrics:

- Cosine similarity
- Dot product
- L2 distance

## 🗄️ 4. Vector Databases (Vector Stores)

Vector DBs store embeddings + metadata and support fast similarity search.

You used two vector stores:

### 4.1 FAISS (Facebook AI Similarity Search)

Best for: large datasets, speed, local CPU/GPU performance.

Pros:

- Extremely fast
- Supports indexing algorithms (Flat, IVFFlat, HNSW, PQ)
- Good for scalable offline use

Cons:

- Not persistent by default
- Not designed as a distributed database

### 4.2 ChromaDB

Best for: simple, persistent vector storage.

Pros:

- Built-in persistence
- Easy to integrate
- Good for small-to-medium datasets

Cons:

- Not as fast as FAISS for large-scale retrieval
- No GPU acceleration

## 🔍 5. Similarity Search

To answer a user query, embeddings are used to retrieve top-K matching chunks.

query → embed → vector DB → top-K similar chunks

This is the core retrieval step of RAG.

## 🧱 6. Indexing Concepts

Vector DBs require index structures to speed up similarity search:

FAISS indexing types:

- Flat Index — exact search, slowest but most accurate
- IVF Index — partitions data for faster lookup
- HNSW Graph — graph-based, very fast for large datasets
- PQ (Product Quantization) — compresses vectors for memory efficiency

Chroma indexing

- Uses simple embedded indexes
- Suitable for moderate-size datasets

## 🧠 7. Prompt Engineering with Retrieved Context

RAG uses retrieved information to build the final prompt:

Relevant Context:  
[Chunk0] ...  
[Chunk1] ...

User Question:  
"What is fine-tuning?"

Answer using the context above.

This enforces factual grounding.

## 🤖 8. LLM Generation

After retrieval, the prompt is passed to an LLM such as:

- OpenAI GPT models
- EPAM DIAL models
- Local LLMs (LLaMA, Mistral, etc.)

The LLM produces the final response using both:

- Retrieved chunks
- Conversation history (if used)

## ⚡ 9. Query Performance Concepts

You analyzed:

- Indexing performance (time to create FAISS/Chroma index)
- Retrieval latency (query response time)
- Storage footprint

General results:

- FAISS → faster search, better for large datasets
- Chroma → persistent and easy to use

## 📊 10. Embedding Model Comparison Concepts

You compared embedding models on:

- Dimensionality
- Speed
- Semantic accuracy
- Performance in similarity search

Key idea:  
Higher-dimensional models produce better semantic representations but are slower.

## 🌐 11. Graph-RAG (Advanced RAG Concept)

Graph-RAG is an advanced retrieval technique where documents are stored as a knowledge graph, not as independent chunks.

Why Graph-RAG?

- Captures relationships between entities
- Enables multi-hop reasoning
- Provides more structured understanding
- Reduces irrelevant chunk retrieval

Graph-RAG Pipeline  
Data → Entity Extraction → Relation Extraction → Knowledge Graph → Query → Graph Traversal → LLM Uses Structured Context

### Components of Graph-RAG

#### 11.1 Entity Extraction

Identify:

- People
- Organizations
- Topics
- Concepts

#### 11.2 Relation Extraction

Define how entities are connected:

- COVID → caused_by → SARS-CoV-2  
- Elon Musk → founded → SpaceX

#### 11.3 Knowledge Graph

Graph databases used:

- Neo4j
- ArangoDB
- Memgraph
- NebulaGraph

#### 11.4 Graph Retrieval

Instead of retrieving top-K chunks, Graph-RAG retrieves:

- Nodes
- Edges
- Subgraphs

#### 11.5 LLM Integration

LLM receives structured graph context:

Entities:  
- Apple Inc.  
- iPhone

Relations:  
- Apple manufactures iPhone

Question:  
"Who manufactures the iPhone?"

Graph-RAG vs Standard RAG

| Feature    | Standard RAG       | Graph-RAG              |
|------------|--------------------|------------------------|
| Retrieval  | Embedding similarity | Graph traversal        |
| Data Structure | Chunks         | Nodes + edges          |
| Strength   | Semantic match     | Relationship reasoning |
| Weakness   | No structure       | Complex to build       |

## 🧩 12. Key Differences: Vector RAG vs Graph RAG

| Concept         | Vector-Based RAG | Graph-RAG             |
|-----------------|------------------|-----------------------|
| Storage         | Embeddings       | Knowledge Graph       |
| Retrieval       | Cosine similarity| Graph algorithms      |
| Context Quality | High for semantics | High for structured facts |
| Good For        | Unstructured text | Connected knowledge   |

## 🏁 13. Summary of Concepts in Assignment 1

- ✔ Chunking
- ✔ Embeddings
- ✔ FAISS vs Chroma
- ✔ Similarity Search
- ✔ Persistent Vector Store
- ✔ Prompt Construction
- ✔ LLM Response Generation
- ✔ Embedding Model Comparison
- ✔ Indexing Performance Evaluation
- ✔ Graph-RAG Overview