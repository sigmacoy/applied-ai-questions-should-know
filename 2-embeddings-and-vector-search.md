# 02. Embeddings & Vector Search

> Applied AI engineering questions on dense vs. sparse vectors, distance metrics, bi-encoders, cross-encoders, and Approximate Nearest Neighbor (ANN) index mechanics.

---

## 1. What is the fundamental difference between dense embeddings and sparse lexical vectors?

### Direct Answer
Dense embeddings represent concepts as continuous, low-dimensional floating-point vectors. Sparse vectors represent text across high-dimensional vocabulary spaces where most values are zero. Dense vectors capture abstract semantic meaning. Sparse vectors match exact keywords and lexical identifiers.

### System Mechanics & Trade-offs
```
Dense Embedding:
"payment gateway timeout" ──> [ 0.042, -0.312, 0.891, ..., 0.015 ]  (e.g., 768 or 1536 dimensions)
• All dimensions contain non-zero float values.
• Matches synonyms: "stripe processing delay", "billing checkout lag".
• Fails on exact product keys, alphanumeric identifiers, or rare acronyms.

Sparse Lexical Vector (BM25 / SPLADE):
"payment gateway timeout" ──> { token_412: 2.15, token_9881: 3.41, token_1204: 1.88 } (30k+ dimensions)
• 99.9% of vector coordinates are zero.
• Matches exact alphanumeric tokens: "ERR_504_GATEWAY", "SKU-99214".
• Fails on conceptual paraphrases without shared vocabulary.
```

- **Dense Strengths:** Dense vectors map semantically related phrases to nearby coordinates. Queries find relevant documents even when authors use different words.
- **Sparse Strengths:** BM25 uses term frequency and inverse document frequency (TF-IDF mechanics). It matches exact serial numbers, error codes, and unique entity names with zero semantic drift.
- **Learned Sparsity (SPLADE):** SPLADE uses a transformer to expand queries with predicted related terms while maintaining sparse representation. This bridges lexical precision and semantic recall.

### Production Rule of Thumb
Never build an enterprise search pipeline using dense embeddings alone. Always pair dense embeddings with sparse keyword search (BM25 or SPLADE) in a hybrid retrieval architecture.

---

## 2. Which distance metric should you select: Cosine Similarity, Dot Product, or Euclidean Distance?

### Direct Answer
Cosine Similarity measures the angle between vectors. Dot Product measures both angle and vector magnitude. Euclidean Distance measures geometric straight-line separation. When you normalize vectors to unit length ($L_2$ norm), all three metrics yield identical ranking orders.

### System Mechanics & Trade-offs
```
Vector Comparison Mechanics:
Cosine Similarity:   cos(θ) = (A · B) / ( ||A|| * ||B|| )   ──> Range: [-1.0, 1.0]
Dot Product:         A · B  = Σ (A_i * B_i)                ──> Unbounded
Euclidean Distance:  ||A - B|| = sqrt( Σ (A_i - B_i)^2 )   ──> Range: [0.0, ∞)

When ||A|| = 1.0 and ||B|| = 1.0:
Cosine Similarity = Dot Product
Euclidean Distance^2 = 2 * (1 - Dot Product)
```

- **Computation Overhead:** Cosine similarity requires computing vector magnitudes at query time. This introduces extra square root and division operations.
- **Normalized Dot Product:** If you normalize vectors to unit length ($||V||_2 = 1.0$) during indexing, Dot Product requires only multiply-accumulate operations. Hardware accelerators execute this operation with maximum throughput.
- **Magnitude Significance:** Un-normalized embeddings encode confidence or frequency in their vector length. In these cases, Dot Product and Cosine Similarity produce different search rankings.

### Production Rule of Thumb
Normalize all embedding vectors to unit length before insertion into your vector database. Select **Inner Product (Dot Product)** as your index metric to achieve maximum query throughput on CPU and GPU hardware.

---

## 3. What is the architectural and latency trade-off between Bi-Encoders and Cross-Encoders?

### Direct Answer
Bi-encoders process queries and documents independently to generate separate embeddings. Cross-encoders process the query and document together through joint self-attention. Bi-encoders enable sub-millisecond vector indexing. Cross-encoders provide superior accuracy but require a full forward pass per candidate document.

### System Mechanics & Trade-offs
```
Bi-Encoder Architecture (Fast Retrieval):
Query ────> [ Transformer ] ──> Vector Q ─┐
                                          ├─> Fast Dot Product (Sub-millisecond)
Document ─> [ Transformer ] ──> Vector D ─┘

Cross-Encoder Architecture (High-Accuracy Reranking):
[ Query + Document ] ──> [ Full Self-Attention Transformer ] ──> Relevance Score (0.0 - 1.0)
• Tokens in the query interact directly with all tokens in the document.
• Computation scales linearly with candidate document count.
```

- **Bi-Encoder Scaling:** You can compute document embeddings offline and store them in an Approximate Nearest Neighbor (ANN) index. Query latency remains under 10 milliseconds across millions of vectors.
- **Cross-Encoder Accuracy:** Joint self-attention allows every query token to attend to every document token. This captures complex nuances, negations, and term dependencies that bi-encoders compress away.
- **Latency Cost:** Running 100 documents through a cross-encoder requires 100 forward passes of a transformer. This adds 50 to 200 milliseconds of latency to the query path.

### Production Rule of Thumb
Use a bi-encoder for first-stage retrieval to fetch the top 50 candidates from millions of records. Use a cross-encoder as a second-stage reranker to filter those 50 candidates down to the top 5 chunks.

---

## 4. How does Hierarchical Navigable Small World (HNSW) work, and what are its memory trade-offs?

### Direct Answer
HNSW builds a multi-layer graph where each layer contains skip-lists of nearest neighbors. Upper layers provide long-range hops across vector space. The bottom layer contains dense local connections. HNSW provides fast logarithmic search time but consumes large amounts of system memory.

### System Mechanics & Trade-offs
```
Layer 2 (Express Lane):       (•) ─────────────────────────> (•)
                               │                              │
Layer 1 (Regional Lane):      (•) ───────> (•) ────────────> (•) ───────> (•)
                               │            │                 │            │
Layer 0 (Dense Local Graph):  (•)─(•)─(•)─(•)─(•)─(•)─(•)─(•)─(•)─(•)─(•)─(•)
```

- **Search Mechanics:** The query enters at the top layer. It greedily traverses edges to the closest node until reaching a local minimum. It then drops to the next layer and repeats the search until finishing at Layer 0.
- **Search Complexity:** HNSW achieves $O(\log N)$ search complexity with high recall rates exceeding 95%.
- **Memory Consumption:** HNSW stores all raw vector floats alongside bidirectional graph edge pointers in RAM. A 1,536-dimensional index with $M=16$ neighbors per node requires approximately 8 to 10 KB of RAM per vector.
- **Index Build Cost:** Inserting vectors requires computing nearest neighbors and updating graph links dynamically. Index build operations are compute-heavy and difficult to parallelize across distributed nodes.

### Production Rule of Thumb
Use HNSW when query latency must remain below 10 milliseconds and dataset size fits in available RAM. If vector counts exceed 10 million and RAM budgets are tight, migrate to IVF-PQ or disk-backed vector graphs.

---

## 5. How does Inverted File with Product Quantization (IVF-PQ) compress billion-scale vector indexes?

### Direct Answer
IVF partitions vector space into Voronoi clusters to reduce search space. Product Quantization splits high-dimensional vectors into smaller subvectors and quantizes each into byte centroids. IVF-PQ reduces vector memory footprints by up to 95% at the cost of recall accuracy.

### System Mechanics & Trade-offs
```
Original Vector (1024 dims, 4096 bytes FP32):
[ f0, f1, ..., f127 | f128, ..., f255 | ... | f896, ..., f1023 ]
         │                  │                       │
         ▼                  ▼                       ▼
Subvector 1 (128 dims) Subvector 2 (128 dims)  Subvector 8 (128 dims)
         │                  │                       │
         ▼                  ▼                       ▼
Quantize to Centroid:  Quantize to Centroid:   Quantize to Centroid:
    Byte ID: 42         Byte ID: 189            Byte ID: 77

Compressed Vector (8 bytes total!): [ 42, 189, ..., 77 ] (512x compression)
```

- **Inverted File (IVF):** k-means clusters the dataset into $C$ centroids. During search, the engine inspects only the closest $n_{probe}$ clusters. This skips 90% to 99% of the dataset.
- **Product Quantization (PQ):** A 1024-dimensional vector splits into 8 subvectors of 128 dimensions. Each subvector is mapped to the nearest centroid index in a 256-entry codebook. This compresses 4,096 bytes of float data into 8 bytes.
- **Asymmetric Distance Computation (ADC):** The query vector remains uncompressed. Distances between uncompressed query subvectors and centroid tables are precomputed. This enables extremely fast table lookups.

### Production Rule of Thumb
Deploy IVF-PQ when storing hundreds of millions of vectors on cost-constrained hardware. Always execute an asymmetric re-ranking step against uncompressed vectors on the top 100 retrieved results to restore recall accuracy.

---

## 6. What is the Curse of Dimensionality, and how does it degrade distance metrics in high dimensions?

### Direct Answer
As vector dimensionality grows, the volume of the space increases exponentially. Distances between the nearest neighbor and the farthest point converge to nearly identical values. In high dimensions, standard Euclidean distance loses discriminatory power.

### System Mechanics & Trade-offs
```
Mathematical Convergence Effect:
lim_{d -> ∞} ( Dist_max - Dist_min ) / Dist_min = 0

Result in High Dimensions (d > 1000):
• Almost all random pairs of vectors become orthogonal (Dot Product ~ 0).
• Ratio between nearest neighbor distance and random point distance approaches 1.0.
• Distance distributions concentrate into narrow peaks.
```

- **Concentration of Measure:** In 1,536-dimensional space, all data points sit near the outer boundary of the space. Data points become almost equidistant from one another.
- **Empty Space Problem:** High-dimensional spaces are mostly empty. Finding clusters requires exponentially more training samples to avoid extreme sparsity.
- **Metric Degradation:** Euclidean distance ($L_2$) suffers heavily from noisy and irrelevant dimensions. Cosine distance mitigates magnitude distortion by measuring angular deviation on unit spheres.

### Production Rule of Thumb
Do not arbitrarily increase embedding dimensions hoping for better semantic quality. High-dimensional vectors ($d > 2048$) increase memory and compute costs without proportional recall gains. Prefer dense models between 768 and 1,536 dimensions.

---

## 7. What are Matryoshka Representation Learning (MRL) embeddings, and how do they reduce vector storage costs?

### Direct Answer
Matryoshka embeddings pack the most critical semantic information into early vector dimensions. You can truncate the vector to smaller dimensions while preserving most retrieval accuracy. This enables multi-stage vector search with dramatic savings in memory and network bandwidth.

### System Mechanics & Trade-offs
```
Full Vector (1536 dims, 6144 bytes):
[ v_1, v_2, ..., v_256 | v_257, ..., v_512 | ... | v_1536 ]
 └──── 256 dims ──────┘
       (Retains ~97% of full retrieval accuracy!)

Two-Tier Search Strategy:
Step 1: Search millions of documents using truncated 256-dim index (Fast, low RAM).
Step 2: Rerank top 50 candidates using full 1536-dim vectors (High precision).
```

- **Nested Loss Functions:** MRL models train with separate loss functions applied to nested prefixes of the embedding (for example: 64, 128, 256, 512, and 1536 dimensions).
- **Graceful Degradation:** Truncating an OpenAI or modern open-weight MRL embedding from 1,536 to 256 dimensions cuts storage by 83% while sacrificing less than 2% in retrieval recall.
- **Caching Efficiency:** Storing 256-dimensional vectors in fast memory lets you retain entire indexes in CPU L3 cache or GPU SRAM during first-pass filtering.

### Production Rule of Thumb
Use Matryoshka-enabled embeddings to truncate vectors by 50% to 75% for large-scale indexing. Store truncated vectors in memory for rapid approximate search, and store full-dimensional vectors in cold storage for final ranking.
