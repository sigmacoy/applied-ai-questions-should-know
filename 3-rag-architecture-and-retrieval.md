# 03. RAG Architecture & Retrieval

> Applied AI engineering questions on chunking strategies, hybrid search, query expansion, reranking, metadata filtering, and context window organization.

---

## 1. What chunking strategy should you choose: fixed-size, recursive character, semantic, or structure-aware?

### Direct Answer
Fixed-size chunking splits text at strict character or token intervals. Recursive character chunking splits text along hierarchical delimiters to preserve paragraphs. Semantic chunking splits text based on embedding distance shifts between sentences. Structure-aware chunking parses document syntax (Markdown, HTML, PDF tables) to keep logical elements intact.

### System Mechanics & Trade-offs
```
Document Flow:
[ Document ] ──> Structural Parser (Markdown / HTML / PDF)
                     │
                     ├── Level 1: Split on H1/H2 Headers
                     ├── Level 2: Split on Paragraphs (\n\n)
                     ├── Level 3: Keep Tables / Code Blocks Unified
                     └── Level 4: Fallback to Sentence Delimiters (.)
```

- **Fixed-Size Failure Modes:** Cutting text at 500 tokens often splits sentences and compound nouns in half. This breaks semantic meaning and destroys search recall for keywords located at chunk boundaries.
- **Recursive Character Chunking:** This strategy inspects delimiters in order: `\n\n`, `\n`, space, and empty string. It fills chunks up to target size while preserving complete sentences and paragraphs.
- **Semantic Chunking Cost:** This approach compares embeddings of neighboring sentences. It creates a split when semantic similarity drops below a threshold. It produces coherent chunks but requires multiple model forward passes during ingestion.
- **Structure-Aware Chunking:** Parses tables, code functions, and lists as atomic units. Splitting a financial table across chunks destroys row-column relationships and generates hallucinations.

### Production Rule of Thumb
Default to recursive character chunking with an overlap of 10% to 15%. For technical documentation, financial filings, and structured PDFs, always use a structure-aware parser to prevent splitting tables and code blocks.

---

## 2. How does Hybrid Search combine dense vector search with sparse keyword search using Reciprocal Rank Fusion (RRF)?

### Direct Answer
Hybrid search executes dense semantic retrieval and sparse lexical search in parallel. Reciprocal Rank Fusion merges their separate ranking lists without requiring score normalization. RRF calculates document scores based on reciprocal ranking positions in each list.

### System Mechanics & Trade-offs
```
User Query: "Error code 404 in payment service"
     │
     ├── Dense Vector Search ──> Ranked List A: [ Doc 12, Doc 45, Doc 89, ... ]
     │
     └── Sparse BM25 Search   ──> Ranked List B: [ Doc 45, Doc 99, Doc 12, ... ]
                                         │
                                         ▼
                           [ Reciprocal Rank Fusion (RRF) ]
                                         │
                                         ▼
                               Unified Ranked List
```

- **Reciprocal Rank Fusion Formula:**
  $$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  $M$ is the set of search systems. $r_m(d)$ is the rank position of document $d$. $k$ is a constant (typically $k=60$) that smooths outlier rankings.
- **Score Normalization Problem:** Dense search produces cosine similarities (between -1.0 and 1.0). BM25 produces unbounded positive floating-point scores. Direct score addition fails without careful calibration. RRF solves this problem by using rank positions instead of raw scores.
- **Failure Mode Defense:** Dense search handles semantic concepts and paraphrases. Sparse search handles exact part numbers, variable names, and error codes. Combining both eliminates single-retriever blind spots.

### Production Rule of Thumb
Deploy hybrid search using dense vectors plus BM25 merged with RRF ($k=60$). This combination consistently improves top-10 retrieval recall by 15% to 30% over dense retrieval alone.

---

## 3. What is Query Transformation (HyDE, Multi-Query, Step-Back), and how does it bridge the query-document gap?

### Direct Answer
User queries are short and ambiguous. Documents are detailed and structured. Query transformation uses an LLM to rewrite, expand, or hypothesize content before retrieval. This aligns the semantic representation of the query with indexed candidate documents.

### System Mechanics & Trade-offs
```
Query Transformation Techniques:

1. HyDE (Hypothetical Document Embeddings):
Query ──> [ LLM ] ──> Generate "Fake" Answer ──> Embed Answer ──> Search Documents

2. Multi-Query Expansion:
Query ──> [ LLM ] ──> Generate 3-5 Query Variations ──> Search All ──> Union & Deduplicate

3. Step-Back Prompting:
Specific Query ──> [ LLM ] ──> Abstract High-Level Concept ──> Search Foundational Context
```

- **HyDE Mechanics:** A user asks, *"Why did connection pool drop packets?"* The question has low vector overlap with technical troubleshooting guides. HyDE instructs an LLM to generate a hypothetical answer. The embedding of the hypothetical answer sits much closer to real technical documents in vector space.
- **Multi-Query Expansion:** Generates semantic synonyms and alternative technical phrasings. It retrieves documents when users use informal jargon instead of formal documentation terminology.
- **Latency & Cost Trade-off:** Every query transformation adds an LLM generation call to the request path. This introduces 200 to 800 milliseconds of latency and extra token costs before retrieval begins.

### Production Rule of Thumb
Avoid running HyDE on latency-critical user queries. Reserve HyDE and Multi-Query expansion for asynchronous agents, complex research assistants, or high-value customer workflows where accuracy outranks latency.

---

## 4. Why is a Cross-Encoder Reranker essential in production RAG pipelines?

### Direct Answer
Vector retrieval prioritizes speed over precision and admits false positives. A cross-encoder reranker inspects full token interactions between the query and candidate chunks. It reorders results accurately and eliminates distractor chunks before context injection.

### System Mechanics & Trade-offs
```
First-Stage Retrieval (Bi-Encoder / ANN):
Database: 1,000,000 Chunks ──> High Recall, Low Precision ──> Top 50 Chunks (Fast: ~10ms)
                                                                    │
                                                                    ▼
Second-Stage Reranking (Cross-Encoder / Cohere / BGE-Reranker):
Top 50 Chunks + Query ──> Deep Token-to-Token Cross-Attention ───> Top 5 Chunks (High Precision: ~50ms)
                                                                    │
                                                                    ▼
                                                            Injected into LLM Context
```

- **Eliminating Distractors:** Bi-encoders compress entire chunks into single fixed-size vectors. Chunks that match the broad topic but lack the specific answer score high in vector search. A cross-encoder reads exact token alignments to penalize irrelevant text.
- **Context Window Economy:** Passing 50 chunks directly into an LLM balloons prompt token costs and degrades reasoning quality. Reranking filters the pool to the 3 to 5 most relevant chunks.
- **Hardware Profile:** Rerankers are small transformer models (e.g., 100M to 600M parameters). You can host them on low-cost GPU instances or modern CPU nodes with ONNX runtime.

### Production Rule of Thumb
Always configure two-stage retrieval. Fetch 50 candidate chunks using hybrid search, and pass them through a cross-encoder reranker to extract the top 5 chunks for LLM generation.

---

## 5. How does Metadata Filtering work, and why does Pre-Filtering cause fragmentation in HNSW vector graphs?

### Direct Answer
Metadata filtering restricts vector search results by scalar attributes like user ID, tenant, or date. Pre-filtering applies metadata constraints before vector graph traversal. This disconnects graph paths in HNSW indexes and causes premature search termination.

### System Mechanics & Trade-offs
```
Pre-Filtering on HNSW Graph:
Unfiltered Graph:  (A) ─── (B) ─── (C) ─── (D) ─── (E)
Filter: Tenant = 'Alpha'
Filtered Nodes:    (A)      [X]     [X]     (D)      [X]   <-- Nodes B, C, E are disabled!

Result: Traversal gets stuck at node (A). It cannot reach node (D) because connecting paths are severed.
```

- **Pre-Filtering Trap:** If a metadata filter eliminates 95% of nodes, the HNSW graph breaks into isolated islands. The greedy search algorithm halts at dead ends and fails to return valid matches.
- **Post-Filtering Trap:** The database performs a standard top-$K$ vector search and removes non-matching metadata records afterward. If none of the top-$K$ vectors match the filter, the query returns an empty list.
- **Iterative Single-Stage Filtering:** Modern vector engines traverse the graph while evaluating metadata on the fly. When encountering an invalid node, the algorithm explores its neighbors without scoring the node.

### Production Rule of Thumb
For strict multi-tenant isolation, partition data physically using tenant-specific indexes or relational foreign keys in PostgreSQL (pgvector). Never rely on post-filtering for multi-tenant security boundaries.

---

## 6. What is the "Lost in the Middle" phenomenon, and how should you order chunks in the prompt context?

### Direct Answer
LLMs pay highest attention to tokens placed at the extreme beginning and end of long prompts. Models frequently overlook information placed in the middle of long contexts. You should order retrieved chunks to place the highest-relevance content at the start and end of the context window.

### System Mechanics & Trade-offs
```
Attention Distribution Across Context Window:
High Attention  ▲   ████                                           ████
                │   ████                                           ████
                │   ████                                           ████
                │   ████             "Lost in the Middle"          ████
Low Attention   ▼   ████═══════════════════════════════════════════████
                    0% (Start)                   50% (Middle)      100% (End)
                    System Prompt / Chunks #1-2  Chunks #3-4       Chunk #5 / User Query
```

- **U-Shaped Attention Curve:** Attention mechanisms suffer from positional bias. Tokens near the system prompt and near the final user question receive higher attention weights during self-attention computation.
- **Context Dilution:** Adding low-relevance chunks to fill the context window increases distraction. Irrelevant chunks in the middle degrade model reasoning and trigger false assertions.
- **Strategic Placement:** If your reranker returns five chunks ordered by relevance [1, 2, 3, 4, 5], assemble the prompt in alternating fashion: `[Chunk 1, Chunk 3, Chunk 5, Chunk 4, Chunk 2]`. This keeps top chunks at the edges.

### Production Rule of Thumb
Do not pack unnecessary chunks into your prompt just because the context window permits it. Pass only the top 3 to 5 reranked chunks. Place the most important chunk immediately above the final user prompt.

---

## 7. What is Contextual Retrieval, and why does prepending context headers solve chunk ambiguity?

### Direct Answer
Individual chunks often lose document context and contain ambiguous pronouns. Contextual Retrieval uses an LLM at ingestion time to generate a short context summary for each chunk. Prepending this summary before embedding prevents scope ambiguity and improves retrieval recall.

### System Mechanics & Trade-offs
```
Raw Chunk (Ingested as-is):
"In the third quarter, revenue increased by 15% due to cloud adoption."
• Problem: Which company? Which year? Dense search fails on "Microsoft 2024 cloud revenue".

Contextual Chunk (Prepended Header):
"This document discusses Microsoft FY2024 financial results.
The section covers cloud infrastructure performance.
Chunk content: In the third quarter, revenue increased by 15% due to cloud adoption."
• Result: Embedding and BM25 index capture company, year, and business unit accurately.
```

- **The Context Amnesia Problem:** Document chunking destroys document hierarchy. A chunk describing a product configuration parameter often lacks the product name mentioned five pages earlier.
- **Ingestion Mechanics:** During document processing, a fast, lightweight LLM reviews the entire document alongside each chunk. It generates a 50 to 100 token explanatory summary prepended directly to the chunk text.
- **Dual Indexing Advantage:** The prepended context enhances both dense vector embeddings and sparse BM25 token frequencies without requiring manual metadata tagging.

### Production Rule of Thumb
Implement contextual chunk headers during the document ingestion pipeline. A 60-word contextual prefix prepended to each chunk reduces failed retrieval queries by up to 40% in enterprise document repositories.
