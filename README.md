# Applied AI Questions Every Engineer Should Know

> Production-first questions, mental models, and architectural patterns for Applied AI and ML Systems Engineers.

Understanding AI systems does not require deriving matrix calculus by hand every day. It requires knowing **how models process tokens, where memory bottlenecks occur, how retrieval fails, and how to scale inference reliably in production**.

This repository is a structured engineering guide focused on real-world mechanics, architectural trade-offs, and systems-level failure modes.

---

## Table of Contents

- [01. Model Mechanics & Foundations](/1-model-mechanics-and-foundations.md) — Training vs. inference, tokens, attention complexity, activations, decoding params.
- [02. Embeddings & Vector Search](/2-embeddings-and-vector-search.md) — Dense vs. sparse vectors, distance metrics, bi-encoders vs. cross-encoders, ANN indexes.
- [03. RAG Architecture & Retrieval](/3-rag-architecture-and-retrieval.md) — Chunking strategies, query expansion, reranking, hybrid search, metadata filtering.
- [04. Inference Optimization & Serving](/4-inference-optimization-and-serving.md) — KV cache mechanics, continuous batching, quantization (INT8/FP8/AWQ), TTFT vs. ITL.
- [05. Evaluation & Observability](/5-evaluation-and-observability.md) — LLM-as-a-judge, RAG triads, semantic drift, latency/cost telemetry.
- [06. Agentic Patterns & Tool Use](/6-agentic-patterns-and-tool-use.md) — Function calling, structured JSON grammars, ReAct loops, state machines.

---

## Related Repositories

- [cs-questions-should-know](https://github.com/sigmacoy/cs-questions-should-know) — Core data structures, algorithms, operating systems, and networking.
- [hardware-questions-for-cs](https://github.com/sigmacoy/hardware-questions-for-cs) — Silicon architecture, cache lines, memory hierarchies, and mechanical sympathy.