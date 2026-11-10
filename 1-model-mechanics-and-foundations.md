# 01. Model Mechanics & Foundations

> Applied AI engineering questions on training mechanics, tokenization, attention complexity, activations, and decoding parameters.

---

## 1. What is the operational difference between LLM training and LLM inference?

### Direct Answer
Training optimizes model weights through forward and backward passes. Inference uses fixed weights to predict tokens sequentially with forward passes only. Training requires massive compute and memory for gradients and optimizer states. Inference requires low latency and high memory bandwidth for token generation.

### System Mechanics & Trade-offs
```
Training Phase (Compute & Memory-Capacity Bound):
Input Tokens ──> [ Forward Pass ] ──> Loss Calculation ──> [ Backward Pass (Gradients) ] ──> Optimizer Update (AdamW)
                    │                                             │                                │
                    └── Activations (~2-4 B/param)                └── Gradients (~2-4 B/param)     └── States (~8-12 B/param)

Inference Phase (Memory-Bandwidth Bound):
Prompt Tokens ──> [ Prefill Stage (Parallel) ] ──> Compute initial KV Cache
                         │
                         ▼
                   [ Decode Stage (Sequential) ] ◄── Read Weights + KV Cache per token
                         │
                         └──> Generated Token (1 by 1)
```

- **Memory Footprint:** Training requires 16 to 24 bytes per parameter. This budget includes weights, gradients, activations, and AdamW optimizer states. Inference requires only 1 to 2 bytes per parameter for quantized weights plus Key-Value (KV) cache memory.
- **Compute vs. Bandwidth:** Training is compute-bound with large batch sizes that saturate Tensor Cores. Inference decoding is memory-bandwidth bound. The hardware must stream gigabytes of weights from memory for every single generated token.
- **State Mutability:** Training modifies weights continuously across epochs. Inference keeps weights static in read-only memory and modifies only runtime KV cache tensors.

### Production Rule of Thumb
Do not size inference clusters using training hardware metrics. Training optimizes for raw floating-point operations (FLOPs) per dollar. Serving optimizes for High Bandwidth Memory (HBM) throughput and memory capacity per concurrent user stream.

---

## 2. What is a token, and why does Byte-Pair Encoding (BPE) cause failures in arithmetic and code?

### Direct Answer
A token is a subword numerical representation created by a statistical tokenizer. Byte-Pair Encoding merges frequent character sequences into single tokens. Tokenizers split numbers and code syntax unpredictably. This fragmentation prevents the model from seeing consistent digit columns or individual characters.

### System Mechanics & Trade-offs
```
Input String:       "35824"
BPE Tokenization:   ["358", "24"]   <-- Tokenized into unequal digit slices

Input String:       " 35824"
BPE Tokenization:   [" ", "35", "824"] <-- Leading space changes token boundaries!
```

- **Loss of Positional Value:** Models do not receive raw digits. The number `35824` might split into tokens `["358", "24"]`, while `35825` splits into `["35", "825"]`. The model cannot apply uniform column-based arithmetic algorithms across these tokens.
- **Whitespace Sensitivity:** BPE merges spaces with alphanumeric characters. An extra space at the beginning of an input string completely changes downstream token boundaries.
- **Code Indentation Overhead:** Code relies heavily on repeated indentation. Inefficient tokenizers assign individual tokens to single spaces. This multiplies sequence lengths and burns context window capacity.

### Production Rule of Thumb
Always format numerical inputs with explicit delimiters (such as commas or spaces between single digits) when you require exact arithmetic. When processing code or structured data, verify that your tokenizer implements byte-level fallback without character loss.

---

## 3. What is the computational and memory complexity of standard Multi-Head Attention (MHA)?

### Direct Answer
Standard Multi-Head Attention scales quadratically with sequence length. The self-attention operation has a computational complexity of $O(N^2 \cdot d)$ and a memory complexity of $O(N^2)$. $N$ represents sequence length and $d$ represents hidden dimension size.

### System Mechanics & Trade-offs
```
Query (Q):  [ N x d ]
Key (K):    [ N x d ]
Attention:  Q x K^T  ──> Score Matrix: [ N x N ]  <-- Quadratic Growth ($N^2$)
Value (V):  [ N x d ]
Softmax:    Softmax( (Q x K^T) / sqrt(d) ) x V ──> Output: [ N x d ]
```

- **The Quadratic Bottleneck:** Generating attention scores requires computing the dot product between every query token and every key token. Doubling sequence length $N$ quadruples both computation steps and memory storage for intermediate attention scores.
- **GPU High Bandwidth Memory (HBM) Wall:** Standard attention writes the intermediate $N \times N$ matrix to GPU HBM before computing Softmax. At long contexts ($N > 32\text{k}$), intermediate attention matrices exceed GPU memory limits.
- **FlashAttention Optimization:** FlashAttention does not change the mathematical result. It tiles the inputs into high-speed GPU SRAM cache. It computes Softmax incrementally without writing the massive $N \times N$ matrix back to slow HBM.

### Production Rule of Thumb
Never serve long-context models without FlashAttention or FlashDecoding enabled in your runtime engine. For sequence lengths exceeding 8,192 tokens, un-fused quadratic attention causes immediate GPU out-of-memory (OOM) crashes.

---

## 4. What are residual connections and non-linear activations (e.g., SwiGLU), and why do deep transformers need them?

### Direct Answer
Residual connections pass unmodified signals directly across transformer layers to prevent vanishing gradients. Non-linear activation functions allow neural networks to learn non-linear patterns. SwiGLU combines Swish activation with a gating mechanism to accelerate convergence.

### System Mechanics & Trade-offs
```
Residual Stream Architecture:
Input x ──┬──────────────────────────────────────────[+]──> Output Layer
          │                                           ▲
          └──> [ LayerNorm ] ──> [ MHA / MLP Block ] ─┘
```

- **Residual Highway:** Deep networks with 32 to 128 layers suffer from vanishing and exploding gradients. Residual connections add the input directly to the layer output: $x_{l+1} = x_l + F(x_l)$. The gradient flows through the addition operator without decay during backpropagation.
- **SwiGLU Mechanics:** Standard activations like ReLU zero out negative values. SwiGLU computes:
  $$\text{SwiGLU}(x) = (x W \cdot \text{sigmoid}(x W)) \otimes (x V)$$
  The gating branch controls information flow dynamically. It improves model stability and downstream reasoning benchmark scores.
- **Compute Trade-off:** Gated linear units require three weight matrices per MLP block instead of two. This increases parameter count and matrix operations by 50% for the feed-forward network layer.

### Production Rule of Thumb
Modern open-weight models (such as Llama 3 and Mistral) use SwiGLU and RMSNorm as their default layer blocks. When evaluating architectures, prefer SwiGLU models over legacy ReLU models for superior parameter-to-reasoning efficiency.

---

## 5. How do decoding parameters (Temperature, Top-p, Top-k, Min-p) alter output probability distributions?

### Direct Answer
Decoding parameters control token selection from the final logit distribution. Temperature rescales logit sharpness. Top-k caps sampling to the top $k$ candidates. Top-p limits choices to tokens within a cumulative probability threshold. Min-p eliminates tokens whose probability falls below a fraction of the top token.

### System Mechanics & Trade-offs
```
Logits: [ 10.0, 8.0, 7.5, 3.0, 0.1 ] 
   │
   ├── Temperature (T): z_i' = z_i / T
   │     • T -> 0: Logits become ultra-sharp (Greedy / Argmax).
   │     • T = 1.0: Original distribution.
   │     • T > 1.0: Flattens distribution (Increases entropy and randomness).
   │
   ├── Softmax: Converts scaled logits into probabilities P(w_i).
   │
   ├── Top-k: Keeps only the k highest probability tokens. Discards the rest.
   ├── Top-p: Sorts tokens; keeps the smallest set whose sum >= p (e.g., 0.90).
   └── Min-p: Keeps tokens where P(token) >= Min-p * P(top_token).
```

- **Temperature Failure Modes:** Setting Temperature to $0$ creates deterministic outputs. This works best for code generation and exact JSON schemas. Setting Temperature above $1.0$ forces the sampler to pick low-probability tail tokens. This causes hallucinations and syntax breakdown.
- **Top-p Dynamic Truncation:** Top-p adapts candidate pool size dynamically. When the model is confident, the top token has $p=0.98$; Top-p at $0.90$ selects only that single token. When uncertain, Top-p expands candidate choices.
- **Min-p Superiority:** Top-p can accidentally include nonsense tokens in the flat tail of a distribution. Min-p scales candidate acceptance relative to the most confident token. This prevents incoherent tail sampling during creative tasks.

### Production Rule of Thumb
For structured data extraction, tool calling, and deterministic JSON generation, set `temperature = 0.0`. For balanced conversational agents, use `temperature = 0.7` combined with `top_p = 0.9` or `min_p = 0.05`.

---

## 6. What are the structural differences between encoder-only, decoder-only, and encoder-decoder models?

### Direct Answer
Encoder-only models use bidirectional attention to process full context simultaneously. Decoder-only models use causal masking to process tokens autoregressively from left to right. Encoder-decoder models split tasks into a bidirectional comprehension phase and a causal generation phase.

### System Mechanics & Trade-offs
| Architecture | Attention Type | Primary Use Case | Strengths | Weaknesses | Examples |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Encoder-Only** | Bidirectional | Embeddings, Classification | Full context visibility; fast parallel processing | Cannot perform open-ended generative tasks | BERT, RoBERTa |
| **Decoder-Only** | Causal (Masked) | Text generation, Chat, Coding | Excellent pretraining scale; flexible zero-shot prompting | Memory bandwidth bottlenecks during decode | Llama 3, GPT-4, Mistral |
| **Encoder-Decoder** | Bidirectional + Cross-Attention | Translation, Summarization | Strong alignment between source and target | Complex serving infrastructure; dual KV cache state | T5, BART |

- **Bidirectional Attention:** Every token attends to every token before and after it. This produces deep, contextual sentence embeddings.
- **Causal Attention:** An upper-triangular mask blocks attention to future positions. Token $T_i$ attends only to tokens $T_1 \dots T_{i-1}$. This mask ensures valid next-token prediction during autoregressive decoding.

### Production Rule of Thumb
Deploy decoder-only models for all general-purpose conversational, agentic, and coding workloads. Deploy encoder-only models for low-latency text classification, semantic search embeddings, and fast content moderation filters.

---

## 7. How do you calculate the memory footprint of model weights across different numerical precisions?

### Direct Answer
Calculate model weight memory using parameter count multiplied by bytes per parameter. Divide that value by $10^9$ to determine gigabytes. FP16 and BF16 use 2 bytes per parameter. INT8 uses 1 byte. INT4 uses 0.5 bytes.

### System Mechanics & Trade-offs
```
Memory Calculation Formula:
Memory (GB) = [ Parameter Count (Billions) x Bytes Per Parameter ] x 1.2 (Overhead Margin)

Data Types:
• FP32 (Full Precision):        4.0 bytes per parameter
• FP16 / BF16 (Half Precision): 2.0 bytes per parameter
• INT8 / FP8:                   1.0 byte per parameter
• INT4 (Quantized):             0.5 bytes per parameter
```

- **Worked Example (70 Billion Parameter Model):**
  - **BF16:** $70 \times 10^9 \times 2\text{ bytes} = 140\text{ GB}$. You need two 80 GB GPUs to load weights alone.
  - **INT8:** $70 \times 10^9 \times 1\text{ byte} = 70\text{ GB}$. Fits inside a single 80 GB GPU for weights.
  - **INT4:** $70 \times 10^9 \times 0.5\text{ bytes} = 35\text{ GB}$. Fits inside a single 48 GB enterprise GPU with room for KV cache.
- **Serving Overhead Factor:** Raw weights represent only part of memory consumption. Production systems must reserve 20% to 40% additional VRAM for the KV cache, CUDA runtime contexts, and activation buffers.

### Production Rule of Thumb
Always add a 1.25x safety multiplier to raw weight sizes when budgeting GPU hardware. If a quantized model requires 35 GB for weights, budget at least 44 GB of total VRAM to prevent out-of-memory errors under peak concurrent traffic.
