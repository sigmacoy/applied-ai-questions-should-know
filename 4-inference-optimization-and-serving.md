# 04. Inference Optimization & Serving

> Applied AI engineering questions on KV cache mechanics, PagedAttention, continuous batching, quantization (AWQ/FP8), latency metrics (TTFT vs. ITL), and speculative decoding.

---

## 1. What is the Key-Value (KV) Cache, and why is autoregressive decoding memory-bandwidth bound?

### Direct Answer
The Key-Value (KV) cache stores previously computed Key and Value attention tensors in GPU memory. It prevents recomputing past tokens during autoregressive generation. Autoregressive decoding processes one token per step. Streaming gigabytes of model weights from memory for a single token saturates memory bandwidth.

### System Mechanics & Trade-offs
```
Generation Step for Token N:
Without KV Cache: Recompute Q, K, V for all tokens 1 to N ──> Quadratic recomputation ($O(N^2)$)
With KV Cache:    Compute Q, K, V for Token N only ───────────> Concatenate with cached K, V tensors

Decoding Arithmetic Intensity:
Arithmetic Intensity = Total FLOPs / Total Bytes Accessed
• For 1 token: FLOPs ≈ 2 * Parameters. Bytes Accessed ≈ Model Weight Bytes + KV Cache Bytes.
• Intensity is under 1.0 FLOP/byte. A GPU capable of 300 TFLOPs sits idle waiting for memory transfer.
```

- **Memory Formula:** KV cache memory consumption per token is:
  $$\text{Bytes per Token} = 2 \times (\text{layers}) \times (\text{kv\_heads}) \times (\text{head\_dim}) \times (\text{precision\_bytes})$$
- **Capacity Impact:** In a 70B parameter model with 16-bit precision, a sequence length of 8,192 tokens consumes approximately 2.6 GB of VRAM per concurrent request.
- **The Memory Wall:** Tensor Cores finish matrix math in fractions of a microsecond. The bottleneck is the physical speed of the High Bandwidth Memory (HBM) bus feeding weights and KV tensors into compute cores.

### Production Rule of Thumb
Autoregressive decoding performance depends on GPU memory bandwidth, not raw TFLOPs. Choose GPUs with high HBM bandwidth (such as H100 or A100) over consumer gaming cards for multi-tenant serving workloads.

---

## 2. How does PagedAttention (vLLM) eliminate memory fragmentation in GPU VRAM?

### Direct Answer
Standard inference engines pre-allocate contiguous VRAM blocks for the maximum possible sequence length. PagedAttention divides the KV cache into fixed-size blocks stored in non-contiguous physical memory. It manages these blocks using a virtual memory page table, eliminating internal and external memory fragmentation.

### System Mechanics & Trade-offs
```
Traditional Contiguous Allocation (Massive Waste):
Request 1 (Max 2048 tokens): [ Allocated VRAM: 2048 tokens ...................... ]
                             [ Actual Usage: 150 tokens ][ Wasted Space: 85% ..... ]

PagedAttention Virtual Memory Model (Zero Waste):
Virtual Token Space:  [ Block 0 ] ──> [ Block 1 ] ──> [ Block 2 ]
                           │               │               │
Physical GPU Pages:   [ Page 42 ]     [ Page 12 ]     [ Page 99 ]  (Non-contiguous allocation)
```

- **Internal Fragmentation:** Legacy runtimes allocate space for maximum context (e.g., 4,096 tokens). If the request terminates after 200 tokens, the remaining 95% of allocated VRAM remains locked and unused.
- **External Fragmentation:** Varying request lifetimes fragment physical memory into gaps too small to host new contiguous allocations.
- **Copy-on-Write Branching:** PagedAttention allows parallel requests (such as beam search or multi-turn agent forks) to share the same physical pages until a token mutation occurs. This saves up to 50% additional memory.

### Production Rule of Thumb
Always use inference runtimes powered by PagedAttention (such as vLLM or TensorRT-LLM). PagedAttention increases serving concurrency and throughput by $2\times$ to $4\times$ on identical GPU hardware.

---

## 3. What is Continuous Batching (Iteration-Level Scheduling), and why does it outperform static batching?

### Direct Answer
Static batching locks a group of requests together until the slowest request finishes generating. Continuous batching operates at the iteration level. It ejects finished sequences immediately and inserts new incoming requests into the batch at every decoding step.

### System Mechanics & Trade-offs
```
Static Batching (GPU Waste):
Request A (50 tokens):  [████████]──────────────────── (GPU idle waiting for B)
Request B (200 tokens): [████████████████████████████] ──> Batch completes

Continuous Batching (Iteration-Level Scheduling):
Step t:       [ Req A (Token 1) ][ Req B (Token 1) ]
Step t+50:    [ Req A FINISHED  ][ Req B (Token 50) ]
Step t+51:    [ Req C (Token 1) ][ Req B (Token 51) ]  <-- Req C injected immediately!
```

- **Head-of-Line Blocking:** In static batching, if one request generates 1,000 tokens while three others generate 20 tokens, the three slots sit idle for 980 steps.
- **Dynamic Insertion:** Continuous batching treats every token iteration as an independent scheduling event. The runtime evicts completed requests and injects new prompts without waiting for an entire batch to complete.
- **Throughput Gains:** Continuous batching keeps GPU Tensor Cores continuously saturated. This yields a $3\times$ to $5\times$ improvement in request throughput over static batching.

### Production Rule of Thumb
Never deploy static batching for production LLM endpoints with variable prompt and output lengths. Configure continuous batching in your serving engine to maintain high hardware utilization under fluctuating user traffic.

---

## 4. What is the operational difference between Time to First Token (TTFT) and Inter-Token Latency (ITL)?

### Direct Answer
Time to First Token (TTFT) measures how long the user waits for generation to start. It covers network transit, batch queuing, and prompt prefill. Inter-Token Latency (ITL) measures the time elapsed between each successive generated token. TTFT is compute-bound, while ITL is memory-bandwidth bound.

### System Mechanics & Trade-offs
```
Inference Lifecycle Latency:
User Sends Prompt
      │
      ├── Queuing & Routing Delay
      ├── Prefill Stage (Process prompt tokens in parallel) ──> [ TTFT Complete ] ──> First Token Streamed
      │
      ├── Decode Step 1 ──> [ ITL 1 ]
      ├── Decode Step 2 ──> [ ITL 2 ]
      └── Decode Step N ──> [ ITL N ] ──> End-of-Sequence (EOS)
```

- **Prefill vs. Decode Profile:**
  - **Prefill (Determines TTFT):** Processes all prompt tokens simultaneously in a single forward pass. It achieves high arithmetic intensity and saturates GPU compute cores.
  - **Decode (Determines ITL):** Generates tokens sequentially one at a time. It achieves low arithmetic intensity and strains memory bandwidth.
- **Prefill Interference:** When a long prompt enters a running continuous batch, its heavy prefill calculation can stall active decoding iterations. This causes noticeable spikes in ITL (jitter) for existing streaming users.
- **Chunked Prefill:** Modern runtimes chop large prompt prefills into smaller chunks. The engine schedules prefill chunks alongside decode iterations to smooth ITL spikes.

### Production Rule of Thumb
For interactive chat and coding assistants, optimize for an ITL under 30 to 50 milliseconds per token to match natural human reading speeds. Optimize TTFT to remain under 800 milliseconds for responsive user experiences.

---

## 5. How do Post-Training Quantization techniques (AWQ, GPTQ, FP8) work, and how do they impact compute kernels?

### Direct Answer
Weight-only quantization (AWQ, GPTQ) compresses weights to 4-bit integers while computing activations in 16-bit floats. FP8 quantizes both weights and activations into 8-bit floating-point numbers. AWQ protects the most salient 1% of weights from degradation. FP8 accelerates raw matrix math directly on modern Tensor Cores.

### System Mechanics & Trade-offs
```
Quantization Approaches:

1. Weight-Only (INT4 / AWQ):
VRAM: Weights stored in INT4 ──> Dequantized on-the-fly to FP16 in SRAM ──> FP16 Tensor Cores
• Advantage: Cuts weight memory by 70%. Saves memory bandwidth.
• Drawback: Compute math remains FP16. Does not accelerate raw compute speed.

2. Weight & Activation (FP8):
VRAM: Weights stored in FP8 ──> Activations quantized to FP8 ──> Native FP8 Tensor Cores
• Advantage: Cuts memory by 50% AND doubles matrix math throughput (TFLOPs).
• Requirement: Requires modern GPU architectures (NVIDIA Ada Lovelace, Hopper, or Blackwell).
```

- **AWQ (Activation-aware Weight Quantization):** Observes which weight channels process the highest activation magnitudes during calibration. It scales and protects that critical 1% of weights, preventing perplexity loss at 4-bit precision.
- **GPTQ:** Uses second-order Taylor series approximations (Hessian matrices) to adjust remaining weights after quantizing each layer column.
- **FP8 Formats:** Uses E4M3 (4 exponent bits, 3 mantissa bits) for high-precision forward passes and E5M2 for broader dynamic range during gradient accumulation.

### Production Rule of Thumb
If serving on NVIDIA H100 or newer hardware, use native FP8 quantization for both weight reduction and compute acceleration. If serving on older hardware (such as A100 or A10G), use AWQ INT4 to minimize memory bandwidth bottlenecks.

---

## 6. What are Multi-Query Attention (MQA) and Grouped-Query Attention (GQA), and how do they reduce KV cache size?

### Direct Answer
Multi-Head Attention maintains separate Key and Value heads for every Query head. Multi-Query Attention forces all Query heads to share a single Key and Value head. Grouped-Query Attention groups Query heads together to share a reduced number of Key and Value heads, balancing memory savings with model accuracy.

### System Mechanics & Trade-offs
```
Multi-Head Attention (MHA):
Queries: [ Q1 ][ Q2 ][ Q3 ][ Q4 ][ Q5 ][ Q6 ][ Q7 ][ Q8 ]
Keys:    [ K1 ][ K2 ][ K3 ][ K4 ][ K5 ][ K6 ][ K7 ][ K8 ]  <-- 8 KV Heads (High VRAM)
Values:  [ V1 ][ V2 ][ V3 ][ V4 ][ V5 ][ V6 ][ V7 ][ V8 ]

Grouped-Query Attention (GQA - 2 Groups):
Queries: [ Q1   Q2   Q3   Q4 ]   [ Q5   Q6   Q7   Q8 ]
Keys:    [      K_G1         ]   [      K_G2         ]  <-- 2 KV Heads (75% VRAM Reduction!)
Values:  [      V_G1         ]   [      V_G2         ]

Multi-Query Attention (MQA):
Queries: [ Q1   Q2   Q3   Q4      Q5   Q6   Q7   Q8 ]
Keys:    [                    K                      ]  <-- 1 KV Head (87.5% VRAM Reduction)
Values:  [                    V                      ]
```

- **The MHA Memory Problem:** In a model with 32 query heads, MHA stores 32 key vectors and 32 value vectors per token. This limits concurrent batch sizes and exhausts GPU memory on long contexts.
- **MQA Degradation:** MQA reduces the KV cache size by a factor of 32. However, forcing all heads to attend to identical key projections harms nuanced multi-step reasoning.
- **GQA Sweet Spot:** GQA (used by Llama 3 and Mistral) achieves 99% of MHA reasoning performance while cutting KV cache memory consumption by a factor of 4 to 8.

### Production Rule of Thumb
Select base model architectures built with Grouped-Query Attention (GQA). GQA allows you to run 4x to 8x larger concurrent batch sizes within identical GPU VRAM constraints compared to legacy MHA models.

---

## 7. What is Speculative Decoding, and how does it accelerate generation without losing accuracy?

### Direct Answer
Speculative decoding uses a small, fast draft model to generate candidate tokens quickly. A large target model evaluates all candidate tokens simultaneously in a single forward pass. It accepts matching tokens and corrects discrepancies, accelerating generation without altering output probability distributions.

### System Mechanics & Trade-offs
```
Draft Phase (Fast Memory-Bandwidth Bound Steps):
Small Draft Model (e.g., Llama-3-8B) generates K tokens sequentially (e.g., K = 4):
Tokens: [ "The", "capital", "of", "France" ]

Verification Phase (Single Compute-Bound Step):
Large Target Model (e.g., Llama-3-70B) runs 1 forward pass across all 4 tokens in parallel:
Evaluates: [ "The" -> OK ][ "capital" -> OK ][ "of" -> OK ][ "France" -> REJECTED, corrected to "Spain" ]

Output: 3 accepted tokens + 1 corrected token in the time of a single 70B forward pass!
```

- **Mathematical Exactness:** Speculative decoding uses modified rejection sampling. The final generated text matches the mathematical output distribution of the large target model with 100% fidelity.
- **Acceptance Rate Factor:** Speedup depends on the acceptance rate of draft tokens. Code generation and boilerplate language achieve acceptance rates above 80%. Creative prose and high-entropy tasks achieve lower acceptance rates.
- **Overhead Trap:** If the draft model acceptance rate falls below 40%, the system expends extra GPU compute without generating tokens faster.

### Production Rule of Thumb
Deploy speculative decoding for code generation, structured extraction, and repetitive data tasks. Ensure the draft model shares the exact tokenizer vocabulary as the target model to eliminate token translation overhead.
