# 06. Agentic Patterns & Tool Use

> Applied AI engineering questions on function calling mechanics, structured JSON grammars, ReAct loops, state machines, supervisor-worker hierarchies, and human-in-the-loop workflows.

---

## 1. What is Function Calling (Tool Use) at the model mechanics level?

### Direct Answer
Language models cannot execute external code. The runtime converts tool schemas into text and injects them into the system prompt. When the model determines a tool is needed, it emits structured text containing the function name and arguments. The host environment executes the function and injects the output back into the context.

### System Mechanics & Trade-offs
```
Execution Lifecycle of a Tool Call:
1. Registration: System prompt formats JSON Schema of available tools.
2. Inference:    Model detects need for external data; generates structured text:
                 `call:get_weather{"location": "Tokyo"}`
3. Interception: Host application stops streaming, parses function arguments, and calls local API.
4. Execution:    Local API returns string result: `{"temp": "18C", "status": "Rain"}`.
5. Continuation: Result injected as a role="tool" message; model resumes autoregressive generation.
```

- **Prompt Injection Mechanics:** Tool definitions consume prompt context tokens. Providing 20 detailed tool schemas can consume 2,000 to 4,000 prompt tokens on every request turn.
- **Model Role:** The model functions solely as a semantic router and argument extractor. It never touches operating system sockets or databases directly.
- **Error Handling Loop:** If the model outputs malformed arguments, the host environment must format the error message and pass it back as a tool result. This lets the model self-correct on the next iteration.

### Production Rule of Thumb
Keep your active tool registry minimal. Expose no more than 3 to 5 focused tools per agent state to reduce token consumption and eliminate tool selection confusion.

---

## 2. How do Constrained Decoding and Structured Grammars (CFG / JSON Schema) guarantee valid outputs?

### Direct Answer
Standard sampling can produce invalid JSON, unescaped characters, or broken schemas. Constrained decoding compiles a JSON Schema or Context-Free Grammar into a Finite State Machine. At each decoding step, the FSM masks invalid tokens from the logit distribution, guaranteeing 100% syntactically valid outputs.

### System Mechanics & Trade-offs
```
Standard Autoregressive Sampling (Fragile):
Model Logits ──> Sample Next Token ──> Might pick syntax-breaking token: `{"name": "Alice", age: 30}` (BROKEN!)

Constrained Decoding with Grammar FSM (Guaranteed):
Current FSM State: Expecting JSON Key
Allowed Tokens:    `"name"`, `"age"`, `"email"`
Masking Action:    Set logits of ALL other vocabulary tokens to -∞.
Sample Next Token: Guaranteed valid transition!
```

- **Grammar Compilation:** Tools like Outlines, SGLang, and llama.cpp convert JSON schemas or EBNF grammars into deterministic finite state automata before generation begins.
- **Zero Hallucination of Syntax:** Constrained decoding eliminates JSON parsing failures and schema mismatch errors entirely.
- **Logit Masking Overhead:** Evaluating grammar state transitions across a 128,000-token vocabulary can introduce minor CPU latency if transition tables are not pre-indexed.

### Production Rule of Thumb
Never use standard unconstrained prompt instructions (such as *"Return only valid JSON"*) for critical production APIs. Enforce structured JSON schemas directly at the inference engine layer using constrained decoding.

---

## 3. What is the ReAct (Reasoning + Acting) pattern, and how does it differ from single-step prompt chains?

### Direct Answer
ReAct interleaves reasoning traces with action executions in an iterative loop. The agent reasons about its current state, executes a tool, observes the environment response, and repeats the cycle. This dynamic feedback loop allows models to correct errors and adjust strategies in real time.

### System Mechanics & Trade-offs
```
ReAct Iterative Cycle:
┌────────────────────────────────────────────────────────┐
│ 1. Thought:     "I need to find the user's order ID."  │
│ 2. Action:      Call lookup_order(email="user@test")   │
│ 3. Observation: Returns "Order #9921 - Shipped"        │
│ 4. Thought:     "Now I must check tracking details."   │
│ 5. Action:      Call get_tracking(order_id="9921")     │
│ 6. Observation: Returns "Arriving Friday"              │
│ 7. Final Ans:   "Your order #9921 arrives on Friday."  │
└────────────────────────────────────────────────────────┘
```

- **Dynamic Correction:** In a fixed prompt chain, an intermediate failure breaks the entire sequence. In a ReAct loop, if an API call returns a 404, the agent observes the error and tries an alternative lookup tool.
- **Compounding Token Explosion:** Each turn appends previous Thoughts, Actions, and Observations to the conversation history. Long loops consume context window capacity quickly.
- **Infinite Loop Risk:** Without strict termination logic, agents can enter repetitive cycles where they call the same failing tool indefinitely.

### Production Rule of Thumb
Always configure strict iteration limits on ReAct loops (typically a maximum of 5 iterations). If an agent fails to achieve its goal within 5 turns, terminate execution and fall back to human intervention.

---

## 4. Why are Finite State Machines (FSMs) and Directed Acyclic Graphs (DAGs) preferred over open-ended autonomous loops?

### Direct Answer
Open-ended autonomous loops suffer from compounding error rates and unpredictable execution paths. FSMs and DAGs constrain agent behavior to explicit states and pre-approved transitions. This architecture guarantees reliability, auditability, and deterministic business logic in production.

### System Mechanics & Trade-offs
```
Open-Ended Autonomous Loop (Unpredictable):
State ──> [ LLM Decides Everything ] ──> Infinite possible paths (High failure rate: 0.95^N)

State Graph / DAG Architecture (Production-Grade):
[ Intake Node ] ──(Deterministic Router)──┬──> [ Billing Node ] ──> [ Human Approval ]
                                          │
                                          └──> [ Tech Support ] ──> [ Resolution ]
```

- **Compounding Error Problem:** If each LLM step has a 95% success rate, a 10-step autonomous execution has only a 60% overall success rate ($0.95^{10} \approx 0.60$).
- **Deterministic Guardrails:** In an FSM architecture (such as LangGraph or Temporal workflows), code defines valid paths. The LLM operates only within the boundaries of its current node.
- **Auditing and Rollback:** Workflows built on state machines allow systems to persist checkpoints, log exact state transitions, and replay failed runs safely.

### Production Rule of Thumb
Do not deploy fully autonomous, open-ended agent loops for mission-critical enterprise workflows. Structure complex processes as deterministic state graphs where LLMs handle reasoning inside tightly bounded nodes.

---

## 5. How do you handle context window compaction and state management in long-running agent workflows?

### Direct Answer
Long-running agent workflows exceed context limits and degrade reasoning. Manage state by separating persistent data into an external structured state object. Compact the conversation history using sliding windows, message summarization, or ephemeral subagent contexts.

### System Mechanics & Trade-offs
```
State Management Strategies:

1. External Structured State (Scratchpad):
{
  "customer_id": "C-1092",
  "verified": true,
  "issue": "damaged_goods",
  "refund_approved": false
}
• State lives in application database. LLM reads and updates exact fields.

2. Context Compaction Pipeline:
Full History (50 turns) ──> [ Background Summarizer ] ──> Compacted Summary (2 turns)
                                                                 │
                                                                 ▼
[ System Prompt ] + [ External State ] + [ Compacted Summary ] + [ Last 3 Turns ]
```

- **Context Saturation:** When prompt contexts exceed 30,000 tokens, models suffer from distraction and higher latency. Prompt costs also scale linearly with every added token.
- **Scratchpad Strategy:** Maintain application state outside the LLM context window. Update database fields explicitly instead of relying on the model to remember facts across 20 conversation turns.
- **Ephemeral Subagents:** Delegate complex research or subtasks to temporary worker agents. Discard the worker conversation history and return only a concise answer string to the primary agent.

### Production Rule of Thumb
Never maintain raw conversation histories beyond 10 turns in automated agent loops. Compress historical messages into an external JSON state object, and pass only the current state plus the latest 3 turns to the model.

---

## 6. What is the Plan-and-Solve (Supervisor-Worker) architecture, and how does it prevent cascading errors?

### Direct Answer
The Plan-and-Solve architecture separates planning from execution. A supervisor model decomposes a complex objective into discrete subtasks. Specialized worker agents execute individual subtasks in isolated contexts. This isolation prevents tool output errors from corrupting the primary reasoning chain.

### System Mechanics & Trade-offs
```
                         ┌───────────────────────┐
                         │   Supervisor / Plan   │
                         └──────────┬────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  Worker 1 (SQL Query)│ │ Worker 2 (Web Scrape)│ │ Worker 3 (Code Exec) │
└──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │ Aggregated Summaries
                                    ▼
                         ┌───────────────────────┐
                         │ Final Synthesis Node  │
                         └───────────────────────┘
```

- **Context Isolation:** A worker executing database queries might generate hundreds of lines of raw SQL errors before succeeding. Isolating the worker prevents these error logs from polluting the supervisor context.
- **Specialized Tool Sets:** Workers receive only the specific tools required for their subtasks. This eliminates tool selection errors and reduces prompt token overhead.
- **Parallel Execution:** When the supervisor generates independent plan steps, the system can execute multiple worker tasks concurrently across separate worker threads.

### Production Rule of Thumb
Decompose tasks requiring three or more distinct tool types into a Supervisor-Worker pattern. Isolate tool execution logs inside ephemeral worker contexts to preserve supervisor context cleanliness.

---

## 7. What is the Human-in-the-Loop (HITL) pattern, and how do you design safe interrupt-and-resume mechanisms?

### Direct Answer
The Human-in-the-Loop pattern halts agent execution before invoking high-impact or irreversible tool operations. The system persists its current execution state to a database and waits for human review. Once approved or edited by an operator, the workflow resumes execution from the persisted state.

### System Mechanics & Trade-offs
```
Interrupt-and-Resume Architecture:
[ Agent Reasoning ] ──> Detects High-Impact Tool Call (e.g., execute_wire_transfer)
                             │
                             ▼
                [ Persist Workflow State to DB ]
                             │
                             ▼
                [ Suspend Execution & Send Webhook ] ──> Alert Operator Dashboard
                                                                │
                                                                ▼
                                                    [ Human Operator Action ]
                                                    • Approve / Reject / Edit
                                                                │
                                                                ▼
                [ Restore Workflow State from DB ] ◄────────────┘
                             │
                             ▼
                [ Execute Action & Resume Agent Loop ]
```

- **Classification of Impact:** Read-only operations (such as document lookups or calculations) execute autonomously. Mutating actions (such as database deletes, payments, or customer emails) require explicit human checkpoints.
- **Stateless Durability:** System nodes cannot keep memory threads alive while waiting hours for human approval. The orchestrator must serialize the execution graph into durable storage (such as PostgreSQL or Redis).
- **Modification on Resume:** Human operators must have the ability to inspect the proposed tool arguments, edit values if necessary, or reject the action with feedback explaining the refusal.

### Production Rule of Thumb
Enforce mandatory human authorization checkpoints for any tool that mutates persistent data or triggers financial transactions. Never grant an LLM autonomous authorization to execute destructive database updates or initiate payments.
