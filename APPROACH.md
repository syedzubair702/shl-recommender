# SHL Assessment Recommender — Approach Document

**Author:** AI Intern Candidate  
**Date:** May 2026

---

## Design Overview

### Problem Decomposition

The core challenge is converting vague hiring intent ("I need to hire a developer") into a grounded shortlist from a fixed catalog. This requires four distinct capabilities: clarification, recommendation, refinement, and comparison — each with different retrieval and generation strategies.

### Architecture

```
POST /chat
  │
  ├─ Input validation (Pydantic, schema guard)
  ├─ Injection/OOS detection (regex guardrails)
  ├─ Catalog retrieval (FAISS semantic search → top-15 candidates)
  │    ↓ injected into system prompt as context
  ├─ Claude (claude-sonnet-4-20250514) → JSON response
  ├─ Output validation (URL whitelist, test_type enum, rec count cap)
  └─ ChatResponse (reply, recommendations[], end_of_conversation)
```

**Stateless by design.** Every `/chat` call carries the full conversation history. The service holds no session state. FAISS index and catalog are loaded once at startup and cached in memory.

---

## Key Design Decisions

### 1. Retrieval Strategy: Semantic Search + Context Injection

**Why FAISS + sentence-transformers instead of keyword search:**  
Keyword search fails when users say "someone who works with numbers" rather than "numerical reasoning." Semantic search with `all-MiniLM-L6-v2` (384-dim, ~80ms) handles natural language intent well at low latency.

**Context injection not RAG:** Rather than a traditional RAG pipeline, I inject the top-15 catalog matches directly into the system prompt as structured text. This keeps the LLM grounded and avoids hallucination while staying within the 30-second timeout.

**Enriched documents:** Each catalog item is embedded with its name, description, test types, duration, plus role-keyword boosters ("java → software developer programming") to improve retrieval recall for common hiring scenarios.

### 2. Agent Behavior via Prompt Engineering

The system prompt encodes all four behaviors using explicit rules and the required JSON schema. The LLM decides when to clarify vs. recommend based on context richness. Key constraints:

- **Clarify first:** The prompt instructs the agent to ask ONE question when context is insufficient
- **No hallucination:** "Only use URLs from the catalog I provide" + post-LLM URL whitelist validation
- **Refinement:** Conversation history in every request means the LLM sees prior shortlists and can update them

### 3. Defense in Depth for Catalog Integrity

The LLM receives catalog URLs in its context. But LLMs can still confabulate. I add a hard post-processing layer:
1. Every returned URL is checked against the valid URL set
2. If URL is invalid but name matches, we look up the correct URL from catalog
3. If neither matches, the recommendation is dropped silently
4. `test_type` values are validated against the enum `{A,B,C,D,E,K,M,P,S}`

### 4. Timeout Management

The evaluator enforces 30-second call timeouts. Design choices to stay under:
- FAISS retrieval: ~50ms
- Model encoding for query: ~80ms
- Claude API call: `httpx` with 25s timeout (leaves 5s headroom)
- `max_tokens=1024` prevents runaway generation

### 5. Turn Cap Enforcement

The evaluator caps at 8 turns. Messages beyond 8 are silently truncated to the last 8 before sending to the LLM. This prevents context overflow and ensures compliance.

---

## Retrieval Setup

- **Model:** `all-MiniLM-L6-v2` (sentence-transformers) — fast, good quality, free
- **Index:** FAISS `IndexFlatIP` with L2-normalized embeddings (cosine similarity)
- **Catalog:** ~60+ Individual Test Solutions scraped from `shl.com/solutions/products/product-catalog/` with fallback to a curated hardcoded catalog if scraping fails
- **Query enrichment:** Last 4 turns of conversation concatenated with the latest user message for retrieval

---

## Evaluation Approach

### What I Tested

1. **Schema compliance:** Every response field validated with Pydantic models
2. **URL whitelist:** Post-hoc validation ensures zero hallucinated URLs
3. **Behavior probes (automated):**
   - Vague query → no recs on turn 1
   - Off-topic → refusal with redirect
   - Prompt injection → refusal
   - Legal/salary → refusal
4. **Recall@10:** Manual eval set of 5 query-shortlist pairs aligned to common hiring personas
5. **Conversation flow tests:** clarify→recommend, JD input, refinement, comparison

### What Didn't Work Initially

- **Embedding only the name:** Low recall for domain-specific queries ("cognitive ability" didn't retrieve "Verify G+"). Fixed by adding description and role-keyword boosters to embedded documents.
- **Single-turn retrieval:** Using only the latest message lost context from earlier turns. Fixed by concatenating last 4 messages for query enrichment.
- **Trusting LLM URLs without validation:** Early testing showed occasional URL mutations. Fixed by the whitelist layer.

---

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| API Framework | FastAPI | Async, Pydantic validation, fast |
| LLM | Claude Sonnet 4 (Anthropic) | Instruction following, JSON output |
| Embedding | `all-MiniLM-L6-v2` | Free, fast, good semantic quality |
| Vector store | FAISS-cpu | No server needed, loads in-process |
| Deployment | Render / Docker | Free tier, cold-start < 2min |

**AI tools used:** Claude assisted with boilerplate generation and test case ideation. All design decisions, prompt engineering, and architecture were human-authored and can be defended.
