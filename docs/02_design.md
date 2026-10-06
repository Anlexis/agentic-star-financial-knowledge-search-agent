# Template Design Specification — FIN-C2-005

**Template ID:** FIN-C2-005
**Template Name:** FinancialKnowledgeBaseSearchAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** FIN

## Position in AgentCore Architecture

- **Agent Class:** `FinancialKnowledgeBaseSearchAgent` (alias `Graph`)
- **L1 Base (framework base class):** `AgentBaseGraph` — direct framework inheritance
- **Inner graph base:** `BaseGraph` — `DomainWorkflowGraph`
- **Pattern:** Cat 2 two-layer nested architecture (outer fixed 5-node backbone +
  `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow)
- **Three-Layer Separation:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`;
    retrieval tuning knobs are plain scalars (see the config-forwarding note below)
  - Node: framework inheritance via `FunctionNode` (override
    `execute(self, state: AgentState) -> dict` ONLY — no `config` parameter)
  - Graph: composition (`register_nodes()` for node substitution); outer
    `add_edges()` is NOT overridden

## Purpose

Financial knowledge-base search agent: compliance, regulatory and product
questions are answered over a financial knowledge base (customer verification
duties, sales-conduct rules, disclosure obligations, record retention) —
retrieve → rerank/filter → grounded answer with citations + a standing advisory
disclaimer. The current build is fully deterministic (keyword retrieval +
rule-based grounded answer assembly; no live model call — see the
Implementation Note below).

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max_retry from config/config.yaml)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) |
| pre_process | `PreProcessNode` | **Trust gate (VERIFIED_EXTERNAL)** + question validation, instruction-override refusal, direct-identifier redaction, and the `input_context` caller-data contract → `validated_input`, `enriched_context` | `TrustLevel.VERIFIED_EXTERNAL` |
| main | `KnowledgeBaseSearchGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; bridges `input_context`; maps inner `formatted_answer` → outer `result` | — (GraphNode delegation) |
| post_process | `PostProcessNode` | **Output gate** (credential withhold + caller-text redaction + identifier redaction + disclaimer invariant) → `formatted_output`; on a correctable rejection, renders the fixed reason sentence instead (see *Reason codes* below) | `TrustLevel.VERIFIED_EXTERNAL` |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer pre_process slot; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|--------------|
| `InputValidateNode` | Re-validate the caller parameters (`category`, `top_k`) with the ingest-boundary rules, re-screen for instruction-override content, normalise and cap the query text | `TrustLevel.ANONYMOUS` | `validated_input` \| `user_input`, `input_context` | `search_query`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic keyword retrieval over the bundled knowledge base (`config/kb/financial_kb.json`): tokenise query, score title/tags/content overlap, apply the category filter | `TrustLevel.ANONYMOUS` | `search_query`, `query_filters`, `retrieval_top_k`, `retrieval_kb_path` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | Rerank candidates (category-match boost), drop entries below `score_threshold`, cap at `top_k` | `TrustLevel.ANONYMOUS` | `retrieved_documents`, `query_filters`, `retrieval_top_k`, `retrieval_score_threshold` | `ranked_documents` |
| `GenerateAnswerNode` | Rule-based grounded answer assembly from the ranked passages only, with numbered citation markers; the caller's question is never rendered into the answer | `TrustLevel.ANONYMOUS` | `ranked_documents` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final answer: body + Sources list + the standing advisory disclaimer (the disclaimer belongs to this node, NOT post_process) | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data Flow

```
user_input + input_context
  → PreProcessNode                                  → validated_input, enriched_context
  → KnowledgeBaseSearchGraphNode.extract_input      → inner DomainWorkflowGraph.invoke(validated_input)
        (stashes input_context in the context bridge)
        → input_validate                            → search_query / query_filters
        → retrieve                                  → retrieved_documents
        → rerank_filter                             → ranked_documents
        → generate_answer                           → grounded_answer / citations
        → output_format                             → formatted_answer (+ advisory disclaimer)
     get_output() → {formatted_answer, citations, status, error_code, ...}
  → KnowledgeBaseSearchGraphNode.merge_output       → result = formatted_answer, knowledge_base_answer
  → PostProcessNode (output gate)                   → formatted_output (gated)
```

## Caller-data contract (`input_context`)

`/invoke` accepts the SDK's first-class `input_context` mapping alongside the
question. Every declared field is bounds-checked at the ingest boundary
(`PreProcessNode`) and re-validated inside the domain workflow
(`InputValidateNode`), so a direct inner-graph invocation gets the same rule.

| Field | Type / bound | Absent behaviour |
|-------|--------------|------------------|
| `channel` | `^[a-z0-9_]{1,32}$` | `"unknown"` |
| `category` | `^[a-z0-9_]{1,32}$` | no category filter |
| `top_k` | integer, 1–20 | the configured retrieval depth |

Rules that hold for every field:

- **Fail CLOSED.** An out-of-contract value stops the run where it is found:
  no retrieval runs and no answer is assembled. The run COMPLETES
  (`status=success`) carrying the reason code `INVALID_REQUEST` in
  `error_code`, and `PostProcessNode` renders the fixed sentence for that code
  as the caller-facing body — so the caller can correct the value and send the
  request again on the same conversation instead of receiving only an
  exception type. The FIELD name stays in `error_log`, the internal audit
  channel — never the rejected value, and never the field name in the
  caller-facing body (caller data must not round-trip into error logs or the
  answer). See *Reason codes for a correctable rejection* below.
- **Inert alphabets for strings.** `channel` and `category` both influence the
  run, so free text is rejected outright. This also keeps the context channel
  free of caller prose: no free text can arrive by a route the question text
  does not take.
- **Strict integer check for `top_k`.** Booleans, floats, numeric strings and
  the non-finite JSON values (`NaN`, `Infinity`, `-Infinity`) all fail the
  `isinstance` rule before the range bound is consulted — a NaN would
  otherwise compare `False` against every bound and pass silently.
- **Caller `top_k` narrows, never widens.** A caller override is applied only
  when it is stricter than the configured depth.
- Undeclared keys are ignored; the HTTP adapter separately caps the serialized
  `input_context` at 256 KB.

The question text is separately capped at 2000 characters, stripped of control
characters, refused when it carries instruction-override content, and screened
for direct identifiers (IBAN, long account numbers, e-mail) before anything is
written to State.

### Reason codes for a correctable rejection

Two kinds of rejection exist, and they end the run differently.

**Correctable by the caller — the run COMPLETES.** The request is not carried
out: no retrieval runs, no answer is assembled, and the same audit events are
emitted as before. What changes is only how the outcome is reported — the run
finishes with `status=success` and a reason code in `error_code`, and the
answer body is the fixed sentence for that code, so the caller can correct the
request and send it again on the same conversation instead of seeing only an
exception type with the reason reachable from the audit trail alone.

| `error_code` | Set by | Condition |
|--------------|--------|-----------|
| `EMPTY_INPUT` | `PreProcessNode` | question empty, missing, or not a string |
| `QUESTION_TOO_LONG` | `PreProcessNode` | question over 2000 characters |
| `INVALID_REQUEST` | `PreProcessNode` | `input_context` is not a mapping, or `channel` / `category` / `top_k` is out of contract |
| `INVALID_REQUEST` | `InputValidateNode` | the inner re-validation of `category` / `top_k` (reached on a direct inner-graph invocation) |

**Not correctable by rewording — the run TERMINATES with `status=error`.**
Instruction-override content in the question (`PreProcessNode`,
`InputValidateNode`) and a credential pattern found in the assembled answer by
the output gate (`PostProcessNode`) both end the run as an error. These are the
template's own refusals, not a caller-fixable value, so they are not softened
into a completed run.

The reason codes are internal routing values, not a caller-facing vocabulary:
`error_code` is a State field and is NOT part of the `/invoke` response
envelope. The caller reads the reason as the answer body only — a fixed
sentence from `src/services/failure_message.py` that names WHAT to correct and
nothing else (never the rejected value, never a field path, never a gate's
wording; those stay in `error_log`).

**Marker propagation.** Once `error_code` is set, every node after it returns
immediately without doing work — otherwise the pipeline would keep running
past the rejection and reach retrieval. `KnowledgeBaseSearchGraphNode.execute()`
skips the inner graph entirely; the inner `get_output()` carries `error_code`
across the graph boundary, and `merge_output()` prefers a reason settled in the
outer graph over one from the inner run, so a specific reason is never
overwritten by a vaguer one for the same rejection. `PostProcessNode` renders
the sentence and writes no structured answer field (`result`,
`knowledge_base_answer`, `citations`), so a run that did not carry out the
request cannot return something shaped like a result.

### Context bridge (outer → inner)

`GraphNode.execute()` does not forward the outer state's `input_context` into
`subgraph.invoke()`, so inner-node reads of `state["input_context"]` would
always see `{}`. `src/graph/context_bridge.py` closes the gap with a
`ContextVar`: `KnowledgeBaseSearchGraphNode.extract_input()` stashes the value
just before the inner invoke, and `DomainWorkflowGraph._extra_initial_state()`
reads it back inside. A `ContextVar` keeps the hand-off correct per
thread/task, so concurrent invocations in one process cannot see each other's
context. The end-to-end proof (a caller `top_k` override changing the number of
cited sources in the final answer) is in `tests/unit/test_caller_data_and_output_schema.py`.

## Configuration

Two files, with different jobs:

| File | Contents | Read by |
|------|----------|---------|
| `config/agent.yaml` | Flat registration manifest — every key at ROOT level (`id`, `name`, `namespace`, `class`, `required_trust_level`, `requires.secrets` / `requires.extras`) | the platform registry, at discovery |
| `config/config.yaml` | Runtime parameters — `max_retry`, `timeout_s`, the `retrieval` block (`top_k`, `score_threshold`, `kb_path`) and the `llm` block | the registry (passed as `Graph(config=...)`) and `_runtime_config()` in the standalone server |

`requires.secrets` and `requires.extras` are both empty because this build
calls `ctx.secrets.require()` nowhere and constructs no model client. Declaring
either would fail the compile-time provisioning check.

`KnowledgeBaseSearchGraphNode._parent_config()` reads `config/config.yaml` and
forwards the `retrieval` + `llm` blocks under `config["configurable"]`, after
validating every numeric for type, finiteness and range — a malformed
configuration file can neither crash graph construction nor weaken the
relevance floor. Invalid or absent keys are simply not forwarded and the
consuming node falls back to its module default.
`DomainWorkflowGraph._extra_initial_state()` then republishes the `retrieval`
block into inner state as the SCALAR fields `retrieval_top_k` /
`retrieval_score_threshold` / `retrieval_kb_path` — the node contract is
`execute(self, state)` with no `config` parameter, so tuning knobs travel
through State. Consumers re-parse them through a finite+bounded check, so a
corrupted (for example checkpoint-restored) scalar cannot reach a comparison.

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `NotRequired[str]` | normalised, identifier-redacted question | outer |
| `knowledge_base_answer` | `NotRequired[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `search_query` | `NotRequired[str]` | normalised search query | inner |
| `query_filters` | `NotRequired[Optional[str]]` (JSON) | validated caller parameters (`category`, `top_k`) | inner |
| `retrieval_top_k` | `NotRequired[int]` | declared retrieval depth (scalar) | inner |
| `retrieval_score_threshold` | `NotRequired[float]` | declared relevance floor (scalar) | inner |
| `retrieval_kb_path` | `NotRequired[str]` | declared knowledge-base path (scalar) | inner |
| `retrieved_documents` | `NotRequired[Optional[str]]` (JSON) | scored candidates | inner |
| `ranked_documents` | `NotRequired[Optional[str]]` (JSON) | reranked + threshold-filtered passages | inner |
| `grounded_answer` | `NotRequired[str]` | rule-assembled grounded answer body | inner |
| `citations` | `NotRequired[Optional[str]]` (JSON) | `[{ref, id, title, source}]` | inner |
| `formatted_answer` | `NotRequired[str]` | final answer + sources + advisory disclaimer | inner |
| `intake_notes` | `NotRequired[Optional[str]]` (JSON) | validation / parse notes (no caller values) | inner |
| `error_code` | `Optional[str]` | reason code for a correctable rejection; makes every later node pass through untouched | both |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer (msgpack
  checkpoint safety). Tuning scalars stay scalars.
- Domain fields are `NotRequired[...]` (valid TypedDict before any node writes).
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No credentials, API keys, or raw personal identifiers in State.
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic models / dataclasses / arbitrary Python objects.

## Security design

- **Trust gate.** Every node declares `required_trust_level` (see the tables
  above). `PreProcessNode` and `PostProcessNode` require VERIFIED_EXTERNAL, so
  an anonymous caller is denied before the workflow runs and can never receive
  an answer. The standalone server elevates authenticated Bearer callers to
  VERIFIED_EXTERNAL (`INVOKE_AUTH_TOKEN`); middleware-established trust is
  never demoted.
- **Instruction-override refusal is the template's own guarantee.** The
  platform input gate refuses instruction-override payloads too, but this
  template does not depend on it: `PreProcessNode` and `InputValidateNode`
  each refuse independently, so the payload is rejected wherever that gate is
  absent or configured off. The pattern set is deliberately narrow —
  a genuine question that merely contains the words is unaffected — and the
  tests assert BEHAVIOUR (error status, no query carried forward, nothing
  published), never any gate's wording. One test calls `execute()` directly so
  the refusal is proven with nothing in front of it.
- **Direct-identifier redaction runs at both boundaries.** IBAN, long account
  number and e-mail patterns are redacted from the question before anything is
  written to State (`PreProcessNode`) and again from the assembled answer
  (`PostProcessNode`), so an identifier that entered from a replaced
  knowledge base cannot reach the external surface either. Both boundaries
  share one pattern set (`src/nodes/screens.py`).
- **Untrusted numerics are finite and bounded.** Every caller-controlled and
  every serialized numeric — `top_k`, `score_threshold`, candidate scores,
  the forwarded configuration values — passes through `_finite_in_range`
  before it reaches a comparison. `float("NaN")` parses fine and compares
  `False` against everything, which would silently turn the relevance floor
  into a no-op.
- **Audit logging.** Every node's `execute()` emits domain-specific
  `emit_trace_event(...)` events on its decision paths; the gate emits one per
  redaction class. Payloads carry counts and field NAMES, never rejected
  values. Nodes do NOT emit `node_start` / `node_complete` / `node_error` —
  `BaseNode.__call__()` emits those. Domain event names:
  `pre_process_complete`, `pre_process_validation_failed`,
  `input_validate_complete`, `input_validate_failed`, `retrieve_complete`,
  `rerank_filter_complete`, `generate_answer_complete`,
  `output_format_complete`, `post_process_complete`,
  `post_process_degraded`, `post_process_credential_violation`,
  `post_process_blocked_field_redaction`,
  `post_process_identifier_redaction`, `post_process_disclaimer_reattached`.
- Domain nodes never override `_security_gate_input` / `_security_gate_output`
  (the framework marks both final); the domain output gate is a module-level
  function called from `execute()`.

## External output schema

The answer this template emits is defined by one invariant, rendered by
`OutputFormatNode` and independently ENFORCED by `PostProcessNode`:

> An emitted answer contains knowledge-base-derived content plus the fixed
> template structure, ALWAYS carries the advisory disclaimer, and NEVER
> carries caller text, direct customer identifiers, or credentials.

The gate enforces it for every ASSEMBLED representation the template can emit —
the grounded answer, the insufficient-coverage answer, and the empty-upstream
fallback — in four layers:

1. **Credential scan → withhold.** API keys, JWTs, Bearer tokens and credential
   assignments anywhere in the assembled answer replace the whole output with a
   withheld stub and return `status=error`.
2. **Verbatim caller-text redaction.** Substantial verbatim embeddings of
   `user_input` / `validated_input` / `search_query` are replaced with
   `[REDACTED]`. The answer is built from retrieved passages only, so such an
   embedding is a leak, not a feature — and it is how a request would smuggle
   fake `[n]` markers into a document whose value is that every line traces to
   a source.
3. **Direct-identifier redaction.** The ingest-boundary pattern set is applied
   again to the assembled answer.
4. **Disclaimer invariant.** The advisory disclaimer is re-asserted; a
   representation that lost it gets it back rather than reaching a reader
   unqualified.

**Layer order is part of the design.** The credential PATTERN scan runs on
untouched text. The verbatim caller-text redaction — which removes whole
strings — runs BEFORE the identifier pattern scan, because rewriting an
identifier *inside* a caller-text embedding would break the exact-substring
match and leave the rest of that caller text on the surface. After the
rewriting layers the credential scan is RE-RUN, so a pattern that only emerges
once text has been removed is still caught.

**The correctable-rejection body is outside these four layers, deliberately.**
It is not an assembled answer: the reason sentence is a fixed module constant,
built from no retrieved passage and no caller text, and it is returned before
the gate layers run. There is nothing for a redaction layer to act on, and the
disclaimer invariant does not apply because no knowledge-base content is being
quoted. Only the assembled representations — which do carry corpus content —
reach the gate.

**No numeric rounding grid applies here, deliberately.** This template computes
no monetary aggregates — it quotes regulatory passages. A gate that snapped
numbers onto a rounding grid would rewrite a quoted threshold, a retention
period or a section number, falsifying the very content a reader is told to
verify against the primary text. Numbers, section numbers, dates and domain
identifiers therefore pass the boundary byte-identical, and
`tests/unit/test_caller_data_and_output_schema.py` pins that in both
directions: leak forms are redacted, structural and quoted tokens are
unchanged.

## Advisory Disclaimer

Every answer carries the standing advisory line (informational only, not
financial/legal/investment advice, verify against primary regulatory text).
`OutputFormatNode` renders it and `PostProcessNode` re-asserts it; both import
the same constant, so renderer and gate cannot drift apart. The correctable-
rejection body is not an answer — it quotes no regulatory content and states
only what to correct — so the advisory line does not apply to it.

## Implementation Note — answer synthesis

The current build is **deterministic end-to-end**: retrieval is keyword scoring
over the bundled knowledge base and `GenerateAnswerNode` assembles the grounded
answer rule-based from the ranked passages (lead sentence + cited passage
excerpts). There is no live model call and no model-client dependency — the
`llm` block in `config/config.yaml` is forwarded through `_parent_config()` for
forward-compatibility but is not consumed by any current node, and no
system prompt is read at runtime. The upgrade seam is documented in
`config/prompts/answer_synthesis_prompt.md`: an upgraded `GenerateAnswerNode`
swaps the rule-based assembly for a model call over the same
`ranked_documents` input and emits the same `grounded_answer` / `citations`
state contract, so no other node changes.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation strategy:** `propagate` (inner errors re-raised as `SubgraphError`).
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## Import Isolation Confirmation
- [x] Template imports the framework packages only — `framework/` and `shared/`.
- [x] No platform-internal SDK imports.
- [x] No agent base-class names in any base position other than the framework
      base classes (`AgentBaseGraph`, `BaseGraph`, `FunctionNode`, `GraphNode`).

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Framework base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step retrieval workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| Answer synthesis | Rule-based assembly | Model call | **Rule-based** | Deterministic and testable without a model; the upgrade swaps in at the documented seam |
| Knowledge-base storage | External vector store | Bundled JSON corpus | **Bundled JSON corpus** | Self-contained, deterministic CI; the retrieval contract (`retrieved_documents` JSON) is store-agnostic for a later vector-store upgrade |
| Caller parameter channel | JSON envelope inside the question | `input_context` mapping | **`input_context`** | Parameters are validated as typed fields at the boundary instead of being parsed out of free text, and the question text stays free text |
| Output invariant | Numeric rounding grid | Grounding + disclaimer + redaction invariant | **Grounding invariant** | The template quotes regulatory figures rather than computing aggregates; rounding them would falsify verifiable content |
