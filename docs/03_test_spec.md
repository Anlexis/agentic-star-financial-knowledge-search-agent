# Test Specification — FIN-C2-005

**Template ID:** FIN-C2-005
**Template Name:** FinancialKnowledgeBaseSearchAgent
**Category:** Cat 2 (nested RAG)

This document is the contract the shipped suite implements
(`tests/unit/` + `tests/proof_of_boundary/`). Every table row below maps to
tests that exist in this repository.

## 1. Scope & Invocation Conventions

- Per-node unit tests for the 5 inner domain nodes + the 2 outer gate nodes.
- The caller-data contract and the external output schema, tested together as
  the two hostile boundaries.
- Manifest/runtime-config consistency (`config/agent.yaml` + `config/config.yaml`
  vs. code) and bundled-corpus integrity.
- Retrieval quality (golden queries over `config/kb/financial_kb.json`).
- Inner-graph (`DomainWorkflowGraph`) and outer-graph
  (`FinancialKnowledgeBaseSearchAgent`) composition / integration.
- Boundary tests: import isolation, State msgpack safety, invoke order,
  HITL propagation (conditional), server boot, the entry-point auth boundary,
  and end-to-end `/invoke`.

**Trust-gate invocation canon.** Every per-node test invokes the node via
`node(state)` — through `BaseNode.__call__`, which runs the trust gate, then
the framework input gate, then `execute()` — never a bare
`node.execute(state)`. The state builder sets `caller_trust_level` to
`TrustLevel.VERIFIED_EXTERNAL.value` for the two outer gate slots
(PreProcessNode / PostProcessNode — the manifest's declared caller level) and
`TrustLevel.ANONYMOUS.value` for the five inner domain nodes.
**Sole carve-out:** the tests that prove a guarantee is the TEMPLATE's own
(instruction-override refusal, identifier redaction) call `execute()` directly,
so the behaviour is demonstrated with no framework gate in front of it. That is
the point of those tests, not a shortcut.

**Two rejection classes, asserted differently.** A rejection the caller can fix
by correcting a value (empty question, oversize question, out-of-contract
`input_context`) COMPLETES: the node returns `status=SUCCESS` carrying a reason
code, publishes nothing downstream, and `PostProcessNode` renders the fixed
reason sentence as the body. Tests for those cases therefore assert
`status=SUCCESS` **together with** the absence of any product of the request
(no `validated_input`, no `search_query`, no answer content) — a bare status
assertion would not distinguish a rejection from a normal run. A rejection the
caller cannot fix by rewording — instruction-override content, a credential
pattern in the assembled answer — still TERMINATES with `status=ERROR`, and
those tests are unchanged.

**Framework masking expectations.** The framework input gate masks
`user_input`/`validated_input`/`llm_response` (e-mail, phone/card digit groups,
Title-Case name bigrams) to `[MASKED]` before `execute()` runs. Positive-path
payloads are therefore lowercase, identifier-free regulatory phrasing;
intentional-identifier tests assert the raw value is gone and the marker
(`[MASKED]`, or `[REDACTED]` for the template's own screen) is present. Domain
fields (`grounded_answer`, `formatted_answer`, `retrieved_documents`, …) are
not masked.

**Audit muting.** `shared.*` is never sys.modules-stubbed (the framework
imports `shared.security` at load time). The domain audit emitter is muted via
an autouse fixture patching `src.nodes.<mod>.emit_trace_event`; the audit
assertion test re-patches the same attribute with a spy and asserts on
`call.args[1]` (the event payload).

## 2. Unit Test Cases

### 2.1 PreProcessNode (outer pre_process slot) — `test_pre_process_node.py`

| ID | Case | Input | Expected |
|----|------|-------|----------|
| PRE-01 | Valid query | lowercase regulatory question | `status=SUCCESS`, `validated_input` set, `enriched_context` carries channel/source |
| PRE-02 | Empty input | `""` / whitespace | correctable rejection: `status=SUCCESS`, `error_log` non-empty, no `validated_input` |
| PRE-03 | Missing / non-string input | `user_input` absent; dict payload | correctable rejection: `status=SUCCESS` |
| PRE-04 | Oversize question | >2000 characters | correctable rejection: `status=SUCCESS`, `error_log` naming the character bound |
| PRE-05 | Identifier screen | IBAN → template screen; e-mail / 4-4-4 digit groups → framework mask | raw identifier absent from `validated_input`; `[REDACTED]` / `[MASKED]` present; the IBAN case is also proven on a direct `execute()` |
| PRE-06 | Instruction-override refusal | 5 override phrasings | `status=ERROR`, nothing carried forward |
| PRE-07 | Refusal without any gate in front | direct `execute()` | `status=ERROR`, refusal reason in `error_log`, no `validated_input` |
| PRE-08 | Legitimate questions unaffected | 4 questions containing the same words in ordinary use | `status=SUCCESS`, `validated_input` set |
| PRE-09 | Caller-data contract | invalid `channel` / `category` / `top_k` / non-mapping | correctable rejection: `status=SUCCESS`, no `validated_input`, rejected value never echoed |
| PRE-10 | Audit | valid query | `pre_process_complete` emitted; payload carries `input_chars` + `identifier_redactions` |

### 2.2 InputValidateNode (inner domain node 1) — `test_input_validate_node.py`

| ID | Case | Expected |
|----|------|----------|
| VAL-01 | Plain text | `search_query` normalised; `query_filters` JSON with both fields `None` |
| VAL-02 | Whitespace collapse | multiple spaces / newlines → single spaces |
| VAL-03 | JSON-looking question | treated as text — parameters ride `input_context`, never an envelope in the question |
| VAL-04 | Valid caller parameters | `category` + `top_k` published into `query_filters` |
| VAL-05 | Invalid `top_k` (10-case matrix incl. `NaN` / `Infinity` / raw floats / bool / numeric string) | correctable rejection: `status=SUCCESS`, no `search_query`, field named in `error_log` |
| VAL-06 | Invalid `category` (5 cases) | correctable rejection: `status=SUCCESS`, field named in `error_log` |
| VAL-07 | Instruction-override refusal | refused both through `__call__` and on a direct `execute()` |
| VAL-08 | Oversize query | truncated to 2000 chars with an intake note |
| VAL-09 | Empty request | note, not error |

### 2.3 RetrieveNode (inner domain node 2) — `test_retrieve_node.py`

| ID | Case | Expected |
|----|------|----------|
| RET-01 | Top hit | the verification query ranks `kb-001` first |
| RET-02 | Ordering | scores strictly descending, all > 0 |
| RET-03 | Entry shape | exactly `{id,title,category,source,score,excerpt}`; excerpt ≤ 400 chars |
| RET-04 | Category filter | candidate pool restricted to the requested category |
| RET-05 | Empty query | no candidates |
| RET-06 | Seeded `kb_path` | unreadable path → empty result + intake note |
| RET-07 | Path containment | a path resolving outside the template is refused with a note |
| RET-08 | Note accumulation | upstream notes are appended to, never clobbered |

### 2.4 RerankFilterNode (inner domain node 3) — `test_rerank_filter_node.py`

| ID | Case | Expected |
|----|------|----------|
| RRF-01 | Default threshold | candidates below 0.25 dropped |
| RRF-02 | Seeded `score_threshold` | stricter floor applied |
| RRF-03 | Seeded `top_k` | survivors capped |
| RRF-04 | Category boost | matching category +0.1 and reranked |
| RRF-05 | Boost cap | score never exceeds 1.0 |
| RRF-06 | Caller `top_k` | narrows when stricter, never widens |
| RRF-07 | Malformed candidates | non-dict skipped; uncoercible score → 0.0 → dropped |
| RRF-08 | Tie-break | deterministic by id |

### 2.5 GenerateAnswerNode (inner domain node 4) — `test_generate_answer_node.py`

| ID | Case | Expected |
|----|------|----------|
| GEN-01 | Citation markers | `[n] <title>: <excerpt>` per ranked passage |
| GEN-02 | No caller text | the question never appears in `grounded_answer` |
| GEN-03 | Citation list | refs mirror ranked order; ids and sources carried |
| GEN-04 | Grounding | body content traces only to the ranked passages |
| GEN-05 | No coverage | empty ranked set → explicit insufficient-coverage answer, zero citations |
| GEN-06 | Counterfeit markers | a `[9]` marker in the question never reaches the answer |

### 2.6 OutputFormatNode (inner domain node 5) — `test_output_format_node.py`

| ID | Case | Expected |
|----|------|----------|
| FMT-01 | Composition | heading + body + `## Sources` + advisory disclaimer; `status` is the plain string |
| FMT-02 | Source suffix | omitted when the source is blank (no empty parentheses) |
| FMT-03 | Disclaimer | present on every answer, including the empty-body case |
| FMT-04 | No citations | explicit "none (no knowledge-base passage cleared the relevance threshold)" line |
| FMT-05 | Missing body | fallback text, still SUCCESS |
| FMT-06 | Shared constant | renderer and output gate import the same disclaimer constant |

### 2.7 PostProcessNode (outer post_process slot) — `test_post_process_node.py`

| ID | Case | Expected |
|----|------|----------|
| POST-01 | Clean answer | passes through byte-identical; `status` is the plain string |
| POST-02 | Empty upstream answer | qualified fallback message, still carrying the disclaimer |
| POST-03 | Missing disclaimer | re-attached by the gate |
| POST-04..07 | Credential forms (API key, credential assignment, JWT, Bearer) | whole answer withheld, `status=ERROR`, secret absent from `formatted_output` AND `result` |
| POST-08 | Withheld output | carries none of the original answer content |

### 2.8 Manifest / runtime config — `test_config_manifest.py`

| ID | Case | Expected |
|----|------|----------|
| CFG-01 | Flat manifest | every key at ROOT level; no `agent:` block; `namespace: fin` |
| CFG-02 | Class contract | `class:` is the dotted path to the graph class; `name` matches `Graph().name` |
| CFG-03 | Classification | Cat 2 / FIN / RAGAgent / `generation_mode: deterministic` |
| CFG-04 | Trust level | manifest value matches both outer gate nodes |
| CFG-05 | Provisioning | `requires.secrets` and `requires.extras` are both empty |
| CFG-06 | Runtime parameters | `max_retry` within the framework ceiling; `timeout_s` declared |
| CFG-07 | Config reader | `_runtime_config()` returns the repo's `config/config.yaml` |
| CFG-08 | Node defaults | module defaults mirror the declared retrieval block; `kb_path` exists |
| CFG-09 | Forwarding | `_parent_config()` forwards the validated retrieval + llm blocks, never empty |
| — | Corpus integrity | well-formed entry list, ≥5 entries, unique ids |

### 2.9 Retrieval quality — `test_retrieval_quality.py`

Golden-query suite over `config/kb/financial_kb.json` driving the real inner
chain (`InputValidateNode` → `RetrieveNode` → `RerankFilterNode`), pinning the
expected top hit per domain query (QUAL-01..QUAL-07). The scorer is
deterministic, so exact top-1 assertions catch corpus / scorer / threshold
regressions.

### 2.10 Caller-data contract + output schema — `test_caller_data_and_output_schema.py`

The largest suite, covering the two hostile boundaries end to end:

| Group | Coverage |
|-------|----------|
| `input_context` validation | defaults, valid full context, unknown keys ignored, non-mapping rejected, invalid `channel`/`category`/`top_k` matrices, rejected values never echoed, bound edges accepted |
| `_finite_in_range` | 14-case rejection matrix (`NaN`, `±Infinity` as strings AND raw floats, bools, non-numerics, out-of-range) + accepted cases |
| Retrieve/Rerank fail-safes | malformed seeded scalars fall back rather than crash or run unbounded; a non-finite threshold or candidate score never clears the relevance floor; caller `top_k` narrows but never widens |
| Runtime-config plumbing | `_runtime_config()` reads the repo file; a missing file degrades to `{}`; declared settings reach inner-state seeding; **the context bridge proven END-TO-END** — a caller `top_k` override changes the number of cited sources in the final answer through the full nested graph |
| Output boundary | verbatim caller-text redaction, identifier redaction from knowledge-base content, the disclaimer invariant on EVERY representation, **layer ORDER** (caller-text removal before the identifier pattern scan), and the credential re-scan after the rewriting layers |
| Both-directions probes | 15 quoted/structural tokens cross the boundary BYTE-IDENTICAL (amounts on and off any grid, thresholds in prose, retention periods, a numbered section heading after a 3-letter code, article references, passage ids, part numbers, document ids, invoice ids, an embedded digit run inside a hyphenated identifier, acronym-years, horizons, long digit runs); 3 identifier leak forms are redacted |

## 3. Integration

### 3.1 Inner graph — `test_domain_workflow_graph.py`

| ID | Case | Expected |
|----|------|----------|
| INT-01 | Composition | inherits `BaseGraph`; registers exactly the 5 domain nodes; no initialize/finalize |
| INT-02 | Seeding | `_extra_initial_state()` seeds the scalar tuning fields, seeds only `input_context` when unconfigured, and bridges the caller's `input_context` |
| INT-03 | Output shape | `get_output()` emits the merge contract; `route()` → END on error |
| INT-04 | Inner e2e | full inner `invoke()` → SUCCESS; formatted answer + disclaimer + `kb-001` citation; inner `node_history` = the 5 domain nodes in linear order |

### 3.2 Outer graph + e2e — `test_graph_composition.py`

| ID | Case | Expected |
|----|------|----------|
| INT-05 | Outer composition | inherits `AgentBaseGraph` directly; `Graph` alias; `add_edges()` NOT overridden |
| INT-06 | Backbone slots | `compile()` fills all 5; pre/main/post are PreProcessNode / KnowledgeBaseSearchGraphNode / PostProcessNode |
| INT-07 | `get_subgraph()` | returns `DomainWorkflowGraph` carrying the forwarded retrieval config |
| INT-08 | `extract_input()` | prefers `validated_input`, falls back to `user_input` |
| INT-09 | `merge_output()` | inner `formatted_answer` → outer `knowledge_base_answer` AND `result`; `citations`/`status` mapped; `error_code` carried across the boundary (blank on a normal pass) so a rejection settled before the inner run can still be reported; changed keys only |
| INT-10 | Config degradation | an unreadable runtime-config file forwards nothing rather than junk; a non-finite or out-of-range declared value is not forwarded |
| INT-11 | e2e happy path | VERIFIED_EXTERNAL invoke → SUCCESS; `output` = gated formatted answer; PostProcessNode traversed |
| INT-12 | e2e trust denial | ANONYMOUS invoke → ERROR; empty `output`; PostProcessNode NOT traversed |
| — | Serialization helpers | `to_json`/`from_json` round-trip; None/malformed handling |

## 4. Boundary suite

| ID | File | Expected |
|----|------|----------|
| PB-IMPORT | `test_import_isolation.py` | no platform-internal import anywhere under `src/` |
| PB-STATE | `test_state_safety.py` | `State` has no credential-named fields and no `BaseModel` / `InvocationContext` annotations |
| PB-6 | `test_pb_invoke_order.py` | full `Graph().invoke()` with `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` (never `for_internal()`) over the payload byte-equal to `deploy/invoke_payload.json`'s `input` → SUCCESS with outer `node_history` exactly `[InitializeNode, PreProcessNode, KnowledgeBaseSearchGraphNode, PostProcessNode, FinalizeNode]` |
| PB-7 | `test_pb7_hitl_interrupt_propagation.py` | **Not applicable** — no graph class declares `propagate_hitl=True` and there is no cross-boundary `interrupt()` checkpoint, so the file ships as a real, importable skip stub |
| PB-BOOT | `test_server_boot.py` | Part 1 (boot): `import src.api.server` does not raise; the module-level agent is this template's class, compiled, and carries the live `config/config.yaml` (`max_retry == 3`); fresh ctor→`compile()` fills the 5 backbone slots; `/health` reports the agent |
| PB-AUTH | `test_server_boot.py` | Part 2 (entry-point auth boundary, agent stubbed): missing / wrong / non-Bearer-scheme / empty / non-ASCII Authorization → generic `401` and the request never reaches the agent; the 401 body leaks neither the token nor the failure reason; an oversized `input_context` → `413` at the adapter; a correct Bearer elevates to VERIFIED_EXTERNAL; an unset or empty token leaves the caller ANONYMOUS; middleware-established trust is preserved, never demoted |
| PB-E2E | `test_invoke_e2e.py` | 19 cases through the REAL ASGI `POST /invoke` with Bearer auth: grounded cited answer, no-coverage as a SUCCESS path, `top_k` override, category filter reaching the inner retrieval; empty input, invalid `channel`/`category` and the 9-case non-finite `top_k` matrix each refused as a correctable rejection — the response carries `status=success` and the reason sentence as `output`, and no knowledge-base content, so the caller can correct the value and send again; instruction-override refused with nothing published (`status=error`, empty `output`); counterfeit citation markers never surfaced; no direct identifier in the answer |

> PB-IMPORT, PB-STATE, PB-6, PB-BOOT, PB-AUTH and PB-E2E are mandatory. PB-7 applies
> only to templates that propagate HITL interrupts across the graph boundary —
> this template does not, so its skip is expected and must not block the gate.

## 5. Test Execution Summary

- Runner: real SDK wheel (`agenticstar-agentcore==1.0.1`), `python -m pytest tests/`
- Collected: 303
- Pass: 302 / Fail: 0 / Skip: 1 (PB-7 — not applicable, see above)
- Determinism: no model call, no network; retrieval and answer assembly are
  rule-based, so every assertion is exact rather than statistical
