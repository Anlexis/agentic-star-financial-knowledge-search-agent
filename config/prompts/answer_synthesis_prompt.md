# Answer Synthesis Prompt — FIN-C2-005 (model-synthesis upgrade seam)

> **The bundled build does NOT use this prompt at runtime.** `GenerateAnswerNode`
> assembles the answer deterministically (rule-based grounded assembly over
> `ranked_documents`); no node reads this file. It documents the synthesis
> contract for the model-backed upgrade described in `docs/02_design.md`
> ("Implementation Note — answer synthesis"), so the swap changes only the
> inside of `GenerateAnswerNode.execute()`.

## Contract (model-backed GenerateAnswerNode)

- **Input:** the same `ranked_documents` JSON (id / title / category / source /
  score / excerpt) the deterministic node reads.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers) and `citations` (JSON list of
  `{ref, id, title, source}`).
- **Grounding rule:** every factual statement in the answer must be traceable
  to one of the supplied passages via a `[n]` marker; content not present in
  the passages must not be asserted.
- **No-coverage rule:** when no passage supports the question, say so and
  recommend refining the query or escalating to the compliance team — never
  answer from parametric knowledge.
- **Caller text stays out of the answer:** the question is NOT rendered into
  the answer body. The external answer carries knowledge-base-derived content
  and the template's own fixed structure only, and the output gate
  (`src/nodes/post_process_node.py`) redacts any verbatim caller text that
  reaches it. Pass the question to the model as an instruction, never as
  content to echo back.
- **Tone:** neutral, compliance-appropriate, no individualized recommendations
  (the advisory disclaimer is appended downstream by `OutputFormatNode`).

## Prompt template

```
You answer financial compliance, regulatory, and product questions strictly
from the knowledge-base passages provided below.

Question:
{search_query}

Passages (each with a reference number):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   knowledge base has insufficient coverage and stop.
2. Mark every factual statement with the [n] reference of its passage.
3. Do not repeat the question back in your answer.
4. Do not give individualized financial, legal, or investment advice.
5. Keep the answer under 300 words.
```

## Configuration coupling

The `llm` block in `config/config.yaml` (`temperature`, `max_tokens`) is already
forwarded to the inner graph via
`KnowledgeBaseSearchGraphNode._parent_config()` under
`config["configurable"]["llm"]`; an upgraded node reads it from there.
