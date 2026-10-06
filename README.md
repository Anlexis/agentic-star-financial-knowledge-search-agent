# Financial Knowledge Search Agent

AI agent for searching a financial knowledge base, built with Agentic Star.

> **Category**: Cat 2 (domain-specific pipeline)
> **Industry**: Finance
> **Template ID**: FIN-C2-005

## Overview

Answers natural-language questions about financial regulation, compliance procedure and
product policy — customer verification duties, suitability rules, disclosure obligations,
record-retention periods — with a retrieval pipeline over a financial knowledge base.
Each question is validated and normalised, matching knowledge-base passages are retrieved,
reranked and filtered by a relevance threshold, and the answer is assembled **only** from
passages that clear that threshold, with numbered `[n]` citations and a standing advisory
disclaimer. When no passage is relevant enough, the agent says the knowledge base has
insufficient coverage instead of inventing an answer — and the answer never echoes the
caller's own text, so every line traces back to a cited source a reader can check.

The bundled knowledge base is a small sample of regulatory and procedural material so the
pipeline runs and tests end-to-end out of the box; a real deployment replaces it with its
own policy corpus behind the same node contract.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent
fails at graph compile / start-up preflight rather than starting in a partially
working state. This is intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/          agent implementation (nodes, graphs, schemas)
tests/        unit and boundary tests
config/       agent manifest, runtime parameters and the sample knowledge base
docs/         design and test documentation
```

See `docs/` for the design specification and the test specification.

## Customising

1. Adjust `config/config.yaml` for your own retrieval depth and relevance threshold.
2. Replace the sample knowledge base (`config/kb/`) with your own policy corpus.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
