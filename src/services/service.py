"""AgentCore Platform v1.0"""

# Service layer: domain queries, external API wrappers, data aggregation.
# Must NOT contain business logic, routing, or credentials.
# Nodes call this; this calls shared/services/ for external integrations.
#
# Contract: FIN-C2-005 runs a fully offline retrieval pipeline - RetrieveNode
# scores the bundled financial knowledge base, so no external domain service
# is called. `Service` is the seam where a real deployment wires its own
# vector store or policy-document API behind the same node contract. It is
# kept as an explicit NotImplementedError stub rather than a silent empty
# result, so an accidental call fails loudly instead of degrading a regulatory
# answer to "no coverage".

from __future__ import annotations

from typing import Any


class Service:
    """Domain data-access seam for FIN-C2-005 (offline build; no external service)."""

    async def fetch(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch domain data for the given query.

        Not implemented in the bundled build - the retrieval pipeline grounds
        on the bundled knowledge base in RetrieveNode. A real deployment
        implements this to call its own backend.
        """
        raise NotImplementedError("Service.fetch() is a deliberate stub in the bundled build")
