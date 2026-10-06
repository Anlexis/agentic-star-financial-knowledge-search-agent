# FIN-C2-005 — Unit Tests: manifest / config consistency
#
# Two files with different jobs, both live configuration rather than
# documentation:
#   config/agent.yaml   flat registration manifest read by the platform registry
#   config/config.yaml  runtime parameters passed to the graph as Graph(config=...)
#                       and read by _runtime_config() in the standalone server
#
# The declared class must BE the src/graph/graph.py agent class, and the
# declared retrieval tuning must reach the domain nodes. These tests pin
# config <-> code consistency so a drift fails fast in CI.
#
# Mirrors docs/03_test_spec.md section 2.8 (CFG-01..CFG-08).
# Deterministic — no model call, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import FinancialKnowledgeBaseSearchAgent, KnowledgeBaseSearchGraphNode, _runtime_config
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_manifest_is_flat_with_root_level_keys(self):
        # The registry reads every key at ROOT level; a nested `agent:` block
        # would make the whole manifest invisible to discovery.
        assert _MANIFEST["id"] == "FIN-C2-005"
        assert "agent" not in _MANIFEST
        assert _MANIFEST["enabled"] is True
        assert _MANIFEST["namespace"] == "fin"

    def test_cfg_02_declared_class_is_the_graph_class(self):
        # The class entry is a single dotted import path (module.Class).
        assert _MANIFEST["class"] == "src.graph.graph.FinancialKnowledgeBaseSearchAgent"
        assert _MANIFEST["class"].rsplit(".", 1)[-1] == FinancialKnowledgeBaseSearchAgent.__name__
        assert _MANIFEST["name"] == FinancialKnowledgeBaseSearchAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "FIN"
        assert _MANIFEST["base_type"] == "RAGAgent"
        assert _MANIFEST["generation_mode"] == "deterministic"


class TestManifestSecurity:
    def test_cfg_04_required_trust_level_matches_outer_gate_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared

    def test_cfg_05_no_secrets_or_extras_are_declared(self):
        # Declaring a secret or extra that is not actually used AND provisioned
        # fails the compile-time provisioning check; this build calls
        # ctx.secrets.require() nowhere and constructs no model client.
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestRuntimeConfig:
    def test_cfg_06_runtime_parameters_are_declared(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10  # framework retry ceiling
        assert isinstance(_RUNTIME["timeout_s"], int)

    def test_cfg_07_runtime_config_reader_sees_the_repo_file(self):
        cfg = _runtime_config()
        assert cfg["max_retry"] == _RUNTIME["max_retry"]
        assert cfg["retrieval"]["top_k"] == _RUNTIME["retrieval"]["top_k"]

    def test_cfg_08_retrieval_block_matches_node_defaults(self):
        # Node module defaults mirror config/config.yaml — a drift silently
        # changes tuning for any node invoked outside the inner graph.
        from src.nodes.rerank_filter_node import _DEFAULT_SCORE_THRESHOLD, _DEFAULT_TOP_K as rerank_top_k
        from src.nodes.retrieve_node import _DEFAULT_KB_PATH, _DEFAULT_TOP_K as retrieve_top_k

        retrieval = _RUNTIME["retrieval"]
        assert retrieval["top_k"] == retrieve_top_k == rerank_top_k
        assert retrieval["score_threshold"] == _DEFAULT_SCORE_THRESHOLD
        assert retrieval["kb_path"] == _DEFAULT_KB_PATH
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_cfg_09_parent_config_forwards_validated_runtime_blocks(self):
        cfg = KnowledgeBaseSearchGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"] == _RUNTIME["retrieval"]
        assert cfg["configurable"]["llm"] == _RUNTIME["llm"]
        assert cfg["configurable"]["retrieval"], "_parent_config() must never forward an empty retrieval block"

    def test_hitl_is_not_enabled(self):
        # This template declares no human-in-the-loop step.
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False


class TestBundledKnowledgeBase:
    def test_kb_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "the bundled knowledge base must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_kb_ids_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))
