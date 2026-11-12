from app.services.copilot_graph import CopilotState, rag_node


def test_rag_node_returns_only_keys_declared_in_the_state(monkeypatch):
    """LangGraph silently drops undeclared keys, so every key a node
    returns must exist in CopilotState."""
    fake_search = {
        "results": [
            {
                "filename": "policy.pdf",
                "page_number": 1,
                "text": "3.2 A variance of more than 5 percent requires an explanation.",
                "document_id": "doc-1",
                "relevance_score": 0.9,
            }
        ]
    }
    monkeypatch.setattr(
        "app.services.copilot_graph.search_documents",
        lambda question, top_k: fake_search,
    )

    update = rag_node({"question": "What does the policy say?"})

    assert set(update) <= set(CopilotState.__annotations__)