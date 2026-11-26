from projects.incident_copilot.copilot import (
    PLAYBOOK_QUESTIONS,
    load_incident_corpus,
    run_ablation_suite,
    run_autonomous_playbook,
    run_chunk_sweep,
)


def test_incident_corpus_loads_twelve_documents_with_metadata() -> None:
    documents = load_incident_corpus()

    assert len(documents) == 12
    kinds = {document.metadata["kind"] for document in documents}
    assert kinds == {"runbook", "postmortem", "architecture"}
    assert all("service" in document.metadata for document in documents)


def test_autonomous_playbook_answers_incidents_and_refuses_traps() -> None:
    assert len(PLAYBOOK_QUESTIONS) >= 6
    outcomes = run_autonomous_playbook()

    assert len(outcomes) == len(PLAYBOOK_QUESTIONS)
    for item in outcomes:
        assert item.grounding.is_verified is True
    answered = [item for item in outcomes if not item.grounding.abstained]
    abstained = [item for item in outcomes if item.grounding.abstained]
    assert len(answered) == 5
    assert len(abstained) == 1


def test_ablation_suite_demonstrates_senior_pipeline_gains() -> None:
    suite = run_ablation_suite()
    configs = suite["configurations"]

    assert set(configs) >= {"dense", "tfidf", "bm25", "hybrid_rrf", "senior_full"}
    senior = configs["senior_full"]
    dense = configs["dense"]
    assert senior["ndcg_at_3"] > dense["ndcg_at_3"]
    assert senior["verified_rate"] > dense["verified_rate"]
    assert senior["hit_rate_at_3"] >= 0.95
    assert senior["abstention_accuracy"] == 1.0
    assert senior["verified_rate"] >= 0.95


def test_chunk_sweep_covers_multiple_chunk_sizes() -> None:
    sweep = run_chunk_sweep()

    assert len(sweep) >= 5
    sizes = [entry["max_chars"] for entry in sweep]
    assert sizes == sorted(sizes)
    assert all(0.0 <= float(entry["ndcg_at_3"]) <= 1.0 for entry in sweep)


def test_incident_copilot_supports_qdrant_backend() -> None:
    from projects.incident_copilot.copilot import build_incident_copilot

    copilot = build_incident_copilot(backend="qdrant")
    outcomes = run_autonomous_playbook(copilot)

    assert len(outcomes) == len(PLAYBOOK_QUESTIONS)
    assert all(item.grounding.is_verified for item in outcomes)
