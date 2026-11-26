from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    _bootstrap_path()
    from projects.incident_copilot.copilot import (
        PLAYBOOK_QUESTIONS,
        build_incident_copilot,
        run_ablation_suite,
        run_autonomous_playbook,
    )

    copilot = build_incident_copilot()
    print(f"Atlas Pay Incident Copilot initialized ({len(copilot.chunks)} structural chunks)")
    print()
    print("Autonomous incident triage and architecture review")
    print("=" * 50)

    answers = run_autonomous_playbook(copilot)
    for index, ((_, filters), answer) in enumerate(
        zip(PLAYBOOK_QUESTIONS, answers, strict=True), start=1
    ):
        scope = (
            ", ".join(f"{key}={value}" for key, value in filters.items())
            if filters
            else "unfiltered"
        )
        status = (
            "abstained (out of scope)"
            if answer.grounding.abstained
            else ("verified" if answer.grounding.is_verified else "unverified")
        )
        print(f"\n[{index}] Question: {answer.question}")
        print(f"    Scope: {scope}")
        print(f"    Answer: {answer.text}")
        if answer.sources and not answer.grounding.abstained:
            top = answer.sources[0].chunk
            section = top.metadata.get("section", "root")
            print(f"    Top citation: {top.source} ({section})")
        print(
            f"    Verification: {status} | tokens={answer.trace.context_tokens} "
            f"| latency={answer.trace.total_ms:.2f}ms"
        )

    print("\n\nPipeline ablation benchmark (24 incident cases)")
    print("=" * 50)
    suite = run_ablation_suite()
    header = (
        f"{'pipeline':<13} {'hit@1':>6} {'mrr@3':>6} {'ndcg@3':>7} "
        f"{'recall':>7} {'abstain':>8} {'verified':>9} {'ms':>7}"
    )
    print(header)
    for name, metrics in suite["configurations"].items():
        print(
            f"{name:<13} {metrics['hit_rate_at_1']:>6.2f} {metrics['mrr_at_3']:>6.2f} "
            f"{metrics['ndcg_at_3']:>7.2f} {metrics['recall_at_3']:>7.2f} "
            f"{metrics['abstention_accuracy']:>8.2f} {metrics['verified_rate']:>9.2f} "
            f"{metrics['mean_latency_ms']:>7.2f}"
        )


if __name__ == "__main__":
    main()
