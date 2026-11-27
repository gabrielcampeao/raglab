from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raglab.retrievers import MetadataFilter

if TYPE_CHECKING:
    from raglab.pipeline import Answer


@dataclass(frozen=True)
class TraceEvent:
    timestamp: str
    question: str
    filters: dict[str, str]
    cache_hit: bool
    verified: bool
    abstained: bool
    grounded_ratio: float
    context_tokens: int
    total_ms: float
    sources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "question": self.question,
            "filters": self.filters,
            "cache_hit": self.cache_hit,
            "verified": self.verified,
            "abstained": self.abstained,
            "grounded_ratio": round(self.grounded_ratio, 4),
            "context_tokens": self.context_tokens,
            "total_ms": round(self.total_ms, 3),
            "sources": list(self.sources),
        }


def build_trace_event(answer: Answer, filters: MetadataFilter | None = None) -> TraceEvent:
    return TraceEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        question=answer.question,
        filters=dict(filters) if filters else {},
        cache_hit=answer.trace.cache_hit,
        verified=answer.grounding.is_verified,
        abstained=answer.grounding.abstained,
        grounded_ratio=answer.grounding.grounded_ratio,
        context_tokens=answer.trace.context_tokens,
        total_ms=answer.trace.total_ms,
        sources=tuple(item.chunk.source for item in answer.sources),
    )


class TraceSink(ABC):
    @abstractmethod
    def record(self, event: TraceEvent) -> None:
        raise NotImplementedError


class InMemoryTraceSink(TraceSink):
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)


class JsonlTraceSink(TraceSink):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, event: TraceEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict()) + "\n")
