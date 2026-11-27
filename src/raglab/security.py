from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from raglab.models import Chunk, Document, SearchResult

_INVISIBLE_PATTERN = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]"
)

DEFAULT_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(
            r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions",
            re.IGNORECASE,
        ),
    ),
    (
        "system_override",
        re.compile(
            r"(?:system\s+override|developer\s+mode|jailbreak\s+mode)\s*:",
            re.IGNORECASE,
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"(?:reveal|output|print|exfiltrate|leak)\s+(?:the\s+)?"
            r"(?:system\s+prompt|api\s+key|admin\s+secret|secret\s+token)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_delimiter_spoof",
        re.compile(r"<\|(?:im_start|system|assistant)\|>", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class SecurityFinding:
    chunk_id: str
    rule: str
    excerpt: str


def strip_invisible_characters(text: str) -> str:
    return _INVISIBLE_PATTERN.sub("", text)


class ContextSanitizer:
    def __init__(
        self,
        *,
        drop_suspicious: bool = True,
        rules: Sequence[tuple[str, re.Pattern[str]]] = DEFAULT_INJECTION_RULES,
    ) -> None:
        self.drop_suspicious = drop_suspicious
        self.rules = tuple(rules)
        self._last_findings: list[SecurityFinding] = []

    @property
    def last_findings(self) -> tuple[SecurityFinding, ...]:
        return tuple(self._last_findings)

    def clean_document(self, document: Document) -> Document:
        return Document(
            id=document.id,
            text=strip_invisible_characters(document.text),
            metadata=dict(document.metadata),
        )

    def scan_chunk(self, chunk: Chunk) -> list[SecurityFinding]:
        cleaned = strip_invisible_characters(chunk.text)
        findings: list[SecurityFinding] = []
        for rule_name, pattern in self.rules:
            match = pattern.search(cleaned)
            if match is not None:
                findings.append(
                    SecurityFinding(
                        chunk_id=chunk.id,
                        rule=rule_name,
                        excerpt=match.group(0),
                    )
                )
        return findings

    def sanitize_results(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        self._last_findings = []
        sanitized: list[SearchResult] = []
        for item in results:
            cleaned_text = strip_invisible_characters(item.chunk.text)
            clean_chunk = Chunk(
                id=item.chunk.id,
                text=cleaned_text,
                document_id=item.chunk.document_id,
                position=item.chunk.position,
                metadata=dict(item.chunk.metadata),
            )
            findings = self.scan_chunk(clean_chunk)
            if findings:
                self._last_findings.extend(findings)
                if self.drop_suspicious:
                    continue
                scrubbed = cleaned_text
                for _, pattern in self.rules:
                    scrubbed = pattern.sub("[redacted]", scrubbed)
                clean_chunk = Chunk(
                    id=clean_chunk.id,
                    text=scrubbed,
                    document_id=clean_chunk.document_id,
                    position=clean_chunk.position,
                    metadata=dict(clean_chunk.metadata),
                )
            sanitized.append(SearchResult(chunk=clean_chunk, score=item.score))
        return sanitized
