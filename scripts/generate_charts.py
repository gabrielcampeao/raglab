from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any

FONT = "Segoe UI, Inter, Arial, sans-serif"


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _text(
    x: float,
    y: float,
    body: str,
    *,
    fill: str = "#f8fafc",
    size: float = 12,
    weight: str = "400",
    anchor: str = "start",
    spacing: str = "",
) -> str:
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" '
        f'font-family="{FONT}" font-size="{size}" font-weight="{weight}"{extra}>'
        f"{escape(body)}</text>"
    )


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        "<defs>",
        '<linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '<stop offset="0%" stop-color="#0b1120"/>',
        '<stop offset="100%" stop-color="#111827"/>',
        "</linearGradient>",
        '<marker id="arrowCyan" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 1 L 9 5 L 0 9 z" fill="#38bdf8"/>',
        "</marker>",
        '<marker id="arrowEmerald" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '<path d="M 0 1 L 9 5 L 0 9 z" fill="#34d399"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="16" fill="url(#bgGrad)"/>',
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="15" '
        'fill="none" stroke="#1e293b" stroke-width="2"/>',
    ]


def _card(
    x: int,
    y: int,
    w: int,
    h: int,
    border: str,
    title: str,
    lines: tuple[str, ...],
) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="#1e293b" stroke="{border}" stroke-width="1.6"/>',
        _text(x + 14, y + 26, title, size=13, weight="700"),
    ]
    for idx, line in enumerate(lines):
        parts.append(_text(x + 14, y + 46 + idx * 17, line, fill="#94a3b8", size=11.5))
    return "\n".join(parts)


def _arrow(x1: int, y1: int, x2: int, y2: int, color: str, marker: str, width: float = 2.0) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"/>'
    )


def render_architecture_svg() -> str:
    width, height = 980, 510
    out = _svg_header(width, height, "Two stage RAG architecture in raglab")
    out.append(_text(36, 42, "Two stage RAG architecture in raglab", size=21, weight="700"))
    subtitle = (
        "Offline structural indexing paired with online hybrid retrieval, "
        "cross feature reranking, MMR, and verification"
    )
    out.append(_text(36, 65, subtitle, fill="#94a3b8", size=13))

    out.append(
        '<rect x="28" y="84" width="924" height="168" rx="12" '
        'fill="#0f172a" stroke="#334155" stroke-width="1.2"/>'
    )
    out.append(
        _text(
            46,
            108,
            "OFFLINE INDEXING PIPELINE",
            fill="#38bdf8",
            size=12,
            weight="700",
            spacing="0.8",
        )
    )

    out.append(
        _card(
            46,
            124,
            195,
            108,
            "#38bdf8",
            "1. Document Loader",
            ("Reads .md and .txt files", "Attaches source, kind,", "and service metadata"),
        )
    )
    out.append(
        _card(
            276,
            124,
            235,
            108,
            "#38bdf8",
            "2. Markdown Chunker",
            ("Tracks ATX heading paths", "Keeps SQL blocks whole", "Packs sentences with overlap"),
        )
    )
    out.append(
        _card(550, 116, 180, 54, "#818cf8", "3a. Vector Store", ("Hashing or TFIDF L2 vectors",))
    )
    out.append(
        _card(550, 182, 180, 54, "#818cf8", "3b. BM25 Index", ("IDF + length normalized counts",))
    )
    out.append(
        _card(
            764,
            124,
            170,
            108,
            "#64748b",
            "JSON Snapshot",
            ("Deterministic save and", "zero refit reload for", "stateless embeddings"),
        )
    )

    out.append(_arrow(241, 178, 272, 178, "#38bdf8", "arrowCyan"))
    out.append(
        '<path d="M 511 168 L 530 168 L 530 143 L 546 143" fill="none" '
        'stroke="#38bdf8" stroke-width="2" marker-end="url(#arrowCyan)"/>'
    )
    out.append(
        '<path d="M 511 188 L 530 188 L 530 209 L 546 209" fill="none" '
        'stroke="#38bdf8" stroke-width="2" marker-end="url(#arrowCyan)"/>'
    )
    out.append(_arrow(730, 143, 760, 162, "#38bdf8", "arrowCyan", 1.8))

    out.append(
        '<rect x="28" y="272" width="924" height="176" rx="12" '
        'fill="#0f172a" stroke="#334155" stroke-width="1.2"/>'
    )
    out.append(
        _text(
            46,
            296,
            "ONLINE QUERY AND VERIFICATION PIPELINE",
            fill="#34d399",
            size=12,
            weight="700",
            spacing="0.8",
        )
    )

    out.append(
        _card(
            46,
            312,
            165,
            116,
            "#34d399",
            "4. Query + Filters",
            ("User question", "Metadata prefilter:", "kind and service"),
        )
    )
    out.append(
        _card(
            235,
            312,
            175,
            116,
            "#34d399",
            "5. Hybrid RRF",
            ("Prefiltered candidate pool", "Fuses Dense + BM25", "by reciprocal rank"),
        )
    )
    out.append(
        _card(
            434,
            312,
            185,
            116,
            "#f59e0b",
            "6. Rerank + MMR",
            ("Term coverage and span", "Section heading boost", "MMR removes duplicates"),
        )
    )
    out.append(
        _card(
            643,
            312,
            140,
            116,
            "#a78bfa",
            "7. Token Packer",
            ("Enforces max token", "context budget with", "numbered sources"),
        )
    )
    out.append(
        _card(
            807,
            312,
            127,
            116,
            "#f43f5e",
            "8. Verifier",
            ("Checks [n] bounds,", "claim support, and", "trap abstention"),
        )
    )

    for x1, x2 in ((211, 231), (410, 430), (619, 639), (783, 803)):
        out.append(_arrow(x1, 370, x2, 370, "#34d399", "arrowEmerald"))

    out.append(
        '<path d="M 640 236 L 640 260 L 322 260 L 322 308" fill="none" stroke="#818cf8" '
        'stroke-width="1.8" stroke-dasharray="5,4" marker-end="url(#arrowCyan)"/>'
    )

    out.append(
        '<rect x="28" y="462" width="924" height="32" rx="8" '
        'fill="#1e293b" stroke="#334155" stroke-width="1"/>'
    )
    footer = (
        "Telemetry attached to every Answer: PipelineTrace "
        "(retrieval_ms, rerank_ms, generation_ms, context_tokens) + GroundingReport"
    )
    out.append(_text(46, 483, footer, fill="#cbd5e1", size=12))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_ablation_svg(ablation: dict[str, Any]) -> str:
    width, height = 960, 470
    configs = ablation["configurations"]
    order = [
        ("dense", "Dense"),
        ("tfidf", "TFIDF"),
        ("bm25", "BM25"),
        ("hybrid_rrf", "Hybrid RRF"),
        ("senior_full", "Senior Full"),
    ]
    series = [
        ("hit_rate_at_1", "Hit@1", "#38bdf8"),
        ("ndcg_at_3", "NDCG@3", "#34d399"),
        ("recall_at_3", "Recall@3", "#a78bfa"),
        ("verified_rate", "Verified E2E", "#f59e0b"),
    ]

    out = _svg_header(width, height, "Pipeline ablation benchmark on Incident Copilot")
    out.append(
        _text(
            36,
            42,
            "Pipeline ablation on Atlas Pay Incident Copilot (24 benchmark cases)",
            size=20,
            weight="700",
        )
    )
    subtitle = (
        "Retrieval ranking quality on 20 answerable incidents plus "
        "end to end verified rate across all 24 cases"
    )
    out.append(_text(36, 64, subtitle, fill="#94a3b8", size=13))

    for idx, (_, label, color) in enumerate(series):
        lx = 548 + idx * 98
        out.append(f'<rect x="{lx}" y="82" width="12" height="12" rx="3" fill="{color}"/>')
        out.append(_text(lx + 18, 92, label, fill="#cbd5e1", size=12))

    chart_left, chart_top, chart_bottom, chart_right = 76, 118, 398, 916
    chart_h = chart_bottom - chart_top

    for step in range(6):
        val = step * 0.2
        y = chart_bottom - val * chart_h
        out.append(
            f'<line x1="{chart_left}" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}" '
            'stroke="#1e293b" stroke-width="1.2"/>'
        )
        out.append(_text(chart_left - 12, y + 4, f"{val:.1f}", fill="#94a3b8", anchor="end"))

    group_w = (chart_right - chart_left) / len(order)
    bar_w, gap = 28, 6

    for g_idx, (key, title) in enumerate(order):
        metrics = configs[key]
        total_bars_w = len(series) * bar_w + (len(series) - 1) * gap
        start_x = chart_left + g_idx * group_w + (group_w - total_bars_w) / 2
        for s_idx, (field, _, color) in enumerate(series):
            val = float(metrics[field])
            bh = max(3.0, val * chart_h)
            bx = start_x + s_idx * (bar_w + gap)
            by = chart_bottom - bh
            out.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" '
                f'height="{bh:.1f}" rx="4" fill="{color}"/>'
            )
            out.append(
                _text(
                    bx + bar_w / 2,
                    by - 6,
                    f"{val:.2f}",
                    fill="#e2e8f0",
                    size=10.5,
                    weight="600",
                    anchor="middle",
                )
            )
        center_x = chart_left + (g_idx + 0.5) * group_w
        out.append(
            _text(center_x, chart_bottom + 26, title, size=13, weight="600", anchor="middle")
        )
        abstain = float(metrics["abstention_accuracy"])
        out.append(
            _text(
                center_x,
                chart_bottom + 45,
                f"trap abstain: {abstain:.0%}",
                fill="#94a3b8",
                size=11,
                anchor="middle",
            )
        )

    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_chunk_sweep_svg(sweep: list[dict[str, float | int]]) -> str:
    width, height = 960, 460
    out = _svg_header(width, height, "Chunk size tradeoff curve")
    out.append(
        _text(
            36,
            42,
            "Chunk size tradeoff: ranking quality versus context token cost",
            size=20,
            weight="700",
        )
    )
    out.append(
        _text(
            36,
            64,
            "Flat sentence chunking sweep from 140 to 1400 max characters (overlap = 1, top k = 3)",
            fill="#94a3b8",
            size=13,
        )
    )

    legend = [
        ("NDCG@3 (left axis)", "#34d399"),
        ("Precision@3 (left axis)", "#38bdf8"),
        ("Avg context tokens (right axis)", "#f59e0b"),
    ]
    for idx, (label, color) in enumerate(legend):
        lx = 415 + idx * 178
        out.append(f'<circle cx="{lx}" cy="88" r="5" fill="{color}"/>')
        out.append(_text(lx + 12, 92, label, fill="#cbd5e1", size=12))

    left, top, bottom, right = 82, 118, 386, 874
    plot_w, plot_h = right - left, bottom - top
    max_tokens = max(float(row["avg_context_tokens"]) for row in sweep) * 1.15

    for step in range(6):
        val = step * 0.2
        y = bottom - val * plot_h
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#1e293b" stroke-width="1.2"/>'
        )
        out.append(_text(left - 12, y + 4, f"{val:.1f}", fill="#94a3b8", anchor="end"))
        out.append(_text(right + 12, y + 4, f"{round(val * max_tokens)}", fill="#f59e0b"))

    xs = [left + idx * (plot_w / (len(sweep) - 1)) for idx in range(len(sweep))]

    for idx, row in enumerate(sweep):
        if int(row["max_chars"]) == 340:
            sx = xs[idx]
            out.append(
                f'<rect x="{sx - 28:.1f}" y="{top}" width="56" height="{plot_h}" '
                'fill="#34d399" fill-opacity="0.08" stroke="#34d399" '
                'stroke-opacity="0.35" stroke-dasharray="4,4"/>'
            )
            out.append(
                _text(
                    sx,
                    top + 18,
                    "340 chars",
                    fill="#34d399",
                    size=11,
                    weight="600",
                    anchor="middle",
                )
            )

    def draw_curve(values: list[float], scale: float, color: str, fmt: str, dy: int) -> None:
        pts = [(xs[i], bottom - (values[i] / scale) * plot_h) for i in range(len(values))]
        poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        out.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.6"/>')
        for (px, py), raw in zip(pts, values, strict=True):
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}"/>')
            out.append(
                _text(px, py + dy, fmt.format(raw), fill="#e2e8f0", size=11, anchor="middle")
            )

    draw_curve([float(r["ndcg_at_3"]) for r in sweep], 1.0, "#34d399", "{:.2f}", -10)
    draw_curve([float(r["precision_at_3"]) for r in sweep], 1.0, "#38bdf8", "{:.2f}", 18)
    draw_curve(
        [float(r["avg_context_tokens"]) for r in sweep], max_tokens, "#f59e0b", "{:.0f}t", -10
    )

    for idx, row in enumerate(sweep):
        out.append(
            _text(
                xs[idx],
                bottom + 26,
                f"{row['max_chars']} chars",
                size=12.5,
                weight="600",
                anchor="middle",
            )
        )
        out.append(
            _text(
                xs[idx],
                bottom + 44,
                f"{row['chunks']} chunks",
                fill="#94a3b8",
                size=11,
                anchor="middle",
            )
        )

    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_latency_quality_svg(ablation: dict[str, Any]) -> str:
    width, height = 960, 440
    configs = ablation["configurations"]
    items = [
        ("bm25", "BM25", "#a78bfa"),
        ("dense", "Dense", "#38bdf8"),
        ("tfidf", "TFIDF", "#818cf8"),
        ("hybrid_rrf", "Hybrid RRF", "#f59e0b"),
        ("senior_full", "Senior Full", "#34d399"),
    ]

    out = _svg_header(width, height, "Latency versus verified answer rate")
    out.append(
        _text(
            36, 42, "Quality versus latency across pipeline configurations", size=20, weight="700"
        )
    )
    subtitle = (
        "End to end verified rate (grounded answers + trap abstentions) "
        "versus mean per query latency in pure Python"
    )
    out.append(_text(36, 64, subtitle, fill="#94a3b8", size=13))

    left, top, bottom, right = 88, 108, 376, 900
    plot_w, plot_h = right - left, bottom - top
    max_ms = max(max(float(configs[k]["mean_latency_ms"]) for k, _, _ in items) * 1.25, 10.0)
    min_ver = min(float(configs[k]["verified_rate"]) for k, _, _ in items)
    y_floor = min(0.60, max(0.0, int(min_ver * 10) / 10))
    y_span = 1.05 - y_floor

    for step in range(5):
        y_val = y_floor + step * ((1.0 - y_floor) / 4)
        y = bottom - ((y_val - y_floor) / y_span) * plot_h
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            'stroke="#1e293b" stroke-width="1.2"/>'
        )
        out.append(_text(left - 12, y + 4, f"{y_val:.2f}", fill="#94a3b8", anchor="end"))

    for step in range(6):
        ms_val = (max_ms / 5) * step
        x = left + (ms_val / max_ms) * plot_w
        out.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" '
            'stroke="#1e293b" stroke-width="1"/>'
        )
        out.append(_text(x, bottom + 24, f"{ms_val:.1f} ms", fill="#94a3b8", anchor="middle"))

    out.append(
        _text(
            (left + right) / 2,
            bottom + 50,
            "Mean query latency (milliseconds)",
            fill="#cbd5e1",
            size=12.5,
            weight="600",
            anchor="middle",
        )
    )

    offsets = {
        "bm25": (14, -12),
        "dense": (14, -12),
        "tfidf": (14, 18),
        "hybrid_rrf": (-14, 22),
        "senior_full": (-14, -14),
    }

    for key, label, color in items:
        ms = float(configs[key]["mean_latency_ms"])
        ver = float(configs[key]["verified_rate"])
        ndcg = float(configs[key]["ndcg_at_3"])
        cx = left + (ms / max_ms) * plot_w
        cy = bottom - ((ver - y_floor) / y_span) * plot_h
        dx, dy = offsets[key]
        anchor = "start" if dx > 0 else "end"
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{color}" '
            'stroke="#0f172a" stroke-width="2"/>'
        )
        caption = f"{label} ({ver:.0%} verified, NDCG {ndcg:.2f})"
        out.append(_text(cx + dx, cy + dy, caption, size=12.5, weight="700", anchor=anchor))

    out.append("</svg>")
    return "\n".join(out) + "\n"


def generate_all(root: Path | None = None) -> dict[str, Any]:
    base = root if root is not None else _project_root()
    from projects.incident_copilot.copilot import run_ablation_suite, run_chunk_sweep

    ablation = run_ablation_suite()
    sweep = run_chunk_sweep()
    payload = {"ablation": ablation, "chunk_sweep": sweep}

    results_path = base / "projects" / "incident_copilot" / "benchmark_results.json"
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    assets = base / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "architecture_overview.svg").write_text(render_architecture_svg(), encoding="utf-8")
    (assets / "retrieval_ablation.svg").write_text(render_ablation_svg(ablation), encoding="utf-8")
    (assets / "chunk_size_tradeoff.svg").write_text(render_chunk_sweep_svg(sweep), encoding="utf-8")
    (assets / "latency_vs_quality.svg").write_text(
        render_latency_quality_svg(ablation), encoding="utf-8"
    )
    return payload


def main() -> None:
    payload = generate_all(_project_root())
    senior = payload["ablation"]["configurations"]["senior_full"]
    print(
        f"Generated 4 SVG charts and benchmark_results.json "
        f"(senior_full NDCG@3={senior['ndcg_at_3']:.2f}, verified={senior['verified_rate']:.2f})"
    )


if __name__ == "__main__":
    main()
