import json
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"
RESULTS_PATH = PROJECT_ROOT / "projects" / "incident_copilot" / "benchmark_results.json"

EXPECTED_SVGS = (
    "architecture_overview.svg",
    "retrieval_ablation.svg",
    "chunk_size_tradeoff.svg",
    "latency_vs_quality.svg",
)


def test_benchmark_results_json_exists_and_has_ablation_and_sweep() -> None:
    assert RESULTS_PATH.is_file()
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assert "ablation" in payload
    assert "chunk_sweep" in payload
    assert "senior_full" in payload["ablation"]["configurations"]


def test_all_svg_charts_exist_parse_as_xml_and_avoid_em_dashes() -> None:
    for filename in EXPECTED_SVGS:
        svg_path = ASSETS_DIR / filename
        assert svg_path.is_file(), f"Missing SVG chart: {filename}"
        raw = svg_path.read_text(encoding="utf-8")
        assert "\u2013" not in raw
        assert "\u2014" not in raw
        root = ET.fromstring(raw)
        assert root.tag.endswith("svg")
