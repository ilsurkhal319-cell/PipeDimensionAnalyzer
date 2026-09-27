from __future__ import annotations

import sys

import pymupdf

import analyze


def candidate(candidate_id: str, value: int, line: list[list[int]] | None = None) -> dict[str, object]:
    item: dict[str, object] = {
        "candidate_id": candidate_id,
        "value_mm": value,
        "bbox": {"x0": 10, "y0": 10, "x1": 20, "y1": 20},
    }
    if line is not None:
        item["dimension_line"] = {"a": line[0], "b": line[1]}
    return item


def dimension(candidate_id: str, value: int, role: str = "main") -> analyze.Dimension:
    return analyze.Dimension(
        segment_id=None,
        candidate_id=candidate_id,
        label=None,
        value_mm=value,
        role=role,
        included=role in {"main", "branch"},
        bbox=analyze.NormalizedBox(x0=0, y0=0, x1=1, y1=1),
        reason="test",
    )


def interpretation(dimensions: list[analyze.Dimension]) -> analyze.PageInterpretation:
    return analyze.PageInterpretation(line_id="TEST", dimensions=dimensions, status="ok", ambiguities=[])


def test_extract_numeric_candidates_from_pdf_text() -> None:
    document = pymupdf.open()
    page = document.new_page(width=200, height=100)
    page.insert_text((20, 30), "125")
    page.insert_text((50, 30), "DN50")

    candidates = analyze.extract_numeric_candidates(page)

    assert [(item["candidate_id"], item["value_mm"]) for item in candidates] == [("C1", 125)]


def test_validate_marks_missing_and_duplicate_candidates_for_review() -> None:
    result = interpretation([dimension("C1", 1), dimension("C1", 1)])

    checked = analyze.validate_dimensions(result, [candidate("C1", 250), candidate("C2", 300)])

    assert checked.status == "review"
    assert checked.dimensions[0].value_mm == 250
    assert checked.dimensions[1].role == "ambiguous"
    assert any("Duplicate candidate: C1" in item for item in checked.ambiguities)
    assert any("Missing candidate IDs: C2" in item for item in checked.ambiguities)


def test_merge_nested_uses_contained_dimension_geometry() -> None:
    candidates = [
        candidate("C1", 1000, [[0, 0], [100, 0]]),
        candidate("C2", 300, [[20, 0], [80, 0]]),
    ]
    topology = interpretation([dimension("C1", 1000), dimension("C2", 300)])
    nested = interpretation([dimension("C1", 1000), dimension("C2", 300)])

    merged = analyze.merge_nested_result(topology, nested, candidates)

    c2 = next(item for item in merged.dimensions if item.candidate_id == "C2")
    assert c2.role == "nested"
    assert c2.included is False
    assert "C1 fully covers" in c2.reason


def test_merge_nested_does_not_exclude_adjacent_segments() -> None:
    candidates = [
        candidate("C1", 500, [[0, 0], [50, 0]]),
        candidate("C2", 500, [[50, 0], [100, 0]]),
    ]
    topology = interpretation([dimension("C1", 500), dimension("C2", 500)])
    nested = interpretation([dimension("C1", 500), dimension("C2", 500, "nested")])

    merged = analyze.merge_nested_result(topology, nested, candidates)

    c2 = next(item for item in merged.dimensions if item.candidate_id == "C2")
    assert c2.role == "main"
    assert c2.included is True


def test_labels_and_formula_order_main_before_branch() -> None:
    result = interpretation([
        dimension("C1", 100, "branch"),
        dimension("C2", 200, "main"),
        dimension("C3", 50, "nested"),
    ])

    labeled = analyze.assign_display_labels(result)

    assert [item.label for item in labeled.dimensions] == ["L2", "L1", None]
    assert analyze.formula_for(labeled) == (300, "100 + 200")


def test_attach_cost_uses_configured_rates(monkeypatch) -> None:
    monkeypatch.setenv("VSEGPT_INPUT_RUB_PER_KTOK", "0.1")
    monkeypatch.setenv("VSEGPT_OUTPUT_RUB_PER_KTOK", "0.5")
    metrics: dict[str, int | float | str | None] = {
        "model": "vis-google/gemini-3-flash-pre",
        "input_tokens": 2000,
        "output_tokens": 1000,
    }

    analyze.attach_cost(metrics)

    assert metrics["cost_rub"] == 0.7


def test_combine_stage_metrics_sums_tokens_time_and_cost() -> None:
    first = {"model": "nested", "input_tokens": 10, "output_tokens": 5, "duration_seconds": 1.2, "cost_rub": 0.1}
    second = {"model": "branch", "input_tokens": 20, "output_tokens": 7, "duration_seconds": 2.3, "cost_rub": 0.2}

    combined = analyze.combine_stage_metrics(first, second)

    assert combined["total_tokens"] == 42
    assert combined["duration_seconds"] == 3.5
    assert combined["cost_rub"] == 0.3


def test_cli_parses_nested_options(monkeypatch, tmp_path) -> None:
    nested_results = tmp_path / "nested.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze.py",
            "drawing.pdf",
            "--start-page",
            "4",
            "--max-pages",
            "1",
            "--nested-model",
            "vis-google/gemini-3-flash-pre",
            "--nested-results",
            str(nested_results),
        ],
    )

    args = analyze.parse_args()

    assert args.pdf.name == "drawing.pdf"
    assert args.start_page == 4
    assert args.max_pages == 1
    assert args.nested_model == "vis-google/gemini-3-flash-pre"
    assert args.nested_results == nested_results
