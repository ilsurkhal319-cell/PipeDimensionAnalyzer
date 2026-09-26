from __future__ import annotations

import csv
import json
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SUBMISSION = ROOT / "submission"

# Branch calls that were replaced by the final reruns of pages 3 and 4.
# They are kept here because the task requires the cost of repeated calls too.
REPEATED_CALLS = [
    {"page_number": 3, "stage": "branch", "duration_seconds": 8.180, "cost_rub": 1.477800},
    {"page_number": 4, "stage": "branch", "duration_seconds": 9.065, "cost_rub": 1.605450},
]


def read_first(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Empty or invalid JSON: {path}")
    return payload[0]


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    page_metrics: list[dict[str, object]] = []
    annotated = pymupdf.open()

    for page_number in range(1, 11):
        page_dir = OUTPUT / f"page{page_number}"
        nested_dir = OUTPUT / f"page{page_number}_nested_cache"
        result_path = page_dir / "results.json"
        branch_metrics_path = page_dir / "metrics.json"
        nested_metrics_path = nested_dir / "metrics.json"
        annotated_path = page_dir / "annotated.pdf"
        required = [result_path, branch_metrics_path, nested_metrics_path, annotated_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing result files: " + ", ".join(missing))

        result = read_first(result_path)
        interpretation = result["interpretation"]
        branch_metrics = read_first(branch_metrics_path)
        nested_metrics = read_first(nested_metrics_path)
        duration = float(branch_metrics.get("duration_seconds") or 0) + float(
            nested_metrics.get("duration_seconds") or 0
        )
        cost = float(branch_metrics.get("cost_rub") or 0) + float(
            nested_metrics.get("cost_rub") or 0
        )
        ambiguities = interpretation.get("ambiguities") or []
        total_mm = int(result.get("total_length_mm") or 0)
        rows.append(
            {
                "page_number": page_number,
                "line_id": interpretation.get("line_id") or "",
                "length_mm": total_mm,
                "length_m": round(total_mm / 1000, 3),
                "formula": result.get("formula") or "",
                "status": interpretation.get("status") or "review",
                "remarks": "; ".join(str(item) for item in ambiguities),
            }
        )
        page_metrics.append(
            {
                "page_number": page_number,
                "model": branch_metrics.get("model"),
                "ai_calls": 2,
                "duration_seconds": round(duration, 3),
                "cost_rub": round(cost, 6),
            }
        )
        with pymupdf.open(annotated_path) as source:
            annotated.insert_pdf(source)

    with (SUBMISSION / "results_10_sheets.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    annotated.save(SUBMISSION / "annotated_10_sheets.pdf", garbage=4, deflate=True)

    total_duration = sum(float(item["duration_seconds"]) for item in page_metrics) + sum(
        float(item["duration_seconds"]) for item in REPEATED_CALLS
    )
    total_cost = sum(float(item["cost_rub"]) for item in page_metrics) + sum(
        float(item["cost_rub"]) for item in REPEATED_CALLS
    )
    total_calls = sum(int(item["ai_calls"]) for item in page_metrics) + len(REPEATED_CALLS)
    summary = {
        "pages": 10,
        "models": sorted({str(item["model"]) for item in page_metrics}),
        "ocr_service": None,
        "text_extraction": "PyMuPDF embedded text",
        "ai_calls_total": total_calls,
        "ai_calls_per_page": round(total_calls / 10, 3),
        "duration_seconds_total": round(total_duration, 3),
        "duration_seconds_per_page": round(total_duration / 10, 3),
        "cost_rub_total": round(total_cost, 6),
        "cost_rub_per_page": round(total_cost / 10, 6),
        "rates": {
            "input_rub_per_1000_tokens": 0.15,
            "output_rub_per_1000_tokens": 0.90,
        },
        "repeated_calls": REPEATED_CALLS,
        "pages_detail": page_metrics,
    }
    (SUBMISSION / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Submission files saved to {SUBMISSION}")


if __name__ == "__main__":
    main()
