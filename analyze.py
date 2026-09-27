from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pymupdf

from pipe_dimension.domain import (
    assign_display_labels,
    attach_cost,
    combine_stage_metrics,
    formula_for,
    merge_nested_result,
)
from pipe_dimension.domain import validate_dimensions as _validate_dimensions
from pipe_dimension.geometry import (
    extract_numeric_candidates,
    mark_candidates_on_image,
    render_page,
)
from pipe_dimension.models import Dimension, NormalizedBox, PageInterpretation
from pipe_dimension.prompts import NESTED_PROMPT
from pipe_dimension.rendering import annotate_page
from pipe_dimension.vision import interpret_page

validate_dimensions = _validate_dimensions


def load_dotenv_file() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv_file()


def analyze(
    pdf_path: Path,
    output_dir: Path,
    model: str,
    dpi: int,
    max_pages: int | None = None,
    start_page: int = 1,
    expected_mm: int | None = None,
    nested_model: str | None = None,
    nested_only: bool = False,
    nested_results: Path | None = None,
) -> None:
    from openai import OpenAI

    output_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.vsegpt.ru/v1"),
    )
    document = pymupdf.open(pdf_path)
    start_index = max(0, start_page - 1)
    if start_index >= len(document):
        raise ValueError(f"start page {start_page} is outside the PDF")
    end_index = len(document) if max_pages is None else min(start_index + max_pages, len(document))
    document.select(range(start_index, end_index))
    candidate_document = pymupdf.open()

    rows: list[dict[str, object]] = []
    page_payloads: list[dict[str, object]] = []
    metrics_payloads: list[dict[str, object]] = []
    cached_nested = {}
    if nested_results:
        cached_nested = {
            int(item["page_number"]): PageInterpretation.model_validate(item["interpretation"])
            for item in json.loads(nested_results.read_text(encoding="utf-8"))
        }

    pages = list(document)
    for page_index, page in enumerate(pages):
        page_number = start_page + page_index
        print(f"Page {page_number}/{len(pages)}: rendering")
        png = render_page(page, dpi=dpi)
        candidates = extract_numeric_candidates(page)
        marked_png = mark_candidates_on_image(png, candidates)
        probe_script = Path(__file__).resolve().with_name("dimension_probe.py")
        geometry_dir = output_dir / "dimension_probe"
        if probe_script.exists():
            probe_env = os.environ.copy()
            probe_env["DIMENSION_PAGE_INDEX"] = str(page_number - 1)
            probe_env["DIMENSION_PDF_PATH"] = str(pdf_path.resolve())
            probe_env["DIMENSION_PROBE_OUTPUT_DIR"] = str(geometry_dir.resolve())
            subprocess.run(
                [sys.executable, str(probe_script)],
                check=True,
                env=probe_env,
                capture_output=True,
                text=True,
            )
        geometry_items: list[dict[str, object]] = []
        geometry_json = geometry_dir / f"dimension_probe_page{page_number}.json"
        geometry_png = geometry_dir / f"dimension_probe_page{page_number}.png"
        if geometry_json.exists() and geometry_png.exists():
            geometry_items = json.loads(geometry_json.read_text(encoding="utf-8"))
            allowed_ids = {item.get("candidate_id") for item in geometry_items if item.get("candidate_id")}
            candidates = [item for item in candidates if item["candidate_id"] in allowed_ids]
            geometry_by_id = {
                str(item["candidate_id"]): item
                for item in geometry_items
                if item.get("candidate_id") and item.get("a") and item.get("b")
            }
            for candidate in candidates:
                geometry = geometry_by_id.get(str(candidate["candidate_id"]))
                if geometry is None:
                    continue
                candidate["dimension_line"] = {
                    "a": [
                        round(float(geometry["a"][0]) / page.rect.width * 1000),
                        round(float(geometry["a"][1]) / page.rect.height * 1000),
                    ],
                    "b": [
                        round(float(geometry["b"][0]) / page.rect.width * 1000),
                        round(float(geometry["b"][1]) / page.rect.height * 1000),
                    ],
                    "mode": geometry.get("mode"),
                }
            marked_png = geometry_png.read_bytes()
            print(f"Page {page_number}: using geometry markup ({len(candidates)} C-ID candidates)")
        candidate_page = candidate_document.new_page(
            width=page.rect.width, height=page.rect.height
        )
        candidate_page.insert_image(candidate_page.rect, stream=marked_png)

        nested_result = cached_nested.get(page_number)
        nested_metrics = None
        if nested_result is None and nested_model:
            print(f"Page {page_number}/{len(pages)}: checking nested dimensions with {nested_model}")
            nested_result, nested_metrics = interpret_page(
                client, nested_model, page_number, marked_png, candidates,
                system_prompt=NESTED_PROMPT,
            )
            attach_cost(nested_metrics, prefix="VSEGPT_NESTED")
        nested_ids = {
            str(item.candidate_id) for item in (nested_result.dimensions if nested_result else [])
            if item.role == "nested" and item.candidate_id
        }
        geometry_nested = {
            str(item["candidate_id"]): str(item.get("covers") or "another dimension")
            for item in geometry_items
            if item.get("nested") and item.get("candidate_id")
        }
        nested_ids.update(geometry_nested)
        branch_candidates = [c for c in candidates if str(c["candidate_id"]) not in nested_ids]
        branch_marked_png = marked_png
        if probe_script.exists():
            branch_env = os.environ.copy()
            branch_env["DIMENSION_PAGE_INDEX"] = str(page_number - 1)
            branch_env["DIMENSION_PDF_PATH"] = str(pdf_path.resolve())
            branch_env["DIMENSION_PROBE_OUTPUT_DIR"] = str(geometry_dir.resolve())
            branch_env["DIMENSION_EXCLUDE_IDS"] = ",".join(sorted(nested_ids))
            branch_env["DIMENSION_OUTPUT_SUFFIX"] = "_branch"
            branch_env["DIMENSION_HIDE_CANDIDATE_LEADERS"] = "1"
            subprocess.run(
                [sys.executable, str(probe_script)], check=True, env=branch_env,
                capture_output=True, text=True,
            )
            branch_png_path = geometry_dir / f"dimension_probe_page{page_number}_branch.png"
            if branch_png_path.exists():
                branch_marked_png = branch_png_path.read_bytes()
                print(f"Page {page_number}: removed {len(nested_ids)} nested candidates from branch markup")

        if nested_only:
            result = PageInterpretation(
                line_id=None,
                dimensions=[Dimension(
                    segment_id=None, candidate_id=str(c["candidate_id"]), label=None,
                    value_mm=int(c["value_mm"]), role="main", included=True,
                    bbox=NormalizedBox.model_validate(c["bbox"]),
                    reason="Baseline geometry candidate; branch analysis disabled.",
                ) for c in candidates],
                status="ok", ambiguities=[],
            )
            proposal_metrics = {"model": "disabled", "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "duration_seconds": 0}
        else:
            print(f"Page {page_number}/{len(pages)}: calling {model} with {len(branch_candidates)} non-nested candidates")
            result, proposal_metrics = interpret_page(
                client, model, page_number, branch_marked_png, branch_candidates
            )
        if nested_result is not None:
            result = merge_nested_result(result, nested_result, candidates)
        result_by_id = {str(item.candidate_id): item for item in result.dimensions}
        candidate_by_id = {str(item["candidate_id"]): item for item in candidates}
        for candidate_id, cover_id in geometry_nested.items():
            dimension = result_by_id.get(candidate_id)
            if dimension is None:
                candidate = candidate_by_id[candidate_id]
                dimension = Dimension(
                    segment_id=None,
                    candidate_id=candidate_id,
                    label=None,
                    value_mm=int(candidate["value_mm"]),
                    role="nested",
                    included=False,
                    bbox=NormalizedBox.model_validate(candidate["bbox"]),
                    reason=f"{cover_id} fully covers this physical interval.",
                )
                result.dimensions.append(dimension)
            dimension.role = "nested"
            dimension.included = False
            dimension.reason = f"{cover_id} fully covers this physical interval."
        leader_ids = {
            item.get("candidate_id")
            for item in geometry_items
            if item.get("mode") == "leader" and item.get("candidate_id")
        }
        for dimension in result.dimensions:
            if dimension.candidate_id in leader_ids:
                dimension.included = True
                if dimension.role == "ambiguous":
                    dimension.role = "main"
                    dimension.reason = "Geometry markup confirms the dimension; branch topology was not proven, so main is used by default."
                elif dimension.role not in {"main", "branch"}:
                    dimension.role = "main"
                    dimension.reason = "Geometry markup confirms the dimension; non-topological role replaced with main."
                else:
                    dimension.reason = dimension.reason.rstrip(".") + "; geometry markup confirms the dimension line"
        attach_cost(proposal_metrics)
        result = assign_display_labels(result)
        metrics = combine_stage_metrics(proposal_metrics, None)
        if nested_metrics is not None:
            metrics = combine_stage_metrics(metrics, nested_metrics)

        total_mm, formula = formula_for(result)
        if expected_mm is not None:
            verdict = "PASS" if total_mm == expected_mm else "FAIL"
            print(f"Page {page_number}: expected {expected_mm} mm, got {total_mm} mm [{verdict}]")
        total_m = total_mm / 1000 if total_mm is not None else None
        remarks = "; ".join(result.ambiguities)

        annotate_page(page, result)
        rows.append(
            {
                "page_number": page_number,
                "line_id": result.line_id or "",
                "length_mm": total_mm if total_mm is not None else "",
                "length_m": total_m if total_m is not None else "",
                "formula": formula or "",
                "status": result.status,
                "remarks": remarks,
            }
        )
        page_payloads.append(
            {
                "page_number": page_number,
                "total_length_mm": total_mm,
                "formula": formula,
                "interpretation": result.model_dump(),
            }
        )
        metrics_payloads.append(metrics)

    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "results_all.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "results.json").write_text(
        json.dumps(page_payloads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payloads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    document.save(output_dir / "annotated.pdf", garbage=4, deflate=True)
    candidate_document.save(output_dir / "candidates.pdf", garbage=4, deflate=True)

    print(f"Done: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate pipeline lengths from isometric PDF")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "vis-google/gemini-3-flash-pre"),
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--expected-mm", type=int, default=None)
    parser.add_argument(
        "--nested-model",
        default=None,
        help="Optional separate vision model used only for nested dimensions",
    )
    parser.add_argument("--nested-only", action="store_true", help="Disable topology model and test nested only")
    parser.add_argument("--nested-results", type=Path, default=None,
                        help="Reuse nested-only results.json and hide nested C-IDs from branch model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        args.pdf,
        args.output,
        args.model,
        args.dpi,
        args.max_pages,
        args.start_page,
        args.expected_mm,
        args.nested_model or (args.model if args.nested_only else None),
        args.nested_only,
        args.nested_results,
    )


if __name__ == "__main__":
    main()
