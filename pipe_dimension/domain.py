from __future__ import annotations

import os

from .models import NormalizedBox, PageInterpretation


def validate_dimensions(
    result: PageInterpretation, candidates: list[dict[str, object]]
) -> PageInterpretation:
    by_id = {str(item["candidate_id"]): item for item in candidates}
    by_value: dict[int, list[dict[str, object]]] = {}
    for item in candidates:
        by_value.setdefault(int(item["value_mm"]), []).append(item)
    used: set[str] = set()
    validation_errors: list[str] = []

    for dimension in result.dimensions:
        candidate = by_id.get(dimension.candidate_id or "")
        if candidate is None:
            same_value = by_value.get(dimension.value_mm, [])
            if len(same_value) == 1:
                candidate = same_value[0]
                dimension.candidate_id = str(candidate["candidate_id"])
        if candidate is None:
            dimension.included = False
            dimension.role = "ambiguous"
            validation_errors.append(f"Unknown candidate: {dimension.candidate_id}")
            continue

        dimension.value_mm = int(candidate["value_mm"])
        dimension.bbox = NormalizedBox.model_validate(candidate["bbox"])
        if dimension.candidate_id in used:
            dimension.included = False
            dimension.role = "ambiguous"
            validation_errors.append(f"Duplicate candidate: {dimension.candidate_id}")
        used.add(dimension.candidate_id)

        if dimension.role not in {"main", "branch"}:
            dimension.included = False

    missing_ids = sorted(set(by_id) - used, key=lambda value: int(value[1:]))
    if missing_ids:
        result.status = "review"
        validation_errors.append("Missing candidate IDs: " + ", ".join(missing_ids))

    if validation_errors:
        result.status = "review"
        result.ambiguities.extend(validation_errors)
    return result


def merge_nested_result(
    topology: PageInterpretation,
    nested: PageInterpretation,
    candidates: list[dict[str, object]],
) -> PageInterpretation:
    """Apply nested decisions while retaining main/branch decisions."""
    geometry = {
        str(item["candidate_id"]): item.get("dimension_line")
        for item in candidates
        if item.get("candidate_id") and item.get("dimension_line")
    }

    def point_distance(a: list[float], b: list[float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def segment_contains(outer: list[list[float]], inner: list[list[float]]) -> bool:
        ax, ay = outer[1][0] - outer[0][0], outer[1][1] - outer[0][1]
        length_sq = ax * ax + ay * ay
        if length_sq == 0:
            return False
        projections = [
            ((point[0] - outer[0][0]) * ax + (point[1] - outer[0][1]) * ay) / length_sq
            for point in inner
        ]
        cross = [
            abs(ax * (point[1] - outer[0][1]) - ay * (point[0] - outer[0][0]))
            / max(length_sq**0.5, 1.0)
            for point in inner
        ]
        return max(cross) <= 30 and min(projections) >= -0.10 and max(projections) <= 1.10

    def is_only_adjacent(candidate_id: str) -> bool:
        line = geometry.get(candidate_id)
        if not line:
            return False
        current = [line["a"], line["b"]]
        for other_id, other_line in geometry.items():
            if other_id == candidate_id or not other_line:
                continue
            other = [other_line["a"], other_line["b"]]
            shared = any(point_distance(a, b) <= 8 for a in current for b in other)
            if shared and not segment_contains(current, other) and not segment_contains(other, current):
                return True
        return False

    auto_nested: dict[str, str] = {}
    for inner_id, inner_line in geometry.items():
        if not inner_line:
            continue
        inner = [inner_line["a"], inner_line["b"]]
        inner_length = point_distance(inner[0], inner[1])
        for outer_id, outer_line in geometry.items():
            if inner_id == outer_id or not outer_line:
                continue
            outer = [outer_line["a"], outer_line["b"]]
            outer_length = point_distance(outer[0], outer[1])
            if outer_length > inner_length * 1.05 and segment_contains(outer, inner):
                auto_nested[inner_id] = outer_id
                break

    nested_ids = {
        item.candidate_id
        for item in nested.dimensions
        if item.role == "nested" and not is_only_adjacent(item.candidate_id)
    }
    nested_ids.update(auto_nested)
    nested_reasons = {
        item.candidate_id: item.reason for item in nested.dimensions if item.role == "nested"
    }
    present_ids = {item.candidate_id for item in topology.dimensions}
    for item in nested.dimensions:
        if item.candidate_id in nested_ids and item.candidate_id not in present_ids:
            topology.dimensions.append(item.model_copy(deep=True))
            present_ids.add(item.candidate_id)
    for item in topology.dimensions:
        if item.candidate_id in nested_ids:
            item.role = "nested"
            item.included = False
            item.reason = (
                nested_reasons.get(item.candidate_id)
                or (
                    f"{auto_nested[item.candidate_id]} fully covers this physical interval."
                    if item.candidate_id in auto_nested
                    else None
                )
                or "Nested dimension excluded by the independent geometry pass."
            )
        elif item.role == "irrelevant":
            item.role = "main"
            item.included = True
            item.reason = "Valid geometry candidate; retained because irrelevant is not allowed."
    topology.status = "review" if nested.status == "review" else topology.status
    topology.ambiguities.extend(
        message for message in nested.ambiguities if message not in topology.ambiguities
    )
    return assign_display_labels(validate_dimensions(topology, candidates))


def combine_stage_metrics(
    first: dict[str, int | float | str | None],
    second: dict[str, int | float | str | None] | None,
) -> dict[str, object]:
    if second is None:
        return {"proposal": first, "review": None, **first}
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in (first, second))
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in (first, second))
    first_cost = first.get("cost_rub")
    second_cost = second.get("cost_rub")
    total_cost = (
        round(float(first_cost) + float(second_cost), 6)
        if isinstance(first_cost, (int, float)) and isinstance(second_cost, (int, float))
        else None
    )
    return {
        "proposal": first,
        "review": second,
        "model": f"{first['model']} -> {second['model']}",
        "duration_seconds": round(
            float(first["duration_seconds"]) + float(second["duration_seconds"]), 3
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_rub": total_cost,
        "cost_basis": "VseGPT RUB per 1000 usage tokens" if total_cost is not None else None,
    }


def assign_display_labels(result: PageInterpretation) -> PageInterpretation:
    ordered = [
        item
        for role in ("main", "branch")
        for item in result.dimensions
        if item.included and item.role == role
    ]
    for item in result.dimensions:
        item.label = None
    for index, item in enumerate(ordered, start=1):
        item.label = f"L{index}"
    return result


def formula_for(result: PageInterpretation) -> tuple[int | None, str | None]:
    included = [
        item.value_mm
        for item in result.dimensions
        if item.included and item.role in {"main", "branch"}
    ]
    if not included:
        return None, None
    return sum(included), " + ".join(str(value) for value in included)


def optional_float(name: str) -> float | None:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else None


def attach_cost(
    metrics: dict[str, int | float | str | None], prefix: str = "VSEGPT"
) -> None:
    input_rate = optional_float(f"{prefix}_INPUT_RUB_PER_KTOK")
    output_rate = optional_float(f"{prefix}_OUTPUT_RUB_PER_KTOK")
    model_name = str(metrics.get("model") or "").lower()
    if input_rate is None or output_rate is None:
        if "gemini-3-flash" in model_name:
            input_rate, output_rate = 0.15, 0.90
        elif "gemini-2.5-flash" in model_name:
            input_rate, output_rate = 0.09, 0.75
    input_tokens = metrics.get("input_tokens")
    output_tokens = metrics.get("output_tokens")
    if input_rate is None or output_rate is None:
        metrics["cost_rub"] = None
        return
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        metrics["cost_rub"] = None
        return
    metrics["cost_rub"] = round(
        input_tokens / 1000 * input_rate + output_tokens / 1000 * output_rate, 6
    )
    metrics["cost_basis"] = "VseGPT RUB per 1000 usage tokens (model default rate if not configured)"
