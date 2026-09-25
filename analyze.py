from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pymupdf
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openai import OpenAI


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


class NormalizedBox(BaseModel):
    """Rectangle in page coordinates normalized to the 0..1000 range."""

    x0: int = Field(ge=0, le=1000)
    y0: int = Field(ge=0, le=1000)
    x1: int = Field(ge=0, le=1000)
    y1: int = Field(ge=0, le=1000)


class Dimension(BaseModel):
    segment_id: str | None = Field(
        default=None,
        description="Physical pipe segment identifier in S<number> format, for example S3",
    )
    candidate_id: str | None = Field(
        default=None,
        description="PDF candidate identifier in C<number> format, for example C17",
    )
    label: str | None
    value_mm: int = Field(gt=0)
    role: Literal["main", "branch", "nested", "irrelevant", "ambiguous"]
    included: bool
    bbox: NormalizedBox
    reason: str


class PageInterpretation(BaseModel):
    line_id: str | None
    dimensions: list[Dimension]
    status: Literal["ok", "review"]
    ambiguities: list[str]



SYSTEM_PROMPT = """
Геометрический этап уже выполнил поиск размеров и удалил служебные числа. Поэтому не ищи новые
кандидаты и не исключай переданные C-ID как служебные. Твоя задача — определить только main/branch.

Входные данные:
- Геометрическая разметка уже отфильтровала служебные числа. Каждый переданный C-ID является
  действительным линейным размером вдоль трассы и должен быть рассмотрен.
- Красная линия показывает размерный отрезок, зелёная — выноску от числа к этому отрезку.
  Зеленую выноску отдельно не складывай. Используй только переданные C-ID.
- Поле dimension_line содержит нормализованные координаты концов красного размерного отрезка.
  Используй их только для понимания, к какому участку относится размер.

ОСНОВНЫЕ ПРАВИЛА:

branch:
Тройник или узел существует только там, где синяя стрелка заканчивается на пересечении
реальных осевых линий труб. Другие места не анализируй.

Синяя стрелка указывает на узел, а не на конкретный выход.

Если в синем обозначении один X, в узле сходятся три осевых выхода:
два продолжают main-трассу, один является branch.

Если в синем обозначении два X, в узле сходятся четыре осевых выхода:
две пары продолжают main-трассы, два оставшихся выхода являются branch.

Определи main по непрерывному продолжению осевых линий через узел. Оставшиеся выходы
назначь branch. Не выбирай роли по длине, углу или положению числа.
Если нет синей линии, указывающей на узел, branch начинаться не может.

Для каждого кандидата верни candidate_id, segment_id, value_mm, role и included. У всех размеров
Не вычисляй сумму: это сделает программа.
При сомнении поставь status=review, перечисли его в ambiguities и всё равно верни лучший
непротиворечивый вариант. Верни только структурированный объект без Markdown.
""".strip()

NESTED_PROMPT = """
Ты определяешь только вложенность размерных отрезков на размеченном чертеже.
Не определяй main/branch и не меняй роль размера по топологии труб.

Сначала включи все переданные размеры. Пометь размер nested только если другой размер
относится к тому же физическому участку осевой трассы и полностью покрывает его границы.
Одна общая конечная точка, параллельность, близость, меньшее число или визуальное
пересечение не доказывают вложенность. Если границы нельзя уверенно сопоставить по
геометрии, оставь оба размера включёнными.

Для nested укажи в reason покрывающий C-ID. Верни все C-ID ровно один раз, сохрани их
значения и bbox, а роль у вложенных размеров укажи nested, у остальных — main.
Верни только структурированный объект без Markdown.
""".strip()




COLORS = {
    "main": (0.0, 0.48, 0.53),
    "branch": (0.9, 0.28, 0.02),
    "nested": (0.55, 0.18, 0.72),
    "irrelevant": (0.45, 0.45, 0.45),
    "ambiguous": (0.9, 0.0, 0.0),
}


def render_page(page: pymupdf.Page, dpi: int) -> bytes:
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return pixmap.tobytes("png")


def as_data_url(png: bytes) -> str:
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def mark_candidates_on_image(
    png: bytes, candidates: list[dict[str, object]]
) -> bytes:
    image = Image.open(BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(10, image.width // 150))
    occupied_labels: list[tuple[int, int, int, int]] = []

    for candidate in candidates:
        box = candidate["bbox"]
        x0 = int(box["x0"] / 1000 * image.width)
        y0 = int(box["y0"] / 1000 * image.height)
        x1 = int(box["x1"] / 1000 * image.width)
        y1 = int(box["y1"] / 1000 * image.height)
        color = (0, 85, 200)
        draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=color, width=2)

        label = str(candidate["candidate_id"])
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0] + 10
        label_height = label_box[3] - label_box[1] + 12
        label_x = min(image.width - label_width - 2, x1 + 8)
        label_y = max(0, y0 - label_height)
        while any(
            label_x < ox + ow
            and label_x + label_width > ox
            and label_y < oy + oh
            and label_y + label_height > oy
            for ox, oy, ow, oh in occupied_labels
        ):
            label_y = max(0, label_y - label_height - 4)
            if label_y == 0:
                break
        occupied_labels.append((label_x, label_y, label_width, label_height))
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=(255, 255, 210),
            outline=color,
        )
        draw.text((label_x + 5, label_y + 5), label, fill=color, font=font)

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def extract_numeric_candidates(page: pymupdf.Page) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    page_width, page_height = page.rect.width, page.rect.height
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not re.fullmatch(r"\d+", text) or int(text) < 1:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                candidates.append(
                    {
                        "candidate_id": f"C{len(candidates) + 1}",
                        "value_mm": int(text),
                        "bbox": {
                            "x0": round(x0 / page_width * 1000),
                            "y0": round(y0 / page_height * 1000),
                            "x1": round(x1 / page_width * 1000),
                            "y1": round(y1 / page_height * 1000),
                        },
                    }
                )
    return candidates


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
        validation_errors.append(
            "Missing candidate IDs: " + ", ".join(missing_ids)
        )

    if validation_errors:
        result.status = "review"
        result.ambiguities.extend(validation_errors)
    return result


def interpret_page(
    client: OpenAI,
    model: str,
    page_number: int,
    original_png: bytes,
    marked_png: bytes,
    candidates: list[dict[str, object]],
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[PageInterpretation, dict[str, int | float | str | None]]:
    started = time.perf_counter()
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Analyze marked PDF page {page_number}. Numeric candidates:\n"
                            + json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
                            + "\nЕдинственное изображение — полный чертёж с геометрической разметкой: "
                            + "красные D-линии и зелёные выноски. "
                            + "Используй только C-ID из списка кандидатов."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": as_data_url(marked_png), "detail": "high"},
                    },
                ],
            },
        ],
        response_format=PageInterpretation,
        reasoning_effort="low",
        temperature=0,
        max_tokens=8192,
    )
    elapsed = time.perf_counter() - started

    result = response.choices[0].message.parsed
    if result is None:
        raise RuntimeError("The model did not return a parsed PageInterpretation")
    result = assign_display_labels(validate_dimensions(result, candidates))

    usage = response.usage
    metrics: dict[str, int | float | str | None] = {
        "page_number": page_number,
        "model": model,
        "duration_seconds": round(elapsed, 3),
        "input_tokens": getattr(usage, "input_tokens", None)
        or getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return result, metrics


def merge_nested_result(
    topology: PageInterpretation,
    nested: PageInterpretation,
    candidates: list[dict[str, object]],
) -> PageInterpretation:
    """Apply only nested decisions; keep main/branch from the topology pass."""
    nested_ids = {
        item.candidate_id
        for item in nested.dimensions
        if item.role == "nested"
    }
    nested_reasons = {
        item.candidate_id: item.reason
        for item in nested.dimensions
        if item.role == "nested"
    }
    for item in topology.dimensions:
        if item.candidate_id in nested_ids:
            item.role = "nested"
            item.included = False
            item.reason = nested_reasons.get(item.candidate_id) or "Nested dimension excluded by the independent geometry pass."
    topology.status = "review" if nested.status == "review" else topology.status
    topology.ambiguities.extend(
        message for message in nested.ambiguities
        if message not in topology.ambiguities
    )
    return assign_display_labels(validate_dimensions(topology, candidates))


def review_page(
    client: OpenAI,
    model: str,
    page_number: int,
    original_png: bytes,
    marked_png: bytes,
    candidates: list[dict[str, object]],
    proposal: PageInterpretation,
) -> tuple[PageInterpretation, dict[str, int | float | str | None]]:
    started = time.perf_counter()
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": REVIEW_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Проверь лист {page_number}. Кандидаты:\n"
                            + json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
                            + "\nПредварительный ответ:\n"
                            + proposal.model_dump_json()
                            + "\nЕдинственное изображение — полный чертёж с геометрической разметкой; используй его для проверки."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": as_data_url(marked_png), "detail": "high"}},
                ],
            },
        ],
        response_format=PageInterpretation,
        reasoning_effort="low",
        temperature=0,
        max_tokens=8192,
    )
    elapsed = time.perf_counter() - started
    result = response.choices[0].message.parsed
    if result is None:
        raise RuntimeError("The review model did not return a parsed PageInterpretation")
    result = assign_display_labels(validate_dimensions(result, candidates))
    usage = response.usage
    return result, {
        "model": model,
        "duration_seconds": round(elapsed, 3),
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "total_tokens": usage.total_tokens if usage else None,
    }


def combine_stage_metrics(
    proposal: dict[str, int | float | str | None],
    review: dict[str, int | float | str | None] | None,
) -> dict[str, object]:
    if review is None:
        return {"proposal": proposal, "review": None, **proposal}
    input_tokens = sum(int(item.get("input_tokens") or 0) for item in (proposal, review))
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in (proposal, review))
    proposal_cost = proposal.get("cost_rub")
    review_cost = review.get("cost_rub")
    total_cost = (
        round(float(proposal_cost) + float(review_cost), 6)
        if isinstance(proposal_cost, (int, float)) and isinstance(review_cost, (int, float))
        else None
    )
    return {
        "proposal": proposal,
        "review": review,
        "model": f"{proposal['model']} -> {review['model']}",
        "duration_seconds": round(float(proposal["duration_seconds"]) + float(review["duration_seconds"]), 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_rub": total_cost,
        "cost_basis": "VseGPT RUB per 1000 usage tokens" if total_cost is not None else None,
    }


def normalized_rect(page: pymupdf.Page, box: NormalizedBox) -> pymupdf.Rect:
    width = page.rect.width
    height = page.rect.height
    return pymupdf.Rect(
        box.x0 / 1000 * width,
        box.y0 / 1000 * height,
        box.x1 / 1000 * width,
        box.y1 / 1000 * height,
    )


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


def annotate_page(page: pymupdf.Page, result: PageInterpretation) -> None:
    for dimension in result.dimensions:
        if dimension.role == "irrelevant":
            continue
        rect = normalized_rect(page, dimension.bbox)
        color = COLORS[dimension.role]
        page.draw_rect(rect, color=color, width=1.2, overlay=True)

        if not dimension.included or not dimension.label:
            continue

        label_height = max(12.0, rect.height + 4.0)
        label_width = max(24.0, 12.0 + 5.0 * len(dimension.label))
        label_x0 = min(rect.x1 + 2, page.rect.width - label_width)
        label_y0 = max(0.0, min(rect.y0, page.rect.height - label_height))
        label_rect = pymupdf.Rect(
            label_x0,
            label_y0,
            label_x0 + label_width,
            label_y0 + label_height,
        )
        page.draw_rect(
            label_rect,
            color=color,
            fill=(1.0, 1.0, 1.0),
            width=1.2,
            overlay=True,
        )
        page.insert_textbox(
            label_rect,
            dimension.label,
            fontsize=7,
            fontname="helv",
            color=color,
            align=pymupdf.TEXT_ALIGN_CENTER,
            overlay=True,
        )


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
        input_tokens / 1000 * input_rate
        + output_tokens / 1000 * output_rate,
        6,
    )
    metrics["cost_basis"] = "VseGPT RUB per 1000 usage tokens (model default rate if not configured)"


def analyze(
    pdf_path: Path,
    output_dir: Path,
    model: str,
    dpi: int,
    max_pages: int | None = None,
    review_model: str | None = None,
    start_page: int = 1,
    expected_mm: int | None = None,
    nested_model: str | None = None,
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

    pages = list(document)
    for page_index, page in enumerate(pages):
        page_number = start_page + page_index
        print(f"Page {page_number}/{len(pages)}: rendering")
        png = render_page(page, dpi=dpi)
        candidates = extract_numeric_candidates(page)
        marked_png = mark_candidates_on_image(png, candidates)
        probe_script = Path("dimension_probe.py")
        if probe_script.exists():
            probe_env = os.environ.copy()
            probe_env["DIMENSION_PAGE_INDEX"] = str(page_number - 1)
            subprocess.run(
                [sys.executable, str(probe_script)],
                check=True,
                env=probe_env,
                capture_output=True,
                text=True,
            )
        geometry_items: list[dict[str, object]] = []
        geometry_dir = Path("output") / "dimension_probe"
        if not geometry_dir.exists():
            geometry_dir = Path("output")
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

        print(f"Page {page_number}/{len(pages)}: calling {model}")
        result, proposal_metrics = interpret_page(
            client, model, page_number, png, marked_png, candidates
        )
        nested_metrics = None
        if nested_model:
            print(f"Page {page_number}/{len(pages)}: checking nested dimensions with {nested_model}")
            nested_result, nested_metrics = interpret_page(
                client, nested_model, page_number, png, marked_png, candidates,
                system_prompt=NESTED_PROMPT,
            )
            attach_cost(nested_metrics, prefix="VSEGPT_NESTED")
            result = merge_nested_result(result, nested_result, candidates)
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
        review_metrics = None
        review_all = os.getenv("OPENAI_REVIEW_ALL", "0").lower() in {"1", "true", "yes"}
        if review_model and (review_all or result.status == "review"):
            print(f"Page {page_number}/{len(pages)}: reviewing with {review_model}")
            result, review_metrics = review_page(
                client, review_model, page_number, png, marked_png, candidates, result
            )
            attach_cost(review_metrics, prefix="VSEGPT_REVIEW")
        # Final roles may be changed by geometry correction or review; labels must
        # be assigned only after all role decisions are complete.
        result = assign_display_labels(result)
        metrics = combine_stage_metrics(proposal_metrics, review_metrics)
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

    # Единый сводный файл по всем обработанным листам. При запуске без
    # --max-pages он содержит строки для всех 10 листов; при запуске одного
    # листа — только выбранный лист.
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
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "vis-openai/gpt-5-mini"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--expected-mm", type=int, default=None)
    parser.add_argument(
        "--review-model",
        default=os.getenv("OPENAI_REVIEW_MODEL"),
        help="Optional second model that verifies and corrects the first answer",
    )
    parser.add_argument(
        "--nested-model",
        default=os.getenv("OPENAI_NESTED_MODEL", "vis-google/gemini-2.5-flash"),
        help="Separate vision model used only for nested dimensions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        args.pdf,
        args.output,
        args.model,
        args.dpi,
        args.max_pages,
        args.review_model,
        args.start_page,
        args.expected_mm,
        args.nested_model,
    )


if __name__ == "__main__":
    main()
