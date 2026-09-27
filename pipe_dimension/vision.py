from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from .domain import assign_display_labels, validate_dimensions
from .geometry import as_data_url
from .models import PageInterpretation
from .prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from openai import OpenAI


def interpret_page(
    client: OpenAI,
    model: str,
    page_number: int,
    marked_png: bytes,
    candidates: list[dict[str, object]],
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[PageInterpretation, dict[str, int | float | str | None]]:
    """Ask one vision model to classify the already-extracted dimensions."""
    started = time.perf_counter()
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
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
        raise RuntimeError("The model did not return a parsed PageInterpretation")
    result = assign_display_labels(validate_dimensions(result, candidates))
    usage = response.usage
    return result, {
        "page_number": page_number,
        "model": model,
        "duration_seconds": round(elapsed, 3),
        "input_tokens": getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
