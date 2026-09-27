from __future__ import annotations

import pymupdf

from .models import NormalizedBox, PageInterpretation

COLORS = {
    "main": (0.0, 0.48, 0.53), "branch": (0.9, 0.28, 0.02),
    "nested": (0.55, 0.18, 0.72), "irrelevant": (0.45, 0.45, 0.45),
    "ambiguous": (0.9, 0.0, 0.0),
}


def normalized_rect(page: pymupdf.Page, box: NormalizedBox) -> pymupdf.Rect:
    return pymupdf.Rect(
        box.x0 / 1000 * page.rect.width, box.y0 / 1000 * page.rect.height,
        box.x1 / 1000 * page.rect.width, box.y1 / 1000 * page.rect.height,
    )


def annotate_page(page: pymupdf.Page, result: PageInterpretation) -> None:
    """Draw final role colours and labels into the output PDF page."""
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
        label_rect = pymupdf.Rect(
            min(rect.x1 + 2, page.rect.width - label_width),
            max(0.0, min(rect.y0, page.rect.height - label_height)),
            min(rect.x1 + 2, page.rect.width - label_width) + label_width,
            max(0.0, min(rect.y0, page.rect.height - label_height)) + label_height,
        )
        page.draw_rect(label_rect, color=color, fill=(1.0, 1.0, 1.0), width=1.2, overlay=True)
        page.insert_textbox(label_rect, dimension.label, fontsize=7, fontname="helv", color=color,
                            align=pymupdf.TEXT_ALIGN_CENTER, overlay=True)
