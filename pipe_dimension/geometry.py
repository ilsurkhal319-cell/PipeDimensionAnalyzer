from __future__ import annotations

import base64
import re
from io import BytesIO

import pymupdf
from PIL import Image, ImageDraw, ImageFont


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
