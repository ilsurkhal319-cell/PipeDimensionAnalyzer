from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
