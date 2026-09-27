from __future__ import annotations

import queue
from types import SimpleNamespace

import analyze
import gui


class Value:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class Widget:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def configure(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))

    def start(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


def make_dimension() -> analyze.Dimension:
    return analyze.Dimension(
        segment_id=None,
        candidate_id="C1",
        label=None,
        value_mm=100,
        role="main",
        included=True,
        bbox=analyze.NormalizedBox(x0=1, y0=2, x1=3, y1=4),
        reason="model",
    )


def test_interpret_page_uses_structured_fake_client_without_network() -> None:
    result = analyze.PageInterpretation(
        line_id="LINE",
        dimensions=[make_dimension()],
        status="ok",
        ambiguities=[],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=result))],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    parse_calls: list[dict[str, object]] = []

    class Completions:
        def parse(self, **kwargs: object) -> object:
            parse_calls.append(kwargs)
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    candidates = [{"candidate_id": "C1", "value_mm": 100, "bbox": {"x0": 1, "y0": 2, "x1": 3, "y1": 4}}]

    parsed, metrics = analyze.interpret_page(client, "fake-model", 1, b"png", candidates)

    assert parsed.dimensions[0].label == "L1"
    assert metrics["total_tokens"] == 15
    assert parse_calls[0]["model"] == "fake-model"
    assert parse_calls[0]["temperature"] == 0


def test_gui_zoom_updates_percentage_without_window() -> None:
    app = gui.App.__new__(gui.App)
    app.result_zoom = 1.0
    app.zoom_var = Value("100%")
    app._render_result_image = lambda: None

    app._zoom_image(1.25)
    assert app.result_zoom == 1.25
    assert app.zoom_var.get() == "125%"

    app._reset_zoom()
    assert app.result_zoom == 1.0
    assert app.zoom_var.get() == "100%"


def test_gui_refreshes_nested_stage_even_when_cache_exists(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(gui, "ROOT", tmp_path)
    pdf = tmp_path / "drawing.pdf"
    pdf.write_bytes(b"%PDF")
    cache = tmp_path / "output/page4_nested_cache"
    cache.mkdir(parents=True)
    (cache / "results.json").write_text("[]", encoding="utf-8")
    started: list[list[list[str]]] = []

    class FakeThread:
        def __init__(self, *, target, args, daemon) -> None:
            started.append(args[0])

        def start(self) -> None:
            return None

    monkeypatch.setattr(gui.threading, "Thread", FakeThread)
    app = gui.App.__new__(gui.App)
    app.proc = None
    app.pdf_var = Value(str(pdf))
    app.model_var = Value("fake-model")
    app.start_var = Value("4")
    app.output_var = Value("")
    app.run_button = Widget()
    app.progress = Widget()

    app._run()

    assert app.output_var.get() == "output/page4"
    assert len(started) == 1
    assert len(started[0]) == 2
    assert "--nested-only" in started[0][0]
    assert "--nested-results" in started[0][1]


def test_gui_worker_queues_completion_without_subprocess(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = ["line 1\n"]

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(gui.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    app = gui.App.__new__(gui.App)
    app.events = queue.Queue()
    app.process_output = []
    app.proc = None

    app._worker([["fake-command"]])

    assert app.process_output == ["line 1\n"]
    assert app.events.get_nowait() == "__DONE__"
