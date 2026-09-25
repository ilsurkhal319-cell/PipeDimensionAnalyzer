from __future__ import annotations

import queue
import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pymupdf


ROOT = Path(__file__).resolve().parent


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Расчёт длины трубопровода")
        self.geometry("1200x900")
        self.proc: subprocess.Popen[str] | None = None
        self.events: queue.Queue[str] = queue.Queue()
        self.process_output: list[str] = []
        self.result_image_source: Image.Image | None = None
        self.result_zoom = 1.0

        self.pdf_var = tk.StringVar(value=str(ROOT / "02_Изометрии_10_листов.pdf"))
        self.model_var = tk.StringVar(value="vis-google/gemini-3-flash-pre")
        self.start_var = tk.StringVar(value="1")
        self.output_var = tk.StringVar(value="output/page1")
        self.start_var.trace_add("write", self._update_output_name)
        self._build()
        self.after(100, self._poll)

    def _build(self) -> None:
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        frame = ttk.Frame(canvas, padding=12)
        frame_window = canvas.create_window((0, 0), window=frame, anchor="nw")
        frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(frame_window, width=event.width))
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="PDF-файл:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.pdf_var).grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Button(frame, text="Выбрать", command=self._choose_pdf).grid(row=0, column=2, padx=(8, 0))

        ttk.Label(frame, text="Модель:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.model_var).grid(row=1, column=1, columnspan=2, sticky="ew", pady=5)
        ttk.Label(frame, text="Начальный лист:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, width=8, textvariable=self.start_var).grid(row=2, column=1, sticky="w", pady=5)
        ttk.Label(frame, text="Результат:").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.output_var, state="readonly").grid(row=3, column=1, columnspan=2, sticky="ew", pady=5)

        self.run_button = ttk.Button(frame, text="Запустить анализ", command=self._run)
        self.run_button.grid(row=4, column=0, columnspan=3, pady=14)
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        self.result_frame = ttk.LabelFrame(frame, text="Результат", height=680)
        self.result_frame.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self.result_frame.grid_propagate(False)
        image_toolbar = ttk.Frame(self.result_frame)
        image_toolbar.grid(row=0, column=0, sticky="w", padx=6, pady=(4, 0))
        ttk.Button(image_toolbar, text="−", width=3, command=lambda: self._zoom_image(0.8)).pack(side="left")
        ttk.Button(image_toolbar, text="100%", command=self._reset_zoom).pack(side="left", padx=4)
        ttk.Button(image_toolbar, text="+", width=3, command=lambda: self._zoom_image(1.25)).pack(side="left")
        self.result_image = ttk.Label(self.result_frame, text="После анализа здесь появится разметка")
        self.result_image.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.result_text = tk.Text(self.result_frame, width=58, height=30, wrap="word", state="disabled",
                                   font=("Arial", 11), padx=12, pady=12)
        self.result_text.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=6, pady=6)
        self.result_frame.columnconfigure(0, weight=0)
        self.result_frame.columnconfigure(1, weight=1)
        self.result_frame.rowconfigure(1, weight=1)

    def _render_result_image(self) -> None:
        if self.result_image_source is None:
            return
        image = self.result_image_source.copy()
        width = max(200, int(image.width * self.result_zoom))
        height = max(150, int(image.height * self.result_zoom))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        image.thumbnail((1100, 900))
        self.result_photo = ImageTk.PhotoImage(image)
        self.result_image.configure(image=self.result_photo, text="")

    def _zoom_image(self, factor: float) -> None:
        self.result_zoom = max(0.25, min(4.0, self.result_zoom * factor))
        self._render_result_image()

    def _reset_zoom(self) -> None:
        self.result_zoom = 1.0
        self._render_result_image()

    def _choose_pdf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf"), ("Все файлы", "*")])
        if path:
            self.pdf_var.set(path)

    def _update_output_name(self, *_args: object) -> None:
        try:
            start = int(self.start_var.get())
            if start < 1:
                return
            self.output_var.set(f"output/page{start}")
        except ValueError:
            pass

    def _write(self, text: str) -> None:
        # Технический вывод намеренно не показывается в интерфейсе.
        pass

    def _show_result(self) -> None:
        try:
            start = int(self.start_var.get())
            out = ROOT / self.output_var.get()
            data = json.loads((out / "results.json").read_text(encoding="utf-8"))
            item = next(x for x in data if x.get("page_number") == start)
            annotated = out / "annotated.pdf"
            dimensions = item.get("interpretation", {}).get("dimensions", [])
            main = [d for d in dimensions if d.get("included") and d.get("role") == "main"]
            branch = [d for d in dimensions if d.get("included") and d.get("role") == "branch"]
            main_sum = sum(d.get("value_mm", 0) for d in main)
            branch_sum = sum(d.get("value_mm", 0) for d in branch)
            total = item.get("total_length_mm") or 0
            self.result_text.configure(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.tag_configure("title", font=("Arial", 13, "bold"))
            self.result_text.tag_configure("total", font=("Arial", 20, "bold"), foreground="#006f73")
            self.result_text.tag_configure("heading", font=("Arial", 11, "bold"))
            self.result_text.tag_configure("main_legend", foreground="#006f73")
            self.result_text.tag_configure("branch_legend", foreground="#e64a19")
            self.result_text.tag_configure("nested_legend", foreground="#7b2cbf")
            self.result_text.insert("end", "Суммарная размерная длина\n", "title")
            self.result_text.insert("end", f"{total} мм = {total / 1000:.3f} м\n", "total")
            self.result_text.insert("end", f"Основная трасса {main_sum} мм + ветви {branch_sum} мм\n\n")
            self.result_text.insert("end", "Метка       Участок                 Размер мм\n", "heading")
            self.result_text.insert("end", "─" * 48 + "\n")
            for d in dimensions:
                if d.get("role") == "nested":
                    continue
                mark = d.get("label") or d.get("candidate_id")
                role = {"main": "Основная трасса", "branch": "Ответвление", "nested": "Вложенный"}.get(d.get("role"), d.get("role"))
                state = "" if d.get("included") else " [исключён]"
                self.result_text.insert("end", f"{mark:<11} {role:<24} {d.get('value_mm'):>7}{state}\n")
            self.result_text.insert("end", "\nОсновная трасса\n", "heading")
            self.result_text.insert("end", " + ".join(str(d.get("value_mm")) for d in main) + f" = {main_sum} мм\n")
            self.result_text.insert("end", "\nС учётом ответвлений\n", "heading")
            self.result_text.insert("end", f"{main_sum} + {branch_sum} = {total} мм = {total / 1000:.3f} м\n")
            self.result_text.insert("end", "\nПочему вложенные размеры не прибавляются\n", "heading")
            self.result_text.insert(
                "end",
                "Вложенные размеры находятся внутри уже выбранных общих участков "
                "и повторно не увеличивают длину трассы.\n\n"
            )
            main_labels = [d.get("label") or d.get("candidate_id") for d in dimensions if d.get("role") == "main" and d.get("included")]
            branch_labels = [d.get("label") or d.get("candidate_id") for d in dimensions if d.get("role") == "branch" and d.get("included")]
            nested_labels = [d.get("label") or d.get("candidate_id") for d in dimensions if d.get("role") == "nested"]
            if main_labels:
                self.result_text.insert("end", f"{', '.join(main_labels)} — размеры основной\nтрассы.\n", "main_legend")
            if branch_labels:
                self.result_text.insert("end", f"{', '.join(branch_labels)} — длина ответвления.\n", "branch_legend")
            if nested_labels:
                self.result_text.insert("end", f"{', '.join(nested_labels)} — исключённые\nвложенные размеры.\n", "nested_legend")
            self.result_text.configure(state="disabled")
            if annotated.exists():
                pdf = pymupdf.open(annotated)
                page = pdf[0]
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2), alpha=False)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                self.result_image_source = image
                self.result_zoom = 1.0
                self._render_result_image()
        except Exception as exc:
            self._write(f"Не удалось показать сводку: {exc}\n")

    def _run(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        pdf = Path(self.pdf_var.get())
        if not pdf.exists():
            messagebox.showerror("Ошибка", "PDF-файл не найден")
            return
        try:
            start = int(self.start_var.get())
            pages = 1
            if start < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Номер листа и количество должны быть положительными числами")
            return
        self.output_var.set(f"output/page{start}")
        cmd = [sys.executable, str(ROOT / "analyze.py"), str(pdf),
               "--output", self.output_var.get(), "--model", self.model_var.get(),
               "--dpi", "150", "--start-page", str(start), "--max-pages", str(pages)]
        self._write("$ " + " ".join(cmd) + "\n\n")
        self.run_button.configure(state="disabled")
        self.progress.start(10)
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd: list[str]) -> None:
        try:
            self.process_output = []
            self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.process_output.append(line)
            code = self.proc.wait()
            self.events.put("__ERROR__" if code else "__DONE__")
        except Exception as exc:
            self.process_output.append(f"Ошибка запуска: {exc}\n")
            self.events.put("__ERROR__")

    def _poll(self) -> None:
        try:
            while True:
                item = self.events.get_nowait()
                if item == "__DONE__":
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    self._show_result()
                elif item == "__ERROR__":
                    self.progress.stop()
                    self.run_button.configure(state="normal")
                    messagebox.showerror("Ошибка анализа", "".join(self.process_output)[-4000:])
                else:
                    self._write(item)
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    App().mainloop()
