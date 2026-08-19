#!/usr/bin/env python3
"""실환경 reference 이미지에 image-exemplar BBox를 등록하는 대화형 라벨링 툴.

상위 폴더(dataset-root)와 하위 scene 경로만 주면 reference/<object_name>/ 아래의
모든 object를 알파벳순으로 훑으며 BBox를 그린다. 결과는 object 폴더의 JSON sidecar에 저장되며,
이후 cross_image_exemplar_real.py가 이 정보를 읽어 reference와 target 이미지를 결합한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageOps, ImageTk

if __package__:
    from .real_object_catalog import (
        RealObjectCatalogError,
        RealObjectIdentity,
        resolve_real_object,
    )
else:  # pragma: no cover - exercised by direct CLI execution
    from real_object_catalog import (
        RealObjectCatalogError,
        RealObjectIdentity,
        resolve_real_object,
    )


DONE_STYLE = {"bg": "#2e7d32", "fg": "white"}
TODO_STYLE = {"bg": "#e0e0e0", "fg": "black"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively select and save SAM 3 real-image exemplar BBoxes for "
            "every object captured by reference-gen under one scene."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Top-level dataset folder that holds objects_metadata.csv, e.g. "
        "/media/uon/data1/gemini",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene path relative to --dataset-root, e.g. "
        "real_v1/home/LivingRoom_Kitchen/dining_table",
    )
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=900,
        help="Max width/height of the image canvas in pixels. Coordinates are "
        "always saved in the original image's pixel space.",
    )
    return parser.parse_args()


@dataclass
class ObjectEntry:
    object_name: str
    directory: Path
    image_path: Path
    bbox_json_path: Path
    preview_path: Path

    @property
    def is_labeled(self) -> bool:
        return self.bbox_json_path.is_file()


def discover_objects(reference_root: Path) -> list[ObjectEntry]:
    """reference_root 아래 object 폴더를 알파벳순으로 스캔한다."""
    entries: list[ObjectEntry] = []
    directories = sorted(
        (path for path in reference_root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    for directory in directories:
        images = sorted(directory.glob("*.png"))
        if not images:
            print(
                f"label-real-reference: skipping {directory} (no reference PNG)",
                file=sys.stderr,
            )
            continue
        if len(images) > 1:
            print(
                f"label-real-reference: {directory} has {len(images)} PNGs; "
                f"using {images[0].name}",
                file=sys.stderr,
            )
        image_path = images[0]
        entries.append(
            ObjectEntry(
                object_name=directory.name,
                directory=directory,
                image_path=image_path,
                bbox_json_path=directory / f"{image_path.stem}.json",
                preview_path=directory / f"{image_path.stem}_preview.jpg",
            )
        )
    return entries


def save_bbox_preview(
    image: Image.Image,
    bbox: list[float],
    object_name: str,
    output_path: Path,
) -> None:
    """BBox JSON과 같은 stem의 검수 이미지를 저장한다."""
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    line_width = max(4, round(max(image.size) / 700))
    color = (0, 255, 255)
    draw.rectangle(bbox, outline=color, width=line_width)
    draw.text(
        (bbox[0], max(0, bbox[1] - line_width * 5)),
        f"(object={object_name})",
        fill=color,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    preview.save(output_path, quality=95)


def update_reference_index(
    *,
    index_path: Path,
    object_name: str,
    reference_image: Path,
    bbox_annotation: Path,
    identity_metadata: dict[str, object] | None = None,
) -> None:
    """객체 이름으로 reference와 BBox를 즉시 찾는 인덱스를 갱신한다."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"objects": {}}
    objects = index.setdefault("objects", {})
    entry: dict[str, object] = {
        "reference_image": Path(
            os.path.relpath(reference_image, start=index_path.parent.resolve())
        ).as_posix(),
        "bbox_annotation": Path(
            os.path.relpath(
                bbox_annotation.resolve(), start=index_path.parent.resolve()
            )
        ).as_posix(),
    }
    if identity_metadata:
        entry.update(identity_metadata)
    objects[object_name] = entry

    # 쓰는 도중 중단되어 기존 인덱스가 손상되지 않도록 임시 파일을 교체한다.
    temporary_path = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    temporary_path.replace(index_path)


class LabelingApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        entries: list[ObjectEntry],
        metadata_path: Path,
        reference_index_path: Path,
        max_image_size: int,
    ) -> None:
        self.root = root
        self.entries = entries
        self.metadata_path = metadata_path
        self.reference_index_path = reference_index_path
        self.max_image_size = max_image_size

        self._identity_cache: dict[str, RealObjectIdentity] = {}
        self._drag_start: tuple[int, int] | None = None
        self._drag_rect_id: int | None = None
        self._display_image: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._display_size = (1, 1)
        self._syncing_listbox = False
        self._syncing_scale = False

        self.index = self._initial_index()

        self._build_ui()
        self._populate_sidebar()
        self._refresh_all()

    def _initial_index(self) -> int:
        for position, entry in enumerate(self.entries):
            if not entry.is_labeled:
                return position
        return 0

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        self.status_var = tk.StringVar()
        self.pos_var = tk.StringVar()

        status_label = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status_label.pack(side="top", fill="x", padx=8, pady=(8, 0))

        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(left, bg="#202020", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        nav = ttk.Frame(left)
        nav.pack(side="top", fill="x", pady=(6, 0))
        self.prev_button = ttk.Button(
            nav, text="◀", width=3, command=lambda: self._set_index(self.index - 1)
        )
        self.prev_button.pack(side="left")
        self.next_button = ttk.Button(
            nav, text="▶", width=3, command=lambda: self._set_index(self.index + 1)
        )
        self.next_button.pack(side="left", padx=(2, 0))
        self.scale_var = tk.DoubleVar()
        self.scale = ttk.Scale(
            nav,
            from_=0,
            to=max(0, len(self.entries) - 1),
            variable=self.scale_var,
            orient="horizontal",
            command=self._on_scale_move,
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(nav, textvariable=self.pos_var, width=12, anchor="e").pack(
            side="left", padx=(6, 0)
        )

        right = ttk.Frame(body, width=300)
        right.pack(side="left", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        legend = ttk.Frame(right)
        legend.pack(side="top", fill="x", pady=(0, 4))
        done_swatch = tk.Label(legend, text="DONE", **DONE_STYLE, width=6)
        done_swatch.pack(side="left")
        todo_swatch = tk.Label(legend, text="TODO", **TODO_STYLE, width=6)
        todo_swatch.pack(side="left", padx=(6, 0))

        list_frame = ttk.Frame(right)
        list_frame.pack(side="top", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self.listbox = tk.Listbox(
            list_frame,
            exportselection=False,
            activestyle="dotbox",
            font=("Courier", 10),
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=self.listbox.yview)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")
        self.listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

    def _populate_sidebar(self) -> None:
        for position, entry in enumerate(self.entries):
            self.listbox.insert(tk.END, self._format_item(entry, position))
            self._apply_item_style(position)

    @staticmethod
    def _format_item(entry: ObjectEntry, position: int) -> str:
        status = "DONE" if entry.is_labeled else "TODO"
        return f"{position:04d}  {status:<4}  {entry.object_name}"

    def _apply_item_style(self, position: int) -> None:
        style = DONE_STYLE if self.entries[position].is_labeled else TODO_STYLE
        self.listbox.itemconfig(position, **style)

    def _refresh_sidebar_row(self, position: int) -> None:
        self.listbox.delete(position)
        self.listbox.insert(position, self._format_item(self.entries[position], position))
        self._apply_item_style(position)

    # -- Navigation --------------------------------------------------------

    def _set_index(self, new_index: int) -> None:
        new_index = max(0, min(len(self.entries) - 1, new_index))
        self.index = new_index
        self._refresh_all()

    def _on_scale_move(self, value: str) -> None:
        if self._syncing_scale:
            return
        index = round(float(value))
        if index != self.index:
            self._set_index(index)

    def _on_listbox_select(self, _event: object) -> None:
        if self._syncing_listbox:
            return
        selection = self.listbox.curselection()
        if selection:
            self._set_index(selection[0])

    def _refresh_all(self) -> None:
        self._render_image()

        self.pos_var.set(f"{self.index + 1}/{len(self.entries)}")
        self._syncing_scale = True
        self.scale_var.set(self.index)
        self._syncing_scale = False
        self.prev_button.state(["disabled" if self.index <= 0 else "!disabled"])
        self.next_button.state(
            ["disabled" if self.index >= len(self.entries) - 1 else "!disabled"]
        )

        self._syncing_listbox = True
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.index)
        self.listbox.see(self.index)
        self._syncing_listbox = False

    def _render_image(self) -> None:
        entry = self.entries[self.index]
        self._drag_start = None
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        labeled = entry.is_labeled
        source_path = entry.preview_path if labeled else entry.image_path
        try:
            if labeled:
                # Preview was already rendered from an EXIF-normalized copy;
                # re-applying exif_transpose here would risk a double rotation.
                with Image.open(source_path) as opened:
                    image = opened.convert("RGB")
            else:
                with Image.open(source_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
        except OSError as exc:
            messagebox.showerror("Load error", str(exc), parent=self.root)
            self.canvas.delete("all")
            self._display_image = None
            return

        width, height = image.size
        scale = min(1.0, self.max_image_size / max(width, height))
        display_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        if scale != 1.0:
            image = image.resize(display_size, Image.Resampling.LANCZOS)
        self._scale = scale
        self._display_size = display_size

        self._display_image = ImageTk.PhotoImage(image)
        self.canvas.config(width=display_size[0], height=display_size[1])
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._display_image)

        state = "LABELED" if labeled else "UNLABELED"
        action = "drag to relabel (overwrite)" if labeled else "drag to label"
        self.status_var.set(
            f"[{self.index + 1}/{len(self.entries)}] {entry.object_name} "
            f"({state}) — {action}"
        )

    # -- BBox drag ---------------------------------------------------------

    def _on_press(self, event: tk.Event) -> None:
        if self._display_image is None:
            return
        self._drag_start = (event.x, event.y)
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def _on_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        if self._drag_rect_id is None:
            self._drag_rect_id = self.canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#00e5ff", width=2
            )
        else:
            self.canvas.coords(self._drag_rect_id, x0, y0, event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        if self._drag_rect_id is not None:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        display_width, display_height = self._display_size
        dx0, dx1 = sorted((x0, event.x))
        dy0, dy1 = sorted((y0, event.y))
        dx0 = max(0.0, float(dx0))
        dy0 = max(0.0, float(dy0))
        dx1 = min(float(display_width), float(dx1))
        dy1 = min(float(display_height), float(dy1))
        if dx1 - dx0 < 2 or dy1 - dy0 < 2:
            return  # A click, not a drag; ignore rather than error out.

        scale = self._scale
        bbox = [
            round(dx0 / scale, 2),
            round(dy0 / scale, 2),
            round(dx1 / scale, 2),
            round(dy1 / scale, 2),
        ]

        entry = self.entries[self.index]
        identity = self._resolve_identity(entry.object_name)
        if identity is None:
            return  # Error already shown to the operator.

        if entry.is_labeled and not messagebox.askyesno(
            "Overwrite label?",
            f"{entry.object_name} is already labeled.\n"
            "Overwrite it with the new box?",
            parent=self.root,
        ):
            return

        self._save_label(entry, bbox, identity)
        self._advance_to_next_unlabeled()

    def _resolve_identity(self, object_name: str) -> RealObjectIdentity | None:
        cached = self._identity_cache.get(object_name)
        if cached is not None:
            return cached
        try:
            identity = resolve_real_object(object_name, self.metadata_path)
        except RealObjectCatalogError as exc:
            messagebox.showerror("Catalog mismatch", str(exc), parent=self.root)
            return None
        self._identity_cache[object_name] = identity
        return identity

    def _save_label(
        self, entry: ObjectEntry, bbox: list[float], identity: RealObjectIdentity
    ) -> None:
        with Image.open(entry.image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        record: dict[str, object] = {
            "object_name": entry.object_name,
            "reference_image": entry.image_path.name,
            "bbox_format": "xyxy",
            "bbox": bbox,
            "image_width": width,
            "image_height": height,
            "exif_orientation_applied": True,
        }
        record.update(identity.metadata())
        entry.bbox_json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        save_bbox_preview(image, bbox, entry.object_name, entry.preview_path)
        update_reference_index(
            index_path=self.reference_index_path,
            object_name=entry.object_name,
            reference_image=entry.image_path,
            bbox_annotation=entry.bbox_json_path,
            identity_metadata=identity.metadata(),
        )
        self._refresh_sidebar_row(self.index)
        self.status_var.set(f"Saved: {entry.object_name} -> {entry.bbox_json_path}")

    def _advance_to_next_unlabeled(self) -> None:
        total = len(self.entries)
        for offset in range(1, total + 1):
            candidate = (self.index + offset) % total
            if not self.entries[candidate].is_labeled:
                self._set_index(candidate)
                return
        self._refresh_all()
        self.status_var.set("All objects are labeled.")


def main() -> None:
    args = parse_args()
    try:
        dataset_root = args.dataset_root.expanduser().resolve()
        metadata_path = dataset_root / "objects_metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"objects_metadata.csv not found: {metadata_path}")

        reference_root = (dataset_root / args.scene / "reference").resolve()
        if not reference_root.is_dir():
            raise FileNotFoundError(f"Reference directory not found: {reference_root}")

        entries = discover_objects(reference_root)
        if not entries:
            raise RuntimeError(
                f"No object folders with a reference PNG found under {reference_root}"
            )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"label-real-reference: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    reference_index_path = reference_root / "reference_index.json"

    root = tk.Tk()
    root.title(f"SAM3 Reference Labeling — {args.scene}")
    root.geometry("1280x860")
    LabelingApp(
        root,
        entries=entries,
        metadata_path=metadata_path,
        reference_index_path=reference_index_path,
        max_image_size=args.max_display_size,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
