#!/usr/bin/env python3
"""실환경 reference 이미지에 image-exemplar BBox를 등록한다.
사용자가 지정한 객체 이름과 선택한 BBox를 별도의 JSON sidecar에 저장하며, 
이후 cross_image_exemplar_real.py가 이 정보를 읽어 reference와 target 이미지를 결합한다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

if __package__:
    from .real_object_catalog import resolve_optional_identity
else:  # pragma: no cover - exercised by direct CLI execution
    from real_object_catalog import resolve_optional_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and save a SAM 3 real-image exemplar BBox."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument(
        "--objects-metadata",
        type=Path,
        help=(
            "Optional objects_metadata.csv. When provided, Object_name, "
            "Old_name, or Class_name input is normalized to the real pipeline "
            "key (Old_name first, Object_name fallback)."
        ),
    )
    parser.add_argument("--bbox-dir", type=Path, default=Path("assets/reference"))
    parser.add_argument(
        "--reference-index",
        type=Path,
        default=Path("assets/reference/reference_index.json"),
    )
    parser.add_argument(
        "--max-display-size",
        type=int,
        default=1400,
        help="선택 창의 최대 가로/세로 크기. 좌표는 원본 크기로 자동 복원된다.",
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    identity = resolve_optional_identity(args.object_name, args.objects_metadata)
    object_name = identity.applied_key if identity is not None else args.object_name
    identity_metadata = identity.metadata() if identity is not None else None
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {image_path}")

    # 스마트폰 JPEG는 픽셀 배열과 실제 표시 방향이 다를 수 있다. BBox 선택과 추론이
    # 같은 좌표계를 쓰도록 EXIF orientation을 먼저 픽셀에 반영한다.
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size

    scale = min(1.0, args.max_display_size / max(width, height))
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    display = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if scale != 1.0:
        display = cv2.resize(
            display,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA,
        )

    window_name = f"Select BBox: {object_name} (ENTER/SPACE=save, C=cancel)"
    x, y, box_width, box_height = cv2.selectROI(
        # The crosshair is only an OpenCV selection guide, but it looks like
        # extra BBoxes in the UI.  Show only the actual ROI outline.
        window_name, display, showCrosshair=False, fromCenter=False
    )
    cv2.destroyAllWindows()
    if box_width <= 0 or box_height <= 0:
        raise RuntimeError("BBox selection was cancelled or empty; nothing was saved.")

    # 화면 축소 비율을 역으로 적용해 EXIF 정규화된 원본 픽셀 좌표를 저장한다.
    x1 = max(0.0, x / scale)
    y1 = max(0.0, y / scale)
    x2 = min(float(width), (x + box_width) / scale)
    y2 = min(float(height), (y + box_height) / scale)

    args.bbox_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.bbox_dir / f"{image_path.stem}.json"
    relative_image = os.path.relpath(image_path, start=args.bbox_dir.resolve())
    bbox = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
    record: dict[str, object] = {
        "object_name": object_name,
        "reference_image": Path(relative_image).as_posix(),
        "bbox_format": "xyxy",
        "bbox": bbox,
        "image_width": width,
        "image_height": height,
        "exif_orientation_applied": True,
    }
    if identity_metadata:
        record.update(identity_metadata)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    # 원본 reference 이미지와 혼동되지 않도록 검수용 파생 이미지임을 파일명에 표시한다.
    preview_path = args.bbox_dir / f"{image_path.stem}_preview.jpg"
    save_bbox_preview(image, bbox, object_name, preview_path)
    update_reference_index(
        index_path=args.reference_index,
        object_name=object_name,
        reference_image=image_path,
        bbox_annotation=output_path,
        identity_metadata=identity_metadata,
    )
    print(f"Saved reference annotation: {output_path}")
    print(f"Saved reference preview: {preview_path}")
    print(f"Updated reference index: {args.reference_index}")
    if identity is not None:
        print(
            f"Resolved object query {args.object_name!r} via "
            f"{identity.matched_field} to real key {object_name!r} "
            f"({identity.object_id})."
        )
    print(f"object_name={object_name}, bbox={record['bbox']}")


if __name__ == "__main__":
    main()
