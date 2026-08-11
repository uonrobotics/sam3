#!/usr/bin/env python3
"""하나의 실환경 reference로 여러 실환경 이미지를 일괄 추론한다.

SAM 3에는 별도 이미지의 exemplar를 target에 직접 전달하는 공개 API가 없다.
따라서 reference와 target을 종횡비가 보존된 1008x1008 canvas에 위아래로 배치하고,
reference BBox를 동일 canvas의 positive geometric prompt로 제공한다.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.visualization_utils import normalize_bbox

if __package__:
    from .real_object_catalog import RealObjectIdentity, resolve_real_object
else:  # pragma: no cover - exercised by direct CLI execution
    from real_object_catalog import RealObjectIdentity, resolve_real_object


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BACKGROUND_RGBA = (0, 0, 0, 0)
UNLABELLED_RGBA = (0, 0, 0, 255)
TARGET_RGBA = (255, 25, 25, 255)


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    offset_x: int
    offset_y: int
    resized_width: int
    resized_height: int
    original_width: int
    original_height: int


@dataclass(frozen=True)
class DatasetModePaths:
    root: Path
    camera_name: str
    image_dir: Path
    reference_index: Path
    capture_manifest: Path
    bbox_dir: Path
    inst_seg_dir: Path
    diagnostics_object_dir: Path
    summary_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch SAM 3 inference using a stitched real-image exemplar."
    )
    parser.add_argument("--object-name", required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Enable the canonical Gemini dataset mode. Inputs are read from "
            "<root>/capture_manifest.json, <root>/reference/reference_index.json, "
            "and <root>/rgb/<camera-name>; canonical outputs are written beneath "
            "the same dataset root. Explicit legacy path arguments are ignored."
        ),
    )
    parser.add_argument(
        "--camera-name",
        default="top_view_camera",
        help="Camera key used by dataset-mode images_by_camera and output paths.",
    )
    parser.add_argument(
        "--objects-metadata",
        type=Path,
        help=(
            "Optional objects_metadata.csv. When provided, Object_name, "
            "Old_name, or Class_name input is normalized to the real pipeline "
            "key (Old_name first, Object_name fallback)."
        ),
    )
    parser.add_argument("--image-dir", type=Path, default=Path("assets/images2"))
    parser.add_argument(
        "--reference-index",
        type=Path,
        default=Path("assets/reference/reference_index.json"),
    )
    parser.add_argument(
        "--scene-meta",
        type=Path,
        default=Path("assets/scene_meta2/capture_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cross_image_exemplar_real"),
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1008,
        help="공식 image notebook과 같은 기본 입력 크기.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="공식 image notebook과 같은 기본 confidence threshold.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--save-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write dataset-mode overlay and stitched prompt diagnostics "
            "(default: enabled)."
        ),
    )
    return parser.parse_args(argv)


def dataset_mode_paths(
    dataset_root: Path,
    camera_name: str,
    object_name: str,
) -> DatasetModePaths:
    """Resolve the fixed Gemini dataset-mode input and output contract."""

    root = dataset_root.expanduser().resolve()
    camera_component = _safe_path_component(camera_name, "camera name")
    object_component = _safe_path_component(object_name, "object name")
    return DatasetModePaths(
        root=root,
        camera_name=camera_component,
        image_dir=root / "rgb" / camera_component,
        reference_index=root / "reference" / "reference_index.json",
        capture_manifest=root / "capture_manifest.json",
        bbox_dir=root / "bbox" / camera_component,
        inst_seg_dir=root / "inst_seg" / camera_component,
        diagnostics_object_dir=(
            root / "diagnostics" / "sam3" / camera_component / object_component
        ),
        summary_path=(
            root
            / "inference_meta"
            / "sam3"
            / camera_component
            / object_component
            / "summary.json"
        ),
    )


def _safe_path_component(value: str, description: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
        or Path(value).is_absolute()
    ):
        raise ValueError(f"Unsafe {description}: {value!r}")
    return value


def load_oriented_rgb(path: Path) -> Image.Image:
    """EXIF orientation을 실제 픽셀에 반영한 RGB 이미지를 반환한다."""
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def load_reference_annotation(
    object_name: str,
    reference_index_path: Path,
    *,
    expected_identity: RealObjectIdentity | None = None,
) -> tuple[Path, dict]:
    """인덱스에서 객체 이름으로 reference와 BBox JSON을 직접 찾는다."""
    if not reference_index_path.is_file():
        raise FileNotFoundError(f"Reference index not found: {reference_index_path}")
    index = json.loads(reference_index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not isinstance(index.get("objects"), dict):
        raise ValueError(
            f"Reference index must contain an objects mapping: "
            f"{reference_index_path}"
        )
    entry = index["objects"].get(object_name)
    if entry is None:
        raise KeyError(
            f"Object {object_name!r} not found in reference index "
            f"{reference_index_path}"
        )
    if not isinstance(entry, dict):
        raise ValueError(
            f"Reference entry for {object_name!r} must be an object in "
            f"{reference_index_path}."
        )
    if expected_identity is not None:
        _validate_identity_metadata(
            entry,
            expected_identity,
            source=f"reference index {reference_index_path}",
        )

    reference_value = entry.get("reference_image")
    annotation_value = entry.get("bbox_annotation")
    if not all(
        isinstance(value, str) and value
        for value in (reference_value, annotation_value)
    ):
        raise ValueError(
            f"Incomplete reference entry for {object_name!r} in {reference_index_path}"
        )
    reference_path = (reference_index_path.parent / reference_value).resolve()
    annotation_path = (reference_index_path.parent / annotation_value).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"BBox annotation not found: {annotation_path}")

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(annotation, dict):
        raise ValueError(f"BBox annotation must be an object: {annotation_path}")
    if annotation.get("object_name") != object_name:
        raise ValueError(
            f"Object mismatch: index={object_name!r}, "
            f"annotation={annotation.get('object_name')!r}"
        )
    if expected_identity is not None:
        _validate_identity_metadata(
            annotation,
            expected_identity,
            source=f"BBox annotation {annotation_path}",
        )
    return reference_path, annotation


def load_target_image_paths(
    object_name: str,
    image_dir: Path,
    scene_meta_path: Path,
    *,
    expected_identity: RealObjectIdentity | None = None,
    camera_name: str | None = None,
) -> list[Path]:
    """수집 metadata에서 해당 객체로 등록된 target 이미지만 반환한다."""
    if not scene_meta_path.is_file():
        raise FileNotFoundError(f"Capture manifest not found: {scene_meta_path}")
    manifest = json.loads(scene_meta_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Capture manifest must be an object: {scene_meta_path}")
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        raise ValueError(
            f"Capture manifest must contain an objects mapping: {scene_meta_path}"
        )
    entry = objects.get(object_name)
    if entry is None:
        raise KeyError(
            f"Object {object_name!r} not found in capture manifest {scene_meta_path}"
        )
    if not isinstance(entry, dict):
        raise ValueError(
            f"Capture manifest entry for {object_name!r} must be an object."
        )
    if expected_identity is not None:
        catalog = manifest.get("object_catalog")
        if not isinstance(catalog, dict):
            raise ValueError(
                "Capture manifest has no object_catalog provenance while "
                "--objects-metadata is enabled."
            )
        manifest_hash = catalog.get("sha256")
        if manifest_hash != expected_identity.object_catalog_sha256:
            raise ValueError(
                "Capture manifest object catalog SHA-256 does not match the "
                "selected objects_metadata.csv: "
                f"{manifest_hash!r} != "
                f"{expected_identity.object_catalog_sha256!r}."
            )
        _validate_identity_metadata(
            entry,
            expected_identity,
            source=f"capture manifest {scene_meta_path}",
        )
    image_names: object
    used_legacy_images = True
    images_by_camera = entry.get("images_by_camera")
    if camera_name is not None and images_by_camera is not None:
        if not isinstance(images_by_camera, dict):
            raise ValueError(f"images_by_camera for {object_name!r} must be an object.")
        if camera_name in images_by_camera:
            image_names = images_by_camera[camera_name]
            used_legacy_images = False
        else:
            image_names = entry.get("images", [])
    else:
        image_names = entry.get("images", [])
    if not isinstance(image_names, list):
        raise ValueError(f"Target images for {object_name!r} must be a list.")
    if not all(isinstance(name, str) and name for name in image_names):
        raise ValueError(
            f"Target image names for {object_name!r} must be non-empty strings."
        )
    if camera_name is not None and used_legacy_images:
        camera_relative_names: list[str] = []
        for image_name in image_names:
            relative_path = Path(image_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Target image must be relative to --image-dir: {image_name!r}"
                )
            if len(relative_path.parts) == 1:
                camera_relative_names.append(image_name)
            elif relative_path.parts[0] == camera_name:
                camera_relative_names.append(Path(*relative_path.parts[1:]).as_posix())
        image_names = camera_relative_names
    if not image_names:
        camera_suffix = (
            f" and camera {camera_name!r}" if camera_name is not None else ""
        )
        raise ValueError(
            f"No target images registered for {object_name!r}{camera_suffix}"
        )
    if len(image_names) != len(set(image_names)):
        raise ValueError(f"Duplicate target image names for {object_name!r}")

    image_root = image_dir.resolve()
    image_paths: list[Path] = []
    for image_name in image_names:
        relative_path = Path(image_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Target image must be relative to --image-dir: {image_name!r}"
            )
        image_path = (image_root / relative_path).resolve()
        try:
            image_path.relative_to(image_root)
        except ValueError as exc:
            raise ValueError(
                f"Target image escapes --image-dir: {image_name!r}"
            ) from exc
        if not image_path.is_file():
            raise FileNotFoundError(f"Target image not found: {image_path}")
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported target image suffix: {image_path}")
        image_paths.append(image_path)
    return image_paths


def load_manifest_object_entry(
    object_name: str,
    scene_meta_path: Path,
) -> dict:
    """Return one manifest object entry for dataset output identity metadata."""

    if not scene_meta_path.is_file():
        raise FileNotFoundError(f"Capture manifest not found: {scene_meta_path}")
    manifest = json.loads(scene_meta_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("objects"), dict):
        raise ValueError(
            f"Capture manifest must contain an objects mapping: {scene_meta_path}"
        )
    entry = manifest["objects"].get(object_name)
    if not isinstance(entry, dict):
        raise KeyError(
            f"Object {object_name!r} not found in capture manifest {scene_meta_path}"
        )
    return entry


def _validate_identity_metadata(
    document: dict,
    identity: RealObjectIdentity,
    *,
    source: str,
) -> None:
    """Require an artifact to match the selected immutable catalog row."""

    for field, expected in identity.metadata().items():
        if field not in document:
            raise ValueError(f"{source} is missing catalog identity field {field!r}.")
        actual = document[field]
        if actual != expected:
            raise ValueError(f"{source} has {field}={actual!r}; expected {expected!r}.")


def letterbox(
    image: Image.Image, tile_width: int, tile_height: int
) -> tuple[Image.Image, LetterboxTransform]:
    """종횡비를 유지해 tile 안에 배치하고 좌표 역변환 정보를 반환한다."""
    original_width, original_height = image.size
    scale = min(tile_width / original_width, tile_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    offset_x = (tile_width - resized_width) // 2
    offset_y = (tile_height - resized_height) // 2

    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (tile_width, tile_height), (0, 0, 0))
    tile.paste(resized, (offset_x, offset_y))
    transform = LetterboxTransform(
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
        resized_width=resized_width,
        resized_height=resized_height,
        original_width=original_width,
        original_height=original_height,
    )
    return tile, transform


def map_box_to_canvas(
    box: list[float], transform: LetterboxTransform, tile_y: int
) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        x1 * transform.scale + transform.offset_x,
        y1 * transform.scale + transform.offset_y + tile_y,
        x2 * transform.scale + transform.offset_x,
        y2 * transform.scale + transform.offset_y + tile_y,
    ]


def map_boxes_to_original(
    boxes: torch.Tensor, transform: LetterboxTransform, tile_y: int
) -> torch.Tensor:
    restored = boxes.clone()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - transform.offset_x) / transform.scale
    restored[:, [1, 3]] = (
        restored[:, [1, 3]] - tile_y - transform.offset_y
    ) / transform.scale
    restored[:, [0, 2]] = restored[:, [0, 2]].clamp(0, transform.original_width)
    restored[:, [1, 3]] = restored[:, [1, 3]].clamp(0, transform.original_height)
    return restored


def restore_masks(
    masks: torch.Tensor,
    transform: LetterboxTransform,
    tile_y: int,
    tile_height: int,
    canvas_width: int,
) -> list[np.ndarray]:
    """Canvas mask에서 target padding을 제거하고 target 원본 크기로 복원한다."""
    restored: list[np.ndarray] = []
    x1 = transform.offset_x
    y1 = tile_y + transform.offset_y
    x2 = x1 + transform.resized_width
    y2 = y1 + transform.resized_height
    for mask in masks:
        mask_np = mask.squeeze(0).detach().cpu().numpy().astype(np.uint8) * 255
        # 방어적으로 processor 출력 크기를 canvas 크기에 맞춘다.
        if mask_np.shape != (tile_height * 2, canvas_width):
            mask_image = Image.fromarray(mask_np, mode="L").resize(
                (canvas_width, tile_height * 2), Image.Resampling.NEAREST
            )
        else:
            mask_image = Image.fromarray(mask_np, mode="L")
        content = mask_image.crop((x1, y1, x2, y2)).resize(
            (transform.original_width, transform.original_height),
            Image.Resampling.NEAREST,
        )
        restored.append(np.asarray(content) > 0)
    return restored


def save_results(
    target: Image.Image,
    masks: list[np.ndarray],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_image = target.convert("RGBA")
    # 실환경 pilot의 주 대상물이 주황/빨강 계열이므로 첫 prediction은 원본 색과
    # 확실히 대비되는 청록색으로 고정한다. 추가 prediction은 기존 순환 팔레트 사용.
    colors = [(0, 255, 255), (255, 0, 255), (50, 120, 255), (48, 220, 80)]
    records: list[dict] = []

    for index, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image.save(output_dir / f"mask_{index:03d}.png")
        color = colors[index % len(colors)]
        color_layer = Image.new("RGBA", target.size, (*color, 0))
        color_layer.putalpha(mask_image.point(lambda value: 100 if value else 0))
        overlay_image = Image.alpha_composite(overlay_image, color_layer)

        records.append(
            {
                "index": index,
                "score": float(score.detach().cpu()),
                "bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()],
            }
        )

    draw = ImageDraw.Draw(overlay_image)
    for record in records:
        index = record["index"]
        color = colors[index % len(colors)]
        box = record["bbox_xyxy"]
        draw.rectangle(box, outline=(*color, 255), width=4)
        draw.text(
            (box[0], max(0, box[1] - 18)),
            f"(id={index}, prob={record['score']:.2f})",
            fill=(*color, 255),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255),
        )

    overlay_image.convert("RGB").save(output_dir / "overlay.jpg", quality=95)
    (output_dir / "predictions.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    return records


def save_dataset_results(
    target: Image.Image,
    masks: list[np.ndarray],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    paths: DatasetModePaths,
    frame_stem: str,
    object_name: str,
    object_id: str,
    source_image: Path,
    prompt_preview: Image.Image | None,
    save_diagnostics: bool,
) -> dict:
    """Publish one frame in the canonical Gemini dataset output layout."""

    frame_component = _safe_path_component(frame_stem, "frame stem")
    _safe_path_component(object_name, "object name")
    if re.fullmatch(r"obj_\d{3}", object_id) is None:
        raise ValueError(
            f"Dataset-mode object_id must match 'obj_NNN', got {object_id!r}."
        )
    if save_diagnostics and prompt_preview is None:
        raise ValueError(
            "prompt_preview is required when dataset diagnostics are enabled."
        )
    width, height = target.size
    candidates = _dataset_prediction_records(
        masks,
        boxes,
        scores,
        width=width,
        height=height,
    )
    selected_index = (
        int(torch.argmax(scores).detach().cpu()) if len(candidates) else None
    )
    for candidate in candidates:
        candidate["selected"] = candidate["index"] == selected_index

    bbox_path = paths.bbox_dir / f"{frame_component}.json"
    inst_seg_path = paths.inst_seg_dir / f"{frame_component}.png"
    mapping_path = paths.inst_seg_dir / f"semantics_mapping_{frame_component}.json"
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :] = UNLABELLED_RGBA
    bbox_document: dict[str, list[int]] = {}
    if selected_index is not None:
        selected_mask = np.asarray(masks[selected_index], dtype=bool)
        rgba[selected_mask] = TARGET_RGBA
        bbox_document[object_id] = candidates[selected_index]["bbox_xyxy"]

    mapping_document = {
        str(BACKGROUND_RGBA): {"class": "BACKGROUND"},
        str(UNLABELLED_RGBA): {"class": "UNLABELLED"},
        str(TARGET_RGBA): {"class": object_id},
    }
    overlay: Image.Image | None = None
    if save_diagnostics:
        overlay = _render_dataset_overlay(target, masks, candidates)

    _atomic_write_json(bbox_path, bbox_document)
    _atomic_save_image(Image.fromarray(rgba, mode="RGBA"), inst_seg_path, "PNG")
    _atomic_write_json(mapping_path, mapping_document)

    diagnostic_paths: dict[str, str] = {}
    if save_diagnostics:
        assert prompt_preview is not None  # validated before publishing artifacts
        assert overlay is not None
        diagnostic_dir = paths.diagnostics_object_dir / frame_component
        overlay_path = diagnostic_dir / "overlay.jpg"
        stitched_path = diagnostic_dir / "stitched_prompt.jpg"
        _atomic_save_image(overlay, overlay_path, "JPEG", quality=95)
        _atomic_save_image(
            prompt_preview.convert("RGB"), stitched_path, "JPEG", quality=92
        )
        diagnostic_paths = {
            "overlay": _dataset_relative_path(overlay_path, paths.root),
            "stitched_prompt": _dataset_relative_path(stitched_path, paths.root),
        }

    return {
        "frame_id": frame_component,
        "image": _dataset_relative_path(source_image, paths.root),
        "image_sha256": _sha256_file(source_image),
        "image_width": width,
        "image_height": height,
        "status": "ok" if selected_index is not None else "no_predictions",
        "prediction_count": len(candidates),
        "selected_prediction_index": selected_index,
        "predictions": candidates,
        "artifacts": {
            "bbox": _dataset_relative_path(bbox_path, paths.root),
            "inst_seg": _dataset_relative_path(inst_seg_path, paths.root),
            "semantics_mapping": _dataset_relative_path(mapping_path, paths.root),
            **diagnostic_paths,
        },
    }


def _dataset_prediction_records(
    masks: list[np.ndarray],
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    width: int,
    height: int,
) -> list[dict]:
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"Prediction boxes must have shape Nx4, got {boxes.shape}.")
    if scores.ndim != 1:
        raise ValueError(f"Prediction scores must have shape N, got {scores.shape}.")
    if len(masks) != len(boxes) or len(boxes) != len(scores):
        raise ValueError(
            "Prediction masks, boxes, and scores must have the same length."
        )

    records: list[dict] = []
    for index, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (height, width):
            raise ValueError(
                f"Prediction mask {index} has shape {mask_array.shape}; "
                f"expected {(height, width)}."
            )
        model_bbox = [float(value) for value in box.detach().cpu().tolist()]
        records.append(
            {
                "index": index,
                "score": float(score.detach().cpu()),
                "bbox_xyxy": _virtual_compatible_bbox(
                    model_bbox, width=width, height=height
                ),
                "model_bbox_xyxy": model_bbox,
                "mask_pixel_count": int(mask_array.sum()),
            }
        )
    return records


def _virtual_compatible_bbox(
    bbox: list[float],
    *,
    width: int,
    height: int,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(width, math.floor(x1))),
        max(0, min(height, math.floor(y1))),
        max(0, min(width, math.ceil(x2))),
        max(0, min(height, math.ceil(y2))),
    ]


def _render_dataset_overlay(
    target: Image.Image,
    masks: list[np.ndarray],
    records: list[dict],
) -> Image.Image:
    overlay_image = target.convert("RGBA")
    colors = [(0, 255, 255), (255, 0, 255), (50, 120, 255), (48, 220, 80)]
    for mask, record in zip(masks, records):
        mask_image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
        color = colors[record["index"] % len(colors)]
        color_layer = Image.new("RGBA", target.size, (*color, 0))
        color_layer.putalpha(mask_image.point(lambda value: 100 if value else 0))
        overlay_image = Image.alpha_composite(overlay_image, color_layer)

    draw = ImageDraw.Draw(overlay_image)
    for record in records:
        color = colors[record["index"] % len(colors)]
        box = record["model_bbox_xyxy"]
        draw.rectangle(box, outline=(*color, 255), width=4)
        draw.text(
            (box[0], max(0, box[1] - 18)),
            f"(id={record['index']}, prob={record['score']:.2f})",
            fill=(*color, 255),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255),
        )
    return overlay_image.convert("RGB")


def _atomic_write_json(path: Path, document: object) -> None:
    payload = (json.dumps(document, indent=2) + "\n").encode("utf-8")
    _atomic_write_bytes(path, payload)


def _atomic_save_image(
    image: Image.Image,
    path: Path,
    image_format: str,
    **save_kwargs,
) -> None:
    stream = BytesIO()
    image.save(stream, format=image_format, **save_kwargs)
    _atomic_write_bytes(path, stream.getvalue())


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o664,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _dataset_relative_path(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Dataset artifact path escapes dataset root: {path}") from exc


def build_dataset_summary(
    *,
    paths: DatasetModePaths,
    object_name: str,
    object_id: str,
    requested_object_name: str,
    identity: RealObjectIdentity | None,
    manifest_entry: dict,
    reference_path: Path,
    reference_box: list[float],
    input_size: int,
    confidence_threshold: float,
    checkpoint: Path | None,
    save_diagnostics: bool,
) -> dict:
    """Create run-level provenance; frame results are appended by the caller."""

    summary: dict = {
        "schema_version": "1.0",
        "artifact_type": "sam3_real_pcs_summary",
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dataset",
        "camera_name": paths.camera_name,
        "object_name": object_name,
        "object_id": object_id,
        "requested_object_name": requested_object_name,
        "reference_image": _dataset_relative_path(reference_path, paths.root),
        "reference_bbox_xyxy": reference_box,
        "input_size": input_size,
        "confidence_threshold": confidence_threshold,
        "checkpoint": str(checkpoint.expanduser().resolve()) if checkpoint else None,
        "save_diagnostics": save_diagnostics,
        "output_contract": {
            "selection_policy": "highest_score_single_manipulation_target",
            "bbox_format": "xyxy_integer_half_open_covering_box",
            "inst_seg_mode": "rgba_uint8",
            "target_rgba": list(TARGET_RGBA),
            "unlabelled_rgba": list(UNLABELLED_RGBA),
        },
        "provenance": {
            "capture_manifest": _dataset_relative_path(
                paths.capture_manifest, paths.root
            ),
            "capture_manifest_sha256": _sha256_file(paths.capture_manifest),
            "reference_index": _dataset_relative_path(
                paths.reference_index, paths.root
            ),
            "reference_index_sha256": _sha256_file(paths.reference_index),
            "image_directory": _dataset_relative_path(paths.image_dir, paths.root),
        },
        "images": [],
    }
    if identity is not None:
        summary.update(identity.metadata())
    else:
        for field in (
            "old_name",
            "catalog_object_name",
            "key_namespace",
            "catalog_year",
            "object_catalog_sha256",
        ):
            if field in manifest_entry:
                summary[field] = manifest_entry[field]
    return summary


def finalize_dataset_summary(paths: DatasetModePaths, summary: dict) -> None:
    """Write the run summary last so it acts as the completed-run marker."""

    images = summary.get("images", [])
    summary["status"] = "complete"
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["frame_count"] = len(images)
    summary["frames_with_predictions"] = sum(
        item.get("status") == "ok" for item in images
    )
    summary["frames_without_predictions"] = sum(
        item.get("status") == "no_predictions" for item in images
    )
    _atomic_write_json(paths.summary_path, summary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_previous_results(output_dir: Path) -> None:
    """재실행 시 해당 target의 과거 생성물만 제거한다."""
    for mask_path in output_dir.glob("mask_*.png"):
        mask_path.unlink()
    for filename in ("overlay.jpg", "predictions.json", "stitched_prompt.jpg"):
        generated_path = output_dir / filename
        if generated_path.is_file():
            generated_path.unlink()


def main() -> None:
    args = parse_args()
    if args.input_size % 2:
        raise ValueError("--input-size must be even for equal reference/target tiles.")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 real batch inference requires a CUDA GPU.")

    identity = (
        resolve_real_object(args.object_name, args.objects_metadata)
        if args.objects_metadata is not None
        else None
    )
    object_name = identity.applied_key if identity is not None else args.object_name
    dataset_paths = (
        dataset_mode_paths(args.dataset_root, args.camera_name, object_name)
        if args.dataset_root is not None
        else None
    )
    image_dir = dataset_paths.image_dir if dataset_paths else args.image_dir
    reference_index = (
        dataset_paths.reference_index if dataset_paths else args.reference_index
    )
    scene_meta = dataset_paths.capture_manifest if dataset_paths else args.scene_meta
    reference_path, annotation = load_reference_annotation(
        object_name,
        reference_index,
        expected_identity=identity,
    )
    if annotation.get("bbox_format") != "xyxy":
        raise ValueError("The pilot implementation requires bbox_format='xyxy'.")
    reference_box = [float(value) for value in annotation["bbox"]]
    reference = load_oriented_rgb(reference_path)
    expected_size = (annotation.get("image_width"), annotation.get("image_height"))
    if None not in expected_size and reference.size != expected_size:
        raise ValueError(
            f"Reference size changed after labeling: JSON={expected_size}, "
            f"current={reference.size}"
        )

    image_paths = load_target_image_paths(
        object_name,
        image_dir,
        scene_meta,
        expected_identity=identity,
        camera_name=dataset_paths.camera_name if dataset_paths else None,
    )
    manifest_entry = (
        load_manifest_object_entry(object_name, scene_meta)
        if dataset_paths is not None
        else None
    )
    object_id = (
        identity.object_id
        if identity is not None
        else manifest_entry.get("object_id") if manifest_entry is not None else None
    )
    if dataset_paths is not None and not isinstance(object_id, str):
        raise ValueError(
            f"Dataset-mode manifest object {object_name!r} has no object_id."
        )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    build_kwargs = {}
    if args.checkpoint is not None:
        build_kwargs.update(checkpoint_path=str(args.checkpoint), load_from_HF=False)
    model = build_sam3_image_model(**build_kwargs)
    processor = Sam3Processor(
        model,
        resolution=args.input_size,
        confidence_threshold=args.confidence_threshold,
    )

    tile_width = args.input_size
    tile_height = args.input_size // 2
    reference_tile, reference_transform = letterbox(reference, tile_width, tile_height)
    prompt_box = map_box_to_canvas(reference_box, reference_transform, tile_y=0)
    normalized_prompt = (
        normalize_bbox(
            box_xyxy_to_cxcywh(torch.tensor(prompt_box).view(1, 4)),
            args.input_size,
            args.input_size,
        )
        .flatten()
        .tolist()
    )

    if dataset_paths is not None:
        summary = build_dataset_summary(
            paths=dataset_paths,
            object_name=object_name,
            object_id=object_id,
            requested_object_name=args.object_name,
            identity=identity,
            manifest_entry=manifest_entry,
            reference_path=reference_path,
            reference_box=reference_box,
            input_size=args.input_size,
            confidence_threshold=args.confidence_threshold,
            checkpoint=args.checkpoint,
            save_diagnostics=args.save_diagnostics,
        )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "object_name": object_name,
            "reference_image": str(reference_path),
            "reference_bbox_xyxy": reference_box,
            "input_size": args.input_size,
            "confidence_threshold": args.confidence_threshold,
            "images": [],
        }
        if identity is not None:
            summary.update(identity.metadata())
            summary["requested_object_name"] = args.object_name

    for target_path in image_paths:
        target = load_oriented_rgb(target_path)
        target_tile, target_transform = letterbox(target, tile_width, tile_height)
        canvas = Image.new("RGB", (args.input_size, args.input_size), (0, 0, 0))
        canvas.paste(reference_tile, (0, 0))
        canvas.paste(target_tile, (0, tile_height))

        prompt_preview = None
        if dataset_paths is None:
            target_output_dir = args.output_dir / target_path.stem
            target_output_dir.mkdir(parents=True, exist_ok=True)
            clear_previous_results(target_output_dir)
            prompt_preview = canvas.copy()
            ImageDraw.Draw(prompt_preview).rectangle(
                prompt_box, outline=(0, 255, 0), width=4
            )
            prompt_preview.save(target_output_dir / "stitched_prompt.jpg", quality=92)
        elif args.save_diagnostics:
            prompt_preview = canvas.copy()
            ImageDraw.Draw(prompt_preview).rectangle(
                prompt_box, outline=(0, 255, 0), width=4
            )

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            state = processor.set_image(canvas)
            output = processor.add_geometric_prompt(
                state=state, box=normalized_prompt, label=True
            )

        boxes = output["boxes"]
        masks = output["masks"]
        scores = output["scores"]
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        keep_target = centers_y >= tile_height
        target_canvas_boxes = boxes[keep_target]
        target_boxes = map_boxes_to_original(
            target_canvas_boxes, target_transform, tile_y=tile_height
        )
        target_masks = restore_masks(
            masks[keep_target],
            target_transform,
            tile_height,
            tile_height,
            args.input_size,
        )
        target_scores = scores[keep_target]
        if dataset_paths is not None:
            frame_result = save_dataset_results(
                target,
                target_masks,
                target_boxes,
                target_scores,
                paths=dataset_paths,
                frame_stem=target_path.stem,
                object_name=object_name,
                object_id=object_id,
                source_image=target_path,
                prompt_preview=prompt_preview,
                save_diagnostics=args.save_diagnostics,
            )
            summary["images"].append(frame_result)
            prediction_count = frame_result["prediction_count"]
        else:
            records = save_results(
                target, target_masks, target_boxes, target_scores, target_output_dir
            )
            summary["images"].append(
                {
                    "image": target_path.name,
                    "prediction_count": len(records),
                    "predictions": records,
                }
            )
            prediction_count = len(records)
        print(f"{target_path.name}: {prediction_count} prediction(s)")

    if dataset_paths is not None:
        finalize_dataset_summary(dataset_paths, summary)
        print(f"Saved canonical dataset results to {dataset_paths.root}")
    else:
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Saved batch results to {args.output_dir}")


if __name__ == "__main__":
    main()
