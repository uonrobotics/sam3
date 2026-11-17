"""Single-image reference helpers for the standalone live server.

Based on examples/cross_image_exemplar_real.py (660a244). Existing data_gen
examples stay unchanged and independent of this module. Compatibility tests
compare geometry and reference identity against that original implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.reference_catalog import RealObjectIdentity, resolve_real_object

GENERIC_REFERENCE = "object"


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
class ReferenceBundle:
    identity: RealObjectIdentity
    object_id: str
    reference_path: Path
    reference_box: list[float]
    reference_tile: Image.Image
    reference_transform: LetterboxTransform
    prompt_box: list[float]
    normalized_prompt: list[float]
    reference_boxes: list[list[float]]
    normalized_prompts: list[list[float]]


def generic_identity() -> RealObjectIdentity:
    return RealObjectIdentity(
        applied_key=GENERIC_REFERENCE,
        object_id=GENERIC_REFERENCE,
        catalog_object_name=GENERIC_REFERENCE,
        old_name=None,
        key_namespace="generic",
        catalog_year=0,
        object_catalog_sha256="",
        matched_field="generic",
    )


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
        raise ValueError(f"Reference index must contain an objects mapping: " f"{reference_index_path}")
    entry = index["objects"].get(object_name)
    if entry is None:
        raise KeyError(f"Object {object_name!r} not found in reference index " f"{reference_index_path}")
    if not isinstance(entry, dict):
        raise ValueError(f"Reference entry for {object_name!r} must be an object in " f"{reference_index_path}.")
    if expected_identity is not None:
        _validate_identity_metadata(
            entry,
            expected_identity,
            source=f"reference index {reference_index_path}",
        )

    reference_value = entry.get("reference_image")
    annotation_value = entry.get("bbox_annotation")
    if not all(isinstance(value, str) and value for value in (reference_value, annotation_value)):
        raise ValueError(f"Incomplete reference entry for {object_name!r} in {reference_index_path}")
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
        raise ValueError(f"Object mismatch: index={object_name!r}, " f"annotation={annotation.get('object_name')!r}")
    if expected_identity is not None:
        _validate_identity_metadata(
            annotation,
            expected_identity,
            source=f"BBox annotation {annotation_path}",
        )
    return reference_path, annotation


def _validate_identity_metadata(
    document: dict,
    identity: RealObjectIdentity,
    *,
    source: str,
) -> None:
    """Check the object name and ID; CSV export hashes are not object identity."""

    for field in ("object_name", "object_id"):
        expected = identity.metadata()[field]
        if field not in document:
            raise ValueError(f"{source} is missing catalog identity field {field!r}.")
        actual = document[field]
        if actual != expected:
            raise ValueError(f"{source} has {field}={actual!r}; expected {expected!r}.")


def reference_boxes_from_annotation(annotation: dict) -> list[list[float]]:
    """`bboxes`가 있으면 그 목록을, 없으면 단일 `bbox`를 반환한다."""
    first = [float(value) for value in annotation["bbox"]]
    if "bboxes" not in annotation:
        return [first]
    boxes = [[float(value) for value in box] for box in annotation["bboxes"]]
    if not boxes or any(len(box) != 4 for box in boxes):
        raise ValueError("bboxes must be a non-empty list of [x1, y1, x2, y2] boxes.")
    if boxes[0] != first:
        raise ValueError(f"bbox {first} must equal the first entry of bboxes {boxes[0]}.")
    return boxes


def load_reference_bundle(
    object_name: str,
    *,
    reference_index_path: Path,
    metadata_path: Path,
    tile_width: int,
    tile_height: int,
    input_size: int,
) -> ReferenceBundle:
    """Reference를 찾아 identity와 함께 검증하고, letterbox/prompt까지 미리 만든다.

    `GENERIC_REFERENCE`는 카탈로그에 없는 범용 reference(여러 종류의 물체 예시)여서 카탈로그 조회와 ID 검증을 건너뛴다.
    """
    generic = object_name == GENERIC_REFERENCE
    identity = generic_identity() if generic else resolve_real_object(object_name, metadata_path)
    reference_path, annotation = load_reference_annotation(
        object_name,
        reference_index_path,
        expected_identity=None if generic else identity,
    )
    if annotation.get("bbox_format") != "xyxy":
        raise ValueError("The pilot implementation requires bbox_format='xyxy'.")
    reference_boxes = reference_boxes_from_annotation(annotation)
    reference = load_oriented_rgb(reference_path)
    expected_size = (annotation.get("image_width"), annotation.get("image_height"))
    if None not in expected_size and reference.size != expected_size:
        raise ValueError(f"Reference size changed after labeling: JSON={expected_size}, " f"current={reference.size}")

    reference_tile, reference_transform = letterbox(reference, tile_width, tile_height)
    prompt_boxes = [map_box_to_canvas(box, reference_transform, tile_y=0) for box in reference_boxes]
    # The canvas is square. Keep normalization independent of visualization
    # utilities (which import pandas/sklearn and generate colors at import).
    normalized_prompts = (box_xyxy_to_cxcywh(torch.tensor(prompt_boxes)) / input_size).tolist()
    return ReferenceBundle(
        identity=identity,
        object_id=identity.object_id,
        reference_path=reference_path,
        reference_box=reference_boxes[0],
        reference_tile=reference_tile,
        reference_transform=reference_transform,
        prompt_box=prompt_boxes[0],
        normalized_prompt=normalized_prompts[0],
        reference_boxes=reference_boxes,
        normalized_prompts=normalized_prompts,
    )


def letterbox(image: Image.Image, tile_width: int, tile_height: int) -> tuple[Image.Image, LetterboxTransform]:
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


def map_box_to_canvas(box: list[float], transform: LetterboxTransform, tile_y: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        x1 * transform.scale + transform.offset_x,
        y1 * transform.scale + transform.offset_y + tile_y,
        x2 * transform.scale + transform.offset_x,
        y2 * transform.scale + transform.offset_y + tile_y,
    ]


def map_boxes_to_original(boxes: torch.Tensor, transform: LetterboxTransform, tile_y: int) -> torch.Tensor:
    restored = boxes.clone()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - transform.offset_x) / transform.scale
    restored[:, [1, 3]] = (restored[:, [1, 3]] - tile_y - transform.offset_y) / transform.scale
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
