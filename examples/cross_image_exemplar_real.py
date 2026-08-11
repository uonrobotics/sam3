#!/usr/bin/env python3
"""하나의 실환경 reference로 여러 실환경 이미지를 일괄 추론한다.

SAM 3에는 별도 이미지의 exemplar를 target에 직접 전달하는 공개 API가 없다.
따라서 reference와 target을 종횡비가 보존된 1008x1008 canvas에 위아래로 배치하고,
reference BBox를 동일 canvas의 positive geometric prompt로 제공한다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class LetterboxTransform:
    scale: float
    offset_x: int
    offset_y: int
    resized_width: int
    resized_height: int
    original_width: int
    original_height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch SAM 3 inference using a stitched real-image exemplar."
    )
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
    return parser.parse_args()


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
    image_names = entry.get("images", [])
    if not isinstance(image_names, list) or not image_names:
        raise ValueError(f"No target images registered for {object_name!r}")
    if not all(isinstance(name, str) and name for name in image_names):
        raise ValueError(
            f"Target image names for {object_name!r} must be non-empty strings."
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


def _validate_identity_metadata(
    document: dict,
    identity: RealObjectIdentity,
    *,
    source: str,
) -> None:
    """Require an artifact to match the selected immutable catalog row."""

    for field, expected in identity.metadata().items():
        if field not in document:
            raise ValueError(
                f"{source} is missing catalog identity field {field!r}."
            )
        actual = document[field]
        if actual != expected:
            raise ValueError(
                f"{source} has {field}={actual!r}; expected {expected!r}."
            )


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
    restored[:, [0, 2]] = (
        restored[:, [0, 2]] - transform.offset_x
    ) / transform.scale
    restored[:, [1, 3]] = (
        restored[:, [1, 3]] - tile_y - transform.offset_y
    ) / transform.scale
    restored[:, [0, 2]] = restored[:, [0, 2]].clamp(
        0, transform.original_width
    )
    restored[:, [1, 3]] = restored[:, [1, 3]].clamp(
        0, transform.original_height
    )
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
    reference_path, annotation = load_reference_annotation(
        object_name,
        args.reference_index,
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
        args.image_dir,
        args.scene_meta,
        expected_identity=identity,
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
    reference_tile, reference_transform = letterbox(
        reference, tile_width, tile_height
    )
    prompt_box = map_box_to_canvas(reference_box, reference_transform, tile_y=0)
    normalized_prompt = normalize_bbox(
        box_xyxy_to_cxcywh(torch.tensor(prompt_box).view(1, 4)),
        args.input_size,
        args.input_size,
    ).flatten().tolist()

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

        target_output_dir = args.output_dir / target_path.stem
        target_output_dir.mkdir(parents=True, exist_ok=True)
        clear_previous_results(target_output_dir)
        prompt_preview = canvas.copy()
        ImageDraw.Draw(prompt_preview).rectangle(prompt_box, outline=(0, 255, 0), width=4)
        prompt_preview.save(target_output_dir / "stitched_prompt.jpg", quality=92)

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
        print(f"{target_path.name}: {len(records)} prediction(s)")

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved batch results to {args.output_dir}")


if __name__ == "__main__":
    main()
