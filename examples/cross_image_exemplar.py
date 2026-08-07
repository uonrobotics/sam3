#!/usr/bin/env python3
"""두 이미지를 이어 붙여 시험하는 cross-image exemplar 추론 스크립트.

SAM 3의 공식 image exemplar는 현재 처리 중인 이미지 내부의 BBox만 받는다.
이 제약을 우회하기 위해 reference를 위쪽, target을 아래쪽에 배치한 합성 이미지를
만든다. Reference 객체의 BBox를 positive exemplar로 입력한 뒤, target 절반에서
검출된 결과만 원래 target 이미지 좌표로 복원한다.

주의: Meta가 공식 지원하는 cross-image API가 아니라, 두 이미지를 하나의 canvas로
만드는 실험적 우회 방법이다. 모델 내부 코드는 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model.box_ops import box_xyxy_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.visualization_utils import normalize_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM 3 cross-image exemplar inference via image stitching."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--bbox-dir", type=Path, default=Path("assets/bbox"))
    parser.add_argument(
        "--scene-meta-dir", type=Path, default=Path("assets/scene_meta")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-size",
        type=int,
        default=1008,
        help="SAM3Processor 입력 해상도. 공식 image notebook 기본값은 1008.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="공식 image notebook과 동일한 기본 confidence threshold.",
    )
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def load_reference_box(
    reference: Path, object_name: str, bbox_dir: Path, scene_meta_dir: Path
) -> list[float]:
    """가상환경 scene metadata에서 object_name에 대응하는 GT BBox를 읽는다."""
    scene_meta_path = scene_meta_dir / f"{reference.stem}.json"
    bbox_path = bbox_dir / f"{reference.stem}.json"
    if not scene_meta_path.is_file():
        raise FileNotFoundError(f"Scene metadata not found: {scene_meta_path}")
    if not bbox_path.is_file():
        raise FileNotFoundError(f"BBox metadata not found: {bbox_path}")

    scene_meta = json.loads(scene_meta_path.read_text(encoding="utf-8"))
    object_ids = [
        object_id
        for object_id, metadata in scene_meta.get("objects", {}).items()
        if metadata.get("object_name") == object_name
    ]
    if not object_ids:
        raise KeyError(f"Object {object_name!r} not found in {scene_meta_path}")
    if len(object_ids) > 1:
        raise ValueError(
            f"Object name {object_name!r} is ambiguous in {scene_meta_path}: "
            f"{object_ids}"
        )

    bboxes = json.loads(bbox_path.read_text(encoding="utf-8"))
    object_id = object_ids[0]
    if object_id not in bboxes:
        raise KeyError(f"BBox for {object_id!r} not found in {bbox_path}")
    box = [float(value) for value in bboxes[object_id]]
    if len(box) != 4:
        raise ValueError(f"Expected XYXY BBox with four values for {object_id}")
    return box


def stitch_vertical(reference: Image.Image, target: Image.Image) -> Image.Image:
    """동일 크기의 reference와 target을 위아래로 결합한다.

    1920x1080 이미지 두 장을 좌우로 결합하면 각 이미지가 매우 가늘게 축소된다.
    위아래 결합은 1920x2160으로 거의 정사각형이므로 SAM 3의 1008x1008 resize에서
    객체 형태가 상대적으로 덜 왜곡된다.
    """
    if reference.size != target.size:
        raise ValueError(
            "Reference and target must have the same size for coordinate-safe "
            f"vertical stitching, got {reference.size} and {target.size}."
        )
    width, height = reference.size
    stitched = Image.new("RGB", (width, height * 2))
    stitched.paste(reference, (0, 0))
    stitched.paste(target, (0, height))
    return stitched


def save_results(
    *,
    target: Image.Image,
    target_masks: torch.Tensor,
    target_boxes: torch.Tensor,
    target_scores: torch.Tensor,
    output_dir: Path,
) -> None:
    """Target 좌표계의 개별 mask, 이미지 overlay, BBox JSON을 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_rgba = target.convert("RGBA")
    colors = [(255, 48, 48), (48, 220, 80), (50, 120, 255), (255, 180, 30)]
    records = []

    for index, (mask, box, score) in enumerate(
        zip(target_masks, target_boxes, target_scores)
    ):
        # SAM 3 bool mask를 검수 도구에서도 읽기 쉬운 0/255 PNG로 저장한다.
        mask_np = mask.squeeze(0).detach().cpu().numpy().astype(bool)
        mask_image = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")
        mask_image.save(output_dir / f"mask_{index:02d}.png")

        # Matplotlib graph가 아니라 원본 target 이미지에 반투명 mask를 직접 합성한다.
        color = colors[index % len(colors)]
        overlay = Image.new("RGBA", target.size, (*color, 0))
        overlay.putalpha(mask_image.point(lambda value: 100 if value else 0))
        target_rgba = Image.alpha_composite(target_rgba, overlay)

        xyxy = [float(value) for value in box.detach().cpu().tolist()]
        records.append(
            {
                "index": index,
                "score": float(score.detach().cpu()),
                "bbox_xyxy": xyxy,
            }
        )

    # BBox와 score도 이미지 픽셀에 직접 그린다. 표기 형식만 notebook에서 사용하는
    # "(id=0, prob=0.71)"을 유지한다.
    draw = ImageDraw.Draw(target_rgba)
    for record in records:
        index = record["index"]
        color = colors[index % len(colors)]
        draw.rectangle(record["bbox_xyxy"], outline=(*color, 255), width=4)
        draw.text(
            (record["bbox_xyxy"][0], max(0, record["bbox_xyxy"][1] - 18)),
            f"(id={index}, prob={record['score']:.2f})",
            fill=(*color, 255),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255),
        )

    target_rgba.convert("RGB").save(output_dir / "target_predictions.png")
    (output_dir / "predictions.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()

    # 공식 image predictor notebook과 동일한 Ampere TF32 설정.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    reference = Image.open(args.reference).convert("RGB")
    target = Image.open(args.target).convert("RGB")
    reference_box = load_reference_box(
        args.reference, args.object_name, args.bbox_dir, args.scene_meta_dir
    )
    stitched = stitch_vertical(reference, target)
    width, reference_height = reference.size

    # 추론 입력과 reference BBox가 올바른지 사람이 바로 확인할 수 있는 디버그 이미지.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stitched_debug = stitched.copy()
    debug_draw = ImageDraw.Draw(stitched_debug)
    debug_draw.rectangle(reference_box, outline=(0, 255, 0), width=5)
    stitched_debug.save(args.output_dir / "stitched_prompt.png")

    build_kwargs = {}
    if args.checkpoint is not None:
        build_kwargs.update(
            checkpoint_path=str(args.checkpoint),
            load_from_HF=False,
        )
    model = build_sam3_image_model(**build_kwargs)
    processor = Sam3Processor(
        model,
        resolution=args.input_size,
        confidence_threshold=args.confidence_threshold,
    )
    # 공식 notebook과 동일하게 CUDA 추론을 BF16 autocast로 실행한다. SAM3의 일부
    # 최적화 커널이 BF16 feature를 반환하므로 autocast 없이 FP32 layer에 전달하면
    # dtype mismatch가 발생할 수 있다.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(stitched)

        # Metadata BBox는 원본 reference의 XYXY 픽셀 좌표다. SAM3 PCS processor는
        # 합성 이미지 기준 normalized CXCYWH를 요구하므로 형식과 좌표계를 변환한다.
        box_xyxy = torch.tensor(reference_box, dtype=torch.float32).view(1, 4)
        box_cxcywh = box_xyxy_to_cxcywh(box_xyxy)
        normalized_box = normalize_bbox(
            box_cxcywh, stitched.width, stitched.height
        ).flatten().tolist()
        output = processor.add_geometric_prompt(
            state=state,
            box=normalized_box,
            label=True,
        )

    boxes = output["boxes"]
    masks = output["masks"]
    scores = output["scores"]
    # 합성 이미지 위쪽(reference)에도 원본 exemplar 자체가 검출될 수 있다.
    # BBox 중심이 경계선 아래에 있는 prediction만 target 결과로 유지한다.
    target_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    keep_target = target_centers_y >= reference_height

    # Target은 합성 이미지에서 y=reference_height부터 시작한다. BBox의 y 좌표에서
    # offset을 빼고 mask도 아래쪽 절반만 잘라 1920x1080 target 좌표계로 되돌린다.
    target_boxes = boxes[keep_target].clone()
    target_boxes[:, [1, 3]] -= reference_height
    target_boxes[:, [0, 2]] = target_boxes[:, [0, 2]].clamp(0, width)
    target_boxes[:, [1, 3]] = target_boxes[:, [1, 3]].clamp(0, target.height)
    target_masks = masks[keep_target, :, reference_height:, :]
    target_scores = scores[keep_target]

    save_results(
        target=target,
        target_masks=target_masks,
        target_boxes=target_boxes,
        target_scores=target_scores,
        output_dir=args.output_dir,
    )
    print(
        f"Saved {len(target_scores)} target prediction(s) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
