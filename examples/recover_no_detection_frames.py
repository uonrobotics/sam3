#!/usr/bin/env python3
"""real_v1_no_det에 남은 SAM3 실패 프레임을 다른 씬의 reference로 순차 재시도한다.

Usage:
    # 씬 하나만
    python sam3/examples/recover_no_detection_frames.py \
        /nas/Dataset/Dataset_2026/real_v1 \
        /nas/Dataset/Dataset_2026/real_v1_no_det \
        Home/Uon_1F_Home/gray_sofa \
        --objects-metadata /nas/Dataset/Dataset_2026/2026_real_world_test_objects_metadata.csv \
        --apply

    # real_v1_no_det 밑의 씬 전부 한 번에
    python sam3/examples/recover_no_detection_frames.py \
        /nas/Dataset/Dataset_2026/real_v1 \
        /nas/Dataset/Dataset_2026/real_v1_no_det \
        --objects-metadata /nas/Dataset/Dataset_2026/2026_real_world_test_objects_metadata.csv \
        --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

if __package__:
    from .cross_image_exemplar_real import (
        FrameWorkItem,
        _ANSI_BOLD_GREEN,
        _atomic_write_json,
        _highlight,
        dataset_mode_paths,
        discover_frame_work_items,
        letterbox,
        load_oriented_rgb,
        load_reference_bundle,
        log_frame_error,
        map_boxes_to_original,
        resolve_scene_relative_image,
        restore_masks,
        save_dataset_results,
    )
else:  # pragma: no cover - exercised by direct CLI execution
    from cross_image_exemplar_real import (
        FrameWorkItem,
        _ANSI_BOLD_GREEN,
        _atomic_write_json,
        _highlight,
        dataset_mode_paths,
        discover_frame_work_items,
        letterbox,
        load_oriented_rgb,
        load_reference_bundle,
        log_frame_error,
        map_boxes_to_original,
        resolve_scene_relative_image,
        restore_masks,
        save_dataset_results,
    )

_ANSI_BOLD_CYAN = "\033[1;36m"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("real_v1_root", type=Path)
    parser.add_argument("real_v1_no_det_root", type=Path)
    parser.add_argument(
        "scene_relative",
        nargs="?",
        default=None,
        help=(
            "real_v1/real_v1_no_det 밑 씬 상대경로, 예: Home/Uon_1F_Home/gray_sofa. "
            "생략하면 real_v1_no_det_root 밑의 씬 전부를 순서대로 처리한다."
        ),
    )
    parser.add_argument(
        "--objects-metadata",
        type=Path,
        default=None,
        help="기본값: <real_v1_root>/objects_metadata.csv",
    )
    parser.add_argument("--input-size", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--save-diagnostics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 GPU 추론을 실행한다. 안 주면 대상 개수/레퍼런스 후보만 보여준다.",
    )
    return parser.parse_args(argv)


def discover_scene_roots(root: Path) -> list[Path]:
    """capture_manifest.json이 직접 있는 폴더를 씬 하나로 본다.

    root 자체가 이미 씬이면(직접 capture_manifest.json이 있으면) [root]만
    돌려준다. data_gen 패키지의 frame_curation.discover_scene_roots()와
    로직은 동일하지만, 이 스크립트는 sam3 서브모듈 전용 환경(data_gen
    패키지 없음)에서 돌기 때문에 그 함수를 그대로 못 가져다 쓰고 여기 다시
    만든다.
    """

    if (root / "capture_manifest.json").is_file():
        return [root]
    return sorted({p.parent for p in root.rglob("capture_manifest.json")})


def build_reference_pool(real_v1_root: Path, self_scene_root: Path) -> list[Path]:
    """real_v1_root 밑의 모든 씬(자기 자신 제외) 중 reference_index.json이
    있는 것만 골라 씬 경로 순으로 정렬해서 돌려준다.

    자기 씬은 뺀다 - 지금 no_det에 있다는 것 자체가 자기 씬의 reference로는
    이미 실패했다는 뜻이라, 다시 시도해도 결과가 같을 게 뻔하기 때문이다.
    """

    self_scene_resolved = self_scene_root.resolve()
    candidates: list[Path] = []
    for manifest_path in real_v1_root.rglob("capture_manifest.json"):
        scene_root = manifest_path.parent
        if scene_root.resolve() == self_scene_resolved:
            continue
        index_path = scene_root / "reference" / "reference_index.json"
        if index_path.is_file():
            candidates.append(index_path)
    return sorted(candidates)


def _current_status(scene_root: Path, camera_name: str, capture_id: str) -> str | None:
    """이 (frame, camera)의 지금 SAM3 상태. 아직 처리 전이면 None."""

    path = scene_root / "inference_meta" / "sam3" / camera_name / f"{capture_id}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return record.get("status")


def discover_pending_items(no_det_scene_root: Path) -> list[FrameWorkItem]:
    """no_det 씬 안에서 아직 no_predictions인 (frame, camera)만 골라낸다.

    frame_is_done()은 안 쓴다 - no-detection 프레임도 결과 파일 4개를 이미
    다 갖고 있어서(빈 bbox 등) frame_is_done()이 "완료"로 오판하기 때문이다.
    대신 inference_meta의 status를 직접 읽어서 판단한다.
    """

    manifest_path = no_det_scene_root / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Capture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    all_items = discover_frame_work_items(manifest, object_name_filter=None)
    return [
        item
        for item in all_items
        if _current_status(no_det_scene_root, item.camera_name, item.capture_id)
        == "no_predictions"
    ]


def _record_recovery_metadata(
    no_det_scene_root: Path,
    camera_name: str,
    capture_id: str,
    *,
    attempt_index: int,
    source_scene_relative: str,
) -> None:
    """어느 씬의 몇 번째 레퍼런스로 복구됐는지 inference_meta에 덧붙인다."""

    path = no_det_scene_root / "inference_meta" / "sam3" / camera_name / f"{capture_id}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["recovery"] = {
        "reference_attempt_index": attempt_index,
        "reference_source_scene": source_scene_relative,
    }
    _atomic_write_json(path, record)


def run_one_attempt(
    processor,
    item: FrameWorkItem,
    reference_index_path: Path,
    *,
    metadata_path: Path,
    tile_width: int,
    tile_height: int,
    input_size: int,
    scene_root: Path,
    save_diagnostics: bool,
) -> dict:
    """cross_image_exemplar_real.main()의 프레임 하나 처리 본문과 동일하다 -
    reference_index_path만 매번 다른 씬 것으로 바뀐다."""

    import torch  # 지연 임포트: --apply 없이 미리보기만 할 때는 torch/GPU가 필요 없다.

    bundle = load_reference_bundle(
        item.object_name,
        reference_index_path=reference_index_path,
        metadata_path=metadata_path,
        tile_width=tile_width,
        tile_height=tile_height,
        input_size=input_size,
    )
    paths = dataset_mode_paths(scene_root, item.camera_name)
    target_path = resolve_scene_relative_image(scene_root, item.rgb_relative_path)
    target = load_oriented_rgb(target_path)
    target_tile, target_transform = letterbox(target, tile_width, tile_height)
    canvas = Image.new("RGB", (input_size, input_size), (0, 0, 0))
    canvas.paste(bundle.reference_tile, (0, 0))
    canvas.paste(target_tile, (0, tile_height))

    prompt_preview = None
    if save_diagnostics:
        prompt_preview = canvas.copy()
        ImageDraw.Draw(prompt_preview).rectangle(
            bundle.prompt_box, outline=(0, 255, 0), width=4
        )

    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        state = processor.set_image(canvas)
        output = processor.add_geometric_prompt(
            state=state, box=bundle.normalized_prompt, label=True
        )

    boxes = output["boxes"]
    masks = output["masks"]
    scores = output["scores"]
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
    keep_target = centers_y >= tile_height
    target_boxes = map_boxes_to_original(
        boxes[keep_target], target_transform, tile_y=tile_height
    )
    target_masks = restore_masks(
        masks[keep_target], target_transform, tile_height, tile_height, input_size
    )
    target_scores = scores[keep_target]

    return save_dataset_results(
        target,
        target_masks,
        target_boxes,
        target_scores,
        paths=paths,
        frame_stem=item.capture_id,
        object_name=item.object_name,
        object_id=bundle.object_id,
        source_image=target_path,
        prompt_preview=prompt_preview,
        save_diagnostics=save_diagnostics,
    )


def recover_one_item(
    processor,
    item: FrameWorkItem,
    reference_pool: list[Path],
    *,
    real_v1_root: Path,
    no_det_scene_root: Path,
    metadata_path: Path,
    tile_width: int,
    tile_height: int,
    input_size: int,
    save_diagnostics: bool,
    errors_path: Path,
) -> tuple[dict, int, str] | tuple[None, None, None]:
    """레퍼런스 후보를 순서대로 시도하다가 처음 성공하는 순간 멈춘다."""

    for attempt_index, reference_index_path in enumerate(reference_pool, start=1):
        try:
            frame_result = run_one_attempt(
                processor,
                item,
                reference_index_path,
                metadata_path=metadata_path,
                tile_width=tile_width,
                tile_height=tile_height,
                input_size=input_size,
                scene_root=no_det_scene_root,
                save_diagnostics=save_diagnostics,
            )
        except Exception as exc:  # noqa: BLE001 - 후보 하나 실패해도 다음 후보로 계속
            log_frame_error(
                errors_path,
                capture_id=item.capture_id,
                camera_name=item.camera_name,
                object_name=item.object_name,
                error=exc,
            )
            continue
        if frame_result["status"] == "ok":
            source_scene_relative = (
                reference_index_path.parent.parent.relative_to(real_v1_root).as_posix()
            )
            _record_recovery_metadata(
                no_det_scene_root,
                item.camera_name,
                item.capture_id,
                attempt_index=attempt_index,
                source_scene_relative=source_scene_relative,
            )
            return frame_result, attempt_index, source_scene_relative
    return None, None, None


def process_one_scene(
    scene_relative: Path,
    *,
    real_v1_root: Path,
    no_det_root: Path,
    metadata_path: Path,
    processor,
    tile_width: int,
    tile_height: int,
    input_size: int,
    save_diagnostics: bool,
    apply: bool,
) -> tuple[int, int] | None:
    """씬 하나를 처리하고 (복구됨, 여전히 실패) 개수를 돌려준다.

    no_det 쪽에 그 씬 폴더 자체가 없으면 건너뛰고 None을 돌려준다 - 여러
    씬을 한 번에 처리할 때 하나 때문에 전체가 멈추지 않게 한다.
    """

    no_det_scene_root = (no_det_root / scene_relative).resolve()
    self_scene_root = (real_v1_root / scene_relative).resolve()

    if not no_det_scene_root.is_dir():
        print(f"  건너뜀: {no_det_scene_root}가 없습니다.\n")
        return None

    pending_items = discover_pending_items(no_det_scene_root)
    reference_pool = build_reference_pool(real_v1_root, self_scene_root)

    print(f"  재시도 대상: {len(pending_items)}개 (frame, camera) 쌍")
    print(f"  레퍼런스 후보: {len(reference_pool)}개 씬 (자기 씬 제외)")
    for index_path in reference_pool:
        print(f"    {index_path.parent.parent.relative_to(real_v1_root)}")

    if not apply:
        print()
        return 0, 0
    if not pending_items:
        print("  재시도할 대상이 없습니다.\n")
        return 0, 0
    if not reference_pool:
        print("  레퍼런스 후보가 없습니다 (다른 씬에 reference_index.json이 없음).\n")
        return 0, 0

    errors_path = no_det_scene_root / "inference_meta" / "sam3" / "recovery_errors.jsonl"
    recovered = still_failed = 0
    print()
    for item in pending_items:
        frame_result, attempt_index, source_scene = recover_one_item(
            processor,
            item,
            reference_pool,
            real_v1_root=real_v1_root,
            no_det_scene_root=no_det_scene_root,
            metadata_path=metadata_path,
            tile_width=tile_width,
            tile_height=tile_height,
            input_size=input_size,
            save_diagnostics=save_diagnostics,
            errors_path=errors_path,
        )
        if frame_result is not None:
            recovered += 1
            print(
                f"  {item.capture_id} [{item.camera_name}] {item.object_name}: "
                f"복구됨 ({attempt_index}번째 레퍼런스, {source_scene})"
            )
        else:
            still_failed += 1
            print(
                f"  {item.capture_id} [{item.camera_name}] {item.object_name}: "
                f"전부 실패 ({len(reference_pool)}개 레퍼런스 시도)"
            )

    print(f"\n  이 씬 결과: 복구됨 {recovered} / 여전히 실패 {still_failed} (총 {len(pending_items)}개 시도)\n")
    return recovered, still_failed


def main() -> None:
    args = parse_args()
    real_v1_root = args.real_v1_root.expanduser().resolve()
    no_det_root = args.real_v1_no_det_root.expanduser().resolve()

    if args.scene_relative is not None:
        scene_relatives = [Path(args.scene_relative)]
    else:
        scene_roots = discover_scene_roots(no_det_root)
        if not scene_roots:
            raise SystemExit(f"{no_det_root} 밑에서 capture_manifest.json이 있는 씬을 못 찾았습니다.")
        scene_relatives = [p.relative_to(no_det_root) for p in scene_roots]

    metadata_path = (
        args.objects_metadata.expanduser().resolve()
        if args.objects_metadata is not None
        else real_v1_root / "objects_metadata.csv"
    )
    if not metadata_path.is_file():
        raise SystemExit(f"objects_metadata.csv를 못 찾았습니다: {metadata_path}")

    mode = "적용" if args.apply else "미리보기 (--apply 없이 실행됨)"
    if len(scene_relatives) > 1:
        print(f"[{mode}] {len(scene_relatives)}개 씬 발견:")
        for relative in scene_relatives:
            print(f"  {relative}")
        print()

    processor = None
    if args.apply:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("SAM 3 재추론은 CUDA GPU가 필요합니다.")
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        build_kwargs = {}
        if args.checkpoint is not None:
            build_kwargs.update(checkpoint_path=str(args.checkpoint), load_from_HF=False)
        model = build_sam3_image_model(**build_kwargs)
        processor = Sam3Processor(
            model, resolution=args.input_size, confidence_threshold=args.confidence_threshold
        )

    tile_width = args.input_size
    tile_height = args.input_size // 2

    total_recovered = total_still_failed = 0
    for index, scene_relative in enumerate(scene_relatives, start=1):
        header = f"[{mode}] [{index}/{len(scene_relatives)}] {scene_relative}"
        print(_highlight(header, _ANSI_BOLD_CYAN))

        result = process_one_scene(
            scene_relative,
            real_v1_root=real_v1_root,
            no_det_root=no_det_root,
            metadata_path=metadata_path,
            processor=processor,
            tile_width=tile_width,
            tile_height=tile_height,
            input_size=args.input_size,
            save_diagnostics=args.save_diagnostics,
            apply=args.apply,
        )
        if result is not None:
            recovered, still_failed = result
            total_recovered += recovered
            total_still_failed += still_failed

    if not args.apply:
        print("--apply를 주면 실제로 GPU 추론을 실행합니다.")
    elif len(scene_relatives) > 1:
        print(
            _highlight(
                f"전체 요약: 복구됨 {total_recovered} / 여전히 실패 {total_still_failed}"
                f" ({len(scene_relatives)}개 씬 대상)",
                _ANSI_BOLD_GREEN,
            )
        )


if __name__ == "__main__":
    main()
