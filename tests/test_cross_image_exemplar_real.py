import json

import numpy as np
import pytest
import torch
from PIL import Image

from examples.cross_image_exemplar_real import (
    BACKGROUND_RGBA,
    TARGET_RGBA,
    UNLABELLED_RGBA,
    build_dataset_summary,
    dataset_mode_paths,
    finalize_dataset_summary,
    letterbox,
    load_oriented_rgb,
    load_reference_annotation,
    load_target_image_paths,
    map_box_to_canvas,
    map_boxes_to_original,
    parse_args,
    restore_masks,
    save_dataset_results,
    save_results,
)
from examples.real_object_catalog import RealObjectIdentity


@pytest.mark.parametrize("image_size", [(1000, 1000), (1600, 900), (900, 1600)])
def test_letterbox_bbox_round_trip(image_size):
    image = Image.new("RGB", image_size)
    _, transform = letterbox(image, tile_width=1008, tile_height=504)
    original_box = [100.0, 150.0, image_size[0] - 100.0, image_size[1] - 150.0]
    canvas_box = map_box_to_canvas(original_box, transform, tile_y=504)
    restored = map_boxes_to_original(torch.tensor([canvas_box]), transform, tile_y=504)
    assert restored[0].tolist() == pytest.approx(original_box, abs=1e-3)


@pytest.mark.parametrize("image_size", [(1000, 1000), (1600, 900), (900, 1600)])
def test_restore_mask_removes_letterbox_padding(image_size):
    image = Image.new("RGB", image_size)
    _, transform = letterbox(image, tile_width=1008, tile_height=504)
    canvas_mask = torch.zeros((1, 1, 1008, 1008), dtype=torch.bool)
    x1 = transform.offset_x
    y1 = 504 + transform.offset_y
    x2 = x1 + transform.resized_width
    y2 = y1 + transform.resized_height
    canvas_mask[:, :, y1:y2, x1:x2] = True

    restored = restore_masks(
        canvas_mask,
        transform,
        tile_y=504,
        tile_height=504,
        canvas_width=1008,
    )
    assert len(restored) == 1
    assert restored[0].shape == (image_size[1], image_size[0])
    assert np.all(restored[0])


def test_restore_masks_accepts_zero_predictions():
    _, transform = letterbox(Image.new("RGB", (1000, 1000)), 1008, 504)
    restored = restore_masks(
        torch.zeros((0, 1, 1008, 1008), dtype=torch.bool),
        transform,
        tile_y=504,
        tile_height=504,
        canvas_width=1008,
    )
    assert restored == []


@pytest.mark.parametrize("orientation", [6, 8])
def test_load_oriented_rgb_applies_exif_orientation(tmp_path, orientation):
    image_path = tmp_path / f"orientation_{orientation}.jpg"
    image = Image.new("RGB", (20, 30), (255, 0, 0))
    exif = image.getexif()
    exif[274] = orientation
    image.save(image_path, exif=exif)

    oriented = load_oriented_rgb(image_path)
    assert oriented.size == (30, 20)
    assert oriented.mode == "RGB"


def test_target_images_are_selected_from_manifest(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for filename in ("selected_b.jpg", "ignored.jpg", "selected_a.png"):
        Image.new("RGB", (10, 10)).save(image_dir / filename)
    manifest_path = tmp_path / "capture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "objects": {
                    "wire_tracker": {"images": ["selected_b.jpg", "selected_a.png"]}
                }
            }
        ),
        encoding="utf-8",
    )

    selected = load_target_image_paths(
        "wire_tracker", image_dir=image_dir, scene_meta_path=manifest_path
    )
    assert [path.name for path in selected] == ["selected_b.jpg", "selected_a.png"]


def test_dataset_diagnostics_boolean_option_defaults_on_and_can_be_disabled():
    enabled = parse_args(["--object-name", "paper_cup", "--dataset-root", "/tmp/scene"])
    disabled = parse_args(
        [
            "--object-name",
            "paper_cup",
            "--dataset-root",
            "/tmp/scene",
            "--no-save-diagnostics",
        ]
    )

    assert enabled.camera_name == "top_view_camera"
    assert enabled.save_diagnostics is True
    assert disabled.save_diagnostics is False


def test_legacy_explicit_path_arguments_remain_available(tmp_path):
    args = parse_args(
        [
            "--object-name",
            "paper_cup",
            "--image-dir",
            str(tmp_path / "rgb"),
            "--reference-index",
            str(tmp_path / "reference_index.json"),
            "--scene-meta",
            str(tmp_path / "capture_manifest.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.dataset_root is None
    assert args.image_dir == tmp_path / "rgb"
    assert args.reference_index == tmp_path / "reference_index.json"
    assert args.scene_meta == tmp_path / "capture_manifest.json"
    assert args.output_dir == tmp_path / "output"


def test_target_images_prefer_requested_camera_then_fall_back_to_legacy(tmp_path):
    image_dir = tmp_path / "rgb" / "top_view_camera"
    image_dir.mkdir(parents=True)
    for filename in ("top.png", "legacy.png"):
        Image.new("RGB", (10, 10)).save(image_dir / filename)
    manifest_path = tmp_path / "capture_manifest.json"
    manifest = {
        "objects": {
            "paper_cup": {
                "images_by_camera": {
                    "top_view_camera": ["top.png"],
                    "side_view_camera": ["side.png"],
                },
                "images": [
                    "top_view_camera/legacy.png",
                    "side_view_camera/side.png",
                ],
            }
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    selected = load_target_image_paths(
        "paper_cup",
        image_dir,
        manifest_path,
        camera_name="top_view_camera",
    )
    assert [path.name for path in selected] == ["top.png"]

    manifest["objects"]["paper_cup"]["images_by_camera"].pop("top_view_camera")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selected = load_target_image_paths(
        "paper_cup",
        image_dir,
        manifest_path,
        camera_name="top_view_camera",
    )
    assert [path.name for path in selected] == ["legacy.png"]


def test_reference_loader_resolves_reference_gen_object_directory(tmp_path):
    reference_root = tmp_path / "reference"
    object_dir = reference_root / "paper_cup"
    object_dir.mkdir(parents=True)
    reference_image = object_dir / "0000.png"
    Image.new("RGB", (20, 10)).save(reference_image)
    annotation_path = object_dir / "0000.json"
    annotation_path.write_text(
        json.dumps(
            {
                "object_name": "paper_cup",
                "bbox_format": "xyxy",
                "bbox": [1, 1, 19, 9],
                "image_width": 20,
                "image_height": 10,
            }
        ),
        encoding="utf-8",
    )
    index_path = reference_root / "reference_index.json"
    index_path.write_text(
        json.dumps(
            {
                "objects": {
                    "paper_cup": {
                        "reference_image": "paper_cup/0000.png",
                        "bbox_annotation": "paper_cup/0000.json",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded_image, annotation = load_reference_annotation("paper_cup", index_path)

    assert loaded_image == reference_image.resolve()
    assert annotation["bbox"] == [1, 1, 19, 9]


def _paper_cup_identity() -> RealObjectIdentity:
    return RealObjectIdentity(
        applied_key="paper_cup",
        object_id="obj_120",
        catalog_object_name="disposable_paper_cup",
        old_name="paper_cup",
        key_namespace="Old_name",
        catalog_year=2025,
        object_catalog_sha256="abc123",
        matched_field="Old_name",
    )


def test_catalog_identity_joins_reference_bbox_and_capture_manifest(tmp_path):
    identity = _paper_cup_identity()
    identity_metadata = identity.metadata()
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference_image = reference_dir / "paper_cup.png"
    Image.new("RGB", (10, 10)).save(reference_image)
    annotation_path = reference_dir / "paper_cup.json"
    annotation_path.write_text(
        json.dumps(
            {
                **identity_metadata,
                "bbox_format": "xyxy",
                "bbox": [1, 1, 9, 9],
            }
        ),
        encoding="utf-8",
    )
    index_path = reference_dir / "reference_index.json"
    index_path.write_text(
        json.dumps(
            {
                "objects": {
                    "paper_cup": {
                        **identity_metadata,
                        "reference_image": "paper_cup.png",
                        "bbox_annotation": "paper_cup.json",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    image_dir = tmp_path / "rgb"
    image_dir.mkdir()
    Image.new("RGB", (10, 10)).save(image_dir / "0000.png")
    manifest_path = tmp_path / "capture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "object_catalog": {"sha256": identity.object_catalog_sha256},
                "objects": {
                    "paper_cup": {
                        **identity_metadata,
                        "images": ["0000.png"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded_reference, annotation = load_reference_annotation(
        "paper_cup", index_path, expected_identity=identity
    )
    targets = load_target_image_paths(
        "paper_cup",
        image_dir,
        manifest_path,
        expected_identity=identity,
    )

    assert loaded_reference == reference_image.resolve()
    assert annotation["object_id"] == "obj_120"
    assert targets == [(image_dir / "0000.png").resolve()]


def test_catalog_mode_rejects_reference_or_manifest_identity_drift(tmp_path):
    identity = _paper_cup_identity()
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"image")
    annotation_path = tmp_path / "reference.json"
    annotation_path.write_text(
        json.dumps({**identity.metadata(), "object_id": "obj_999"}),
        encoding="utf-8",
    )
    index_path = tmp_path / "reference_index.json"
    index_path.write_text(
        json.dumps(
            {
                "objects": {
                    "paper_cup": {
                        **identity.metadata(),
                        "reference_image": "reference.png",
                        "bbox_annotation": "reference.json",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="object_id='obj_999'"):
        load_reference_annotation("paper_cup", index_path, expected_identity=identity)

    image_dir = tmp_path / "rgb"
    image_dir.mkdir()
    Image.new("RGB", (10, 10)).save(image_dir / "0000.png")
    manifest_path = tmp_path / "capture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "object_catalog": {"sha256": "different"},
                "objects": {
                    "paper_cup": {
                        **identity.metadata(),
                        "images": ["0000.png"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_target_image_paths(
            "paper_cup",
            image_dir,
            manifest_path,
            expected_identity=identity,
        )


def test_target_image_cannot_escape_image_directory(tmp_path):
    image_dir = tmp_path / "rgb"
    image_dir.mkdir()
    Image.new("RGB", (10, 10)).save(tmp_path / "outside.png")
    manifest_path = tmp_path / "capture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "objects": {
                    "paper_cup": {"images": ["../outside.png"]},
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative to --image-dir"):
        load_target_image_paths("paper_cup", image_dir, manifest_path)


def test_dataset_mode_writes_canonical_highest_score_outputs_and_summary(tmp_path):
    paths = dataset_mode_paths(tmp_path, "top_view_camera", "paper_cup")
    assert paths.capture_manifest == tmp_path / "capture_manifest.json"
    assert paths.reference_index == tmp_path / "reference" / "reference_index.json"
    assert paths.image_dir == tmp_path / "rgb" / "top_view_camera"

    paths.image_dir.mkdir(parents=True)
    source_image = paths.image_dir / "0007.png"
    target = Image.new("RGB", (6, 4), (20, 30, 40))
    target.save(source_image)
    lower_score_mask = np.zeros((4, 6), dtype=bool)
    lower_score_mask[0:2, 0:2] = True
    selected_mask = np.zeros((4, 6), dtype=bool)
    selected_mask[1:4, 3:6] = True

    frame_result = save_dataset_results(
        target,
        [lower_score_mask, selected_mask],
        torch.tensor(
            [[0.2, 0.1, 1.8, 1.9], [2.2, 0.9, 5.1, 3.2]],
            dtype=torch.float32,
        ),
        torch.tensor([0.6, 0.9], dtype=torch.float32),
        paths=paths,
        frame_stem="0007",
        object_name="paper_cup",
        object_id="obj_120",
        source_image=source_image,
        prompt_preview=Image.new("RGB", (8, 8), (0, 0, 0)),
        save_diagnostics=True,
    )

    bbox_path = tmp_path / "bbox" / "top_view_camera" / "0007.json"
    mask_path = tmp_path / "inst_seg" / "top_view_camera" / "0007.png"
    mapping_path = (
        tmp_path / "inst_seg" / "top_view_camera" / "semantics_mapping_0007.json"
    )
    diagnostic_dir = (
        tmp_path / "diagnostics" / "sam3" / "top_view_camera" / "paper_cup" / "0007"
    )
    assert json.loads(bbox_path.read_text(encoding="utf-8")) == {
        "obj_120": [2, 0, 6, 4]
    }
    with Image.open(mask_path) as opened:
        rgba = np.asarray(opened)
        assert opened.mode == "RGBA"
        assert opened.size == target.size
    assert tuple(rgba[2, 4]) == TARGET_RGBA
    assert tuple(rgba[0, 0]) == UNLABELLED_RGBA
    assert not np.any(np.all(rgba == np.asarray(BACKGROUND_RGBA), axis=-1))
    assert json.loads(mapping_path.read_text(encoding="utf-8")) == {
        str(BACKGROUND_RGBA): {"class": "BACKGROUND"},
        str(UNLABELLED_RGBA): {"class": "UNLABELLED"},
        str(TARGET_RGBA): {"class": "obj_120"},
    }
    assert (diagnostic_dir / "overlay.jpg").is_file()
    assert (diagnostic_dir / "stitched_prompt.jpg").is_file()
    assert not list(tmp_path.rglob("mask_*.png"))
    assert frame_result["selected_prediction_index"] == 1
    assert [item["selected"] for item in frame_result["predictions"]] == [
        False,
        True,
    ]
    assert frame_result["artifacts"] == {
        "bbox": "bbox/top_view_camera/0007.json",
        "inst_seg": "inst_seg/top_view_camera/0007.png",
        "semantics_mapping": ("inst_seg/top_view_camera/semantics_mapping_0007.json"),
        "overlay": ("diagnostics/sam3/top_view_camera/paper_cup/0007/overlay.jpg"),
        "stitched_prompt": (
            "diagnostics/sam3/top_view_camera/paper_cup/0007/" "stitched_prompt.jpg"
        ),
    }

    paths.capture_manifest.write_text('{"objects": {}}', encoding="utf-8")
    paths.reference_index.parent.mkdir(parents=True)
    paths.reference_index.write_text('{"objects": {}}', encoding="utf-8")
    reference_path = tmp_path / "reference" / "paper_cup" / "0000.png"
    reference_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(reference_path)
    summary = build_dataset_summary(
        paths=paths,
        object_name="paper_cup",
        object_id="obj_120",
        requested_object_name="paper_cup",
        identity=_paper_cup_identity(),
        manifest_entry={"object_id": "obj_120"},
        reference_path=reference_path,
        reference_box=[0.0, 0.0, 4.0, 4.0],
        input_size=1008,
        confidence_threshold=0.5,
        checkpoint=None,
        save_diagnostics=True,
    )
    summary["images"].append(frame_result)
    finalize_dataset_summary(paths, summary)
    summary_path = (
        tmp_path
        / "inference_meta"
        / "sam3"
        / "top_view_camera"
        / "paper_cup"
        / "summary.json"
    )
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["status"] == "complete"
    assert written_summary["object_id"] == "obj_120"
    assert written_summary["frame_count"] == 1
    assert written_summary["frames_with_predictions"] == 1
    assert written_summary["images"][0]["predictions"][0]["score"] == pytest.approx(0.6)


def test_dataset_mode_no_predictions_writes_unlabelled_mask_without_diagnostics(
    tmp_path,
):
    paths = dataset_mode_paths(tmp_path, "top_view_camera", "paper_cup")
    source_image = tmp_path / "rgb" / "top_view_camera" / "0008.png"
    source_image.parent.mkdir(parents=True)
    target = Image.new("RGB", (5, 3), (255, 255, 255))
    target.save(source_image)

    result = save_dataset_results(
        target,
        [],
        torch.empty((0, 4), dtype=torch.float32),
        torch.empty((0,), dtype=torch.float32),
        paths=paths,
        frame_stem="0008",
        object_name="paper_cup",
        object_id="obj_120",
        source_image=source_image,
        prompt_preview=None,
        save_diagnostics=False,
    )

    bbox = json.loads(
        (tmp_path / "bbox" / "top_view_camera" / "0008.json").read_text(
            encoding="utf-8"
        )
    )
    assert bbox == {}
    with Image.open(tmp_path / "inst_seg" / "top_view_camera" / "0008.png") as opened:
        rgba = np.asarray(opened)
    assert np.all(rgba == np.asarray(UNLABELLED_RGBA))
    mapping = json.loads(
        (
            tmp_path / "inst_seg" / "top_view_camera" / "semantics_mapping_0008.json"
        ).read_text(encoding="utf-8")
    )
    assert mapping[str(TARGET_RGBA)] == {"class": "obj_120"}
    assert result["status"] == "no_predictions"
    assert result["selected_prediction_index"] is None
    assert "overlay" not in result["artifacts"]
    assert not (tmp_path / "diagnostics").exists()
    assert not list(tmp_path.rglob("mask_*.png"))


def test_legacy_result_writer_keeps_individual_mask_layout(tmp_path):
    target = Image.new("RGB", (5, 4), (0, 0, 0))
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True

    records = save_results(
        target,
        [mask],
        torch.tensor([[2.0, 1.0, 4.0, 3.0]]),
        torch.tensor([0.75]),
        tmp_path,
    )

    assert records[0]["score"] == pytest.approx(0.75)
    assert (tmp_path / "mask_000.png").is_file()
    assert (tmp_path / "overlay.jpg").is_file()
    assert (tmp_path / "predictions.json").is_file()
