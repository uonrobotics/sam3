import json

import numpy as np
import pytest
import torch
from PIL import Image

from examples.cross_image_exemplar_real import (
    BACKGROUND_RGBA,
    TARGET_RGBA,
    UNLABELLED_RGBA,
    dataset_mode_paths,
    discover_frame_work_items,
    frame_is_done,
    letterbox,
    load_oriented_rgb,
    load_reference_annotation,
    log_frame_error,
    map_box_to_canvas,
    map_boxes_to_original,
    parse_args,
    resolve_scene_relative_image,
    restore_masks,
    save_dataset_results,
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


def test_dataset_diagnostics_boolean_option_defaults_on_and_can_be_disabled():
    enabled = parse_args(["--dataset-root", "/tmp/gemini", "--scene", "scene"])
    disabled = parse_args(
        [
            "--dataset-root",
            "/tmp/gemini",
            "--scene",
            "scene",
            "--no-save-diagnostics",
        ]
    )

    assert enabled.object_name is None
    assert enabled.save_diagnostics is True
    assert disabled.save_diagnostics is False


def test_parse_args_accepts_optional_object_name_filter():
    args = parse_args(
        [
            "--dataset-root",
            "/tmp/gemini",
            "--scene",
            "scene",
            "--object-name",
            "mouse",
        ]
    )
    assert args.object_name == "mouse"


def _frame(object_name, views):
    return {"object_name": object_name, "views": views}


def _view(rgb_path):
    return {"files": {"rgb": rgb_path}}


def test_discover_frame_work_items_orders_by_capture_id_as_integer():
    # "10000" sorts before "9999" lexicographically but must come after it
    # numerically once a scene passes 9999 frames.
    manifest = {
        "frames": {
            "10000": _frame("mouse", {"top_view_camera": _view("rgb/top_view_camera/10000.png")}),
            "9999": _frame("mouse", {"top_view_camera": _view("rgb/top_view_camera/9999.png")}),
            "0001": _frame("mouse", {"top_view_camera": _view("rgb/top_view_camera/0001.png")}),
        }
    }
    items = discover_frame_work_items(manifest, object_name_filter=None)
    assert [item.capture_id for item in items] == ["0001", "9999", "10000"]


def test_discover_frame_work_items_skips_unassigned_frames():
    manifest = {
        "frames": {
            "0000": _frame(None, {"top_view_camera": _view("rgb/top_view_camera/0000.png")}),
            "0001": _frame("mouse", {"top_view_camera": _view("rgb/top_view_camera/0001.png")}),
        }
    }
    items = discover_frame_work_items(manifest, object_name_filter=None)
    assert [item.capture_id for item in items] == ["0001"]


def test_discover_frame_work_items_filters_by_object_name():
    manifest = {
        "frames": {
            "0000": _frame("mouse", {"top_view_camera": _view("rgb/top_view_camera/0000.png")}),
            "0001": _frame("cereal_snack", {"top_view_camera": _view("rgb/top_view_camera/0001.png")}),
        }
    }
    items = discover_frame_work_items(manifest, object_name_filter="cereal_snack")
    assert [item.capture_id for item in items] == ["0001"]
    assert items[0].object_name == "cereal_snack"


def test_discover_frame_work_items_visits_every_camera_in_a_frame():
    manifest = {
        "frames": {
            "0000": _frame(
                "mouse",
                {
                    "top_view_camera": _view("rgb/top_view_camera/0000.png"),
                    "side_view_camera": _view("rgb/side_view_camera/0000.png"),
                },
            ),
        }
    }
    items = discover_frame_work_items(manifest, object_name_filter=None)
    assert [item.camera_name for item in items] == ["side_view_camera", "top_view_camera"]
    assert all(item.capture_id == "0000" and item.object_name == "mouse" for item in items)


def test_frame_is_done_requires_all_three_output_files(tmp_path):
    assert frame_is_done(tmp_path, "top_view_camera", "0000") is False

    bbox_dir = tmp_path / "bbox" / "top_view_camera"
    bbox_dir.mkdir(parents=True)
    (bbox_dir / "0000.json").write_text("{}")
    assert frame_is_done(tmp_path, "top_view_camera", "0000") is False

    inst_seg_dir = tmp_path / "inst_seg" / "top_view_camera"
    inst_seg_dir.mkdir(parents=True)
    (inst_seg_dir / "0000.png").write_bytes(b"")
    assert frame_is_done(tmp_path, "top_view_camera", "0000") is False

    (inst_seg_dir / "semantics_mapping_0000.json").write_text("{}")
    assert frame_is_done(tmp_path, "top_view_camera", "0000") is True


def test_log_frame_error_appends_jsonl_records(tmp_path):
    errors_path = tmp_path / "inference_meta" / "sam3" / "errors.jsonl"
    log_frame_error(
        errors_path,
        capture_id="0000",
        camera_name="top_view_camera",
        object_name="mouse",
        error=ValueError("boom"),
    )
    log_frame_error(
        errors_path,
        capture_id="0001",
        camera_name="top_view_camera",
        object_name="mouse",
        error=RuntimeError("boom again"),
    )
    lines = errors_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["capture_id"] == "0000"
    assert first["error_type"] == "ValueError"
    second = json.loads(lines[1])
    assert second["capture_id"] == "0001"
    assert second["error_type"] == "RuntimeError"


def test_resolve_scene_relative_image_accepts_files_under_scene_root(tmp_path):
    image_dir = tmp_path / "rgb" / "top_view_camera"
    image_dir.mkdir(parents=True)
    (image_dir / "0000.png").write_bytes(b"fake")
    resolved = resolve_scene_relative_image(tmp_path, "rgb/top_view_camera/0000.png")
    assert resolved == (image_dir / "0000.png").resolve()


def test_resolve_scene_relative_image_rejects_escaping_paths(tmp_path):
    (tmp_path / "outside.png").write_bytes(b"fake")
    with pytest.raises(ValueError, match="relative to the scene root"):
        resolve_scene_relative_image(tmp_path, "../outside.png")


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


def test_reference_loader_validates_catalog_identity(tmp_path):
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
                "object_name": "paper_cup",
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

    loaded_reference, annotation = load_reference_annotation(
        "paper_cup", index_path, expected_identity=identity
    )

    assert loaded_reference == reference_image.resolve()
    assert annotation["object_id"] == "obj_120"


def test_reference_loader_rejects_identity_drift(tmp_path):
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


def test_dataset_mode_writes_canonical_highest_score_outputs_and_inference_meta(
    tmp_path,
):
    paths = dataset_mode_paths(tmp_path, "top_view_camera")
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
    overlay_dir = tmp_path / "diagnostics" / "sam3" / "top_view_camera" / "overlay"
    stitched_dir = (
        tmp_path / "diagnostics" / "sam3" / "top_view_camera" / "stitched_prompt"
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
    assert (overlay_dir / "0007.jpg").is_file()
    assert (stitched_dir / "0007.jpg").is_file()
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
        "overlay": "diagnostics/sam3/top_view_camera/overlay/0007.jpg",
        "stitched_prompt": "diagnostics/sam3/top_view_camera/stitched_prompt/0007.jpg",
    }

    inference_meta_path = (
        tmp_path / "inference_meta" / "sam3" / "top_view_camera" / "0007.json"
    )
    written_meta = json.loads(inference_meta_path.read_text(encoding="utf-8"))
    assert written_meta == frame_result
    assert written_meta["object_name"] == "paper_cup"
    assert written_meta["object_id"] == "obj_120"
    assert written_meta["predictions"][0]["score"] == pytest.approx(0.6)


def test_dataset_mode_overlay_draws_only_the_selected_prediction(tmp_path):
    """diagnostics/overlay must match inst_seg: only the highest-score
    candidate is drawn, not every raw detection in the target region."""

    paths = dataset_mode_paths(tmp_path, "top_view_camera")
    paths.image_dir.mkdir(parents=True)
    source_image = paths.image_dir / "0007.png"
    background = (20, 30, 40)
    # Large enough, and with the two regions far enough apart, that JPEG's
    # block-based compression can't bleed one region's color into the other.
    target = Image.new("RGB", (64, 64), background)
    target.save(source_image)
    unselected_mask = np.zeros((64, 64), dtype=bool)
    unselected_mask[0:8, 0:8] = True  # region unique to the losing candidate
    selected_mask = np.zeros((64, 64), dtype=bool)
    selected_mask[48:56, 48:56] = True  # region unique to the winning candidate

    frame_result = save_dataset_results(
        target,
        [unselected_mask, selected_mask],
        torch.tensor(
            [[0.0, 0.0, 8.0, 8.0], [48.0, 48.0, 56.0, 56.0]], dtype=torch.float32
        ),
        torch.tensor([0.1, 0.9], dtype=torch.float32),
        paths=paths,
        frame_stem="0007",
        object_name="paper_cup",
        object_id="obj_120",
        source_image=source_image,
        prompt_preview=Image.new("RGB", (8, 8), (0, 0, 0)),
        save_diagnostics=True,
    )

    assert frame_result["selected_prediction_index"] == 1
    overlay_path = (
        tmp_path / "diagnostics" / "sam3" / "top_view_camera" / "overlay" / "0007.jpg"
    )
    with Image.open(overlay_path) as opened:
        overlay = np.asarray(opened.convert("RGB")).astype(int)
    # The losing candidate's region must be left close to the plain
    # background color; the winning candidate's region must be clearly
    # tinted away from it. A loose tolerance absorbs JPEG lossy compression
    # without hiding an actual (much larger) color shift.
    assert np.allclose(overlay[2, 2], background, atol=12)
    assert not np.allclose(overlay[52, 52], background, atol=30)


def test_dataset_mode_no_predictions_writes_unlabelled_mask_without_diagnostics(
    tmp_path,
):
    paths = dataset_mode_paths(tmp_path, "top_view_camera")
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
