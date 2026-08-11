import json

import numpy as np
import pytest
import torch
from PIL import Image

from examples.cross_image_exemplar_real import (
    letterbox,
    load_oriented_rgb,
    load_reference_annotation,
    load_target_image_paths,
    map_box_to_canvas,
    map_boxes_to_original,
    restore_masks,
)
from examples.real_object_catalog import RealObjectIdentity


@pytest.mark.parametrize("image_size", [(1000, 1000), (1600, 900), (900, 1600)])
def test_letterbox_bbox_round_trip(image_size):
    image = Image.new("RGB", image_size)
    _, transform = letterbox(image, tile_width=1008, tile_height=504)
    original_box = [100.0, 150.0, image_size[0] - 100.0, image_size[1] - 150.0]
    canvas_box = map_box_to_canvas(original_box, transform, tile_y=504)
    restored = map_boxes_to_original(
        torch.tensor([canvas_box]), transform, tile_y=504
    )
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
                    "wire_tracker": {
                        "images": ["selected_b.jpg", "selected_a.png"]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    selected = load_target_image_paths(
        "wire_tracker", image_dir=image_dir, scene_meta_path=manifest_path
    )
    assert [path.name for path in selected] == ["selected_b.jpg", "selected_a.png"]


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

    loaded_image, annotation = load_reference_annotation(
        "paper_cup", index_path
    )

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
        load_reference_annotation(
            "paper_cup", index_path, expected_identity=identity
        )

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
