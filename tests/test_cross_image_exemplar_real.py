import json

import numpy as np
import pytest
import torch
from PIL import Image

from examples.cross_image_exemplar_real import (
    letterbox,
    load_oriented_rgb,
    load_target_image_paths,
    map_box_to_canvas,
    map_boxes_to_original,
    restore_masks,
)


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
