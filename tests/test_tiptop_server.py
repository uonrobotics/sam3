"""Live-service geometry and BF16 serialization regression, no model weights."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from sam3 import reference_exemplar as reference
from tiptop_server.server import ReferenceEngine, scene_index


@pytest.mark.parametrize("size", [(640, 400), (1280, 720), (721, 1280)])
def test_server_geometry_agrees_with_unchanged_data_gen_example(size):
    from examples import cross_image_exemplar_real as batch

    image = Image.fromarray(np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3))
    old_tile, old_transform = batch.letterbox(image, 1008, 504)
    new_tile, new_transform = reference.letterbox(image, 1008, 504)
    assert asdict(old_transform) == asdict(new_transform)
    np.testing.assert_array_equal(np.asarray(old_tile), np.asarray(new_tile))
    old_box = batch.map_box_to_canvas([1, 2, size[0] - 3, size[1] - 4], old_transform, 504)
    new_box = reference.map_box_to_canvas([1, 2, size[0] - 3, size[1] - 4], new_transform, 504)
    assert old_box == new_box
    boxes = torch.tensor([old_box])
    torch.testing.assert_close(
        batch.map_boxes_to_original(boxes, old_transform, 504),
        reference.map_boxes_to_original(boxes, new_transform, 504),
        rtol=0,
        atol=0,
    )
    mask = torch.zeros((1, 1, 1008, 1008), dtype=torch.bool)
    mask[:, :, 600:900, 200:700] = True
    np.testing.assert_array_equal(
        batch.restore_masks(mask, old_transform, 504, 504, 1008),
        reference.restore_masks(mask, new_transform, 504, 504, 1008),
    )


@pytest.mark.parametrize("legacy_name", ["", "legacy_pepper"])
def test_server_reference_bundle_preserves_data_gen_identity_and_prompt(tmp_path, legacy_name):
    from examples import cross_image_exemplar_real as batch

    metadata = tmp_path / "catalog.csv"
    metadata.write_text("No,Object_name,Class_name,Old_name,Year\n" + f"1,pepper,obj_001,{legacy_name},2026\n")
    identity = batch.resolve_real_object("pepper", metadata)
    Image.new("RGB", (1280, 720), (80, 100, 120)).save(tmp_path / "reference.png")
    annotation = dict(
        **identity.metadata(),
        bbox_format="xyxy",
        bbox=[100, 120, 600, 650],
        image_width=1280,
        image_height=720,
    )
    (tmp_path / "bbox.json").write_text(json.dumps(annotation))
    index = tmp_path / "reference_index.json"
    index.write_text(
        json.dumps(
            dict(
                objects={
                    identity.applied_key: dict(
                        **identity.metadata(),
                        reference_image="reference.png",
                        bbox_annotation="bbox.json",
                    )
                }
            )
        )
    )
    arguments = dict(
        reference_index_path=index, metadata_path=metadata, tile_width=1008, tile_height=504, input_size=1008
    )
    old = batch.load_reference_bundle(identity.applied_key, **arguments)
    new = reference.load_reference_bundle(identity.applied_key, **arguments)
    assert asdict(old.identity) == asdict(new.identity)
    assert old.reference_path == new.reference_path
    assert old.reference_box == new.reference_box and old.prompt_box == new.prompt_box
    np.testing.assert_array_equal(old.normalized_prompt, new.normalized_prompt)
    np.testing.assert_array_equal(np.asarray(old.reference_tile), np.asarray(new.reference_tile))


def test_reference_readiness_never_runs_prediction():
    engine = object.__new__(ReferenceEngine)
    engine.inference_calls = 0
    engine.reference_bundle = lambda class_id, scene: SimpleNamespace(
        reference_path="reference.png", identity=SimpleNamespace(metadata=lambda: dict(object_name=class_id))
    )
    result = engine.reference_info("pepper", "Home/floor")
    assert result["success"] and result["class_id"] == "pepper"
    assert result["inference_performed"] is False and engine.inference_calls == 0


def test_bfloat16_predictions_restore_original_grid_and_serialize(monkeypatch, tmp_path):
    engine = object.__new__(ReferenceEngine)
    engine.args = SimpleNamespace(reference_root=tmp_path, metadata=tmp_path / "catalog.csv", confidence_threshold=0.5)
    engine.cache, engine.inference_calls = {}, 0
    bundle = SimpleNamespace(
        identity=SimpleNamespace(applied_key="pepper", metadata=lambda: dict(object_name="pepper")),
        reference_path=tmp_path / "reference.png",
        reference_tile=Image.new("RGB", (1008, 504)),
        normalized_prompt=[0.1, 0.1, 0.1, 0.1],
    )
    monkeypatch.setattr(reference, "load_reference_bundle", lambda *a, **kw: bundle)
    masks = torch.zeros((2, 1, 1008, 1008), dtype=torch.bool)
    masks[1, 0, 600:900, 600:800] = True
    engine.processor = SimpleNamespace(
        set_image=lambda image: {},
        add_geometric_prompt=lambda **kw: dict(
            boxes=torch.tensor([[10, 10, 50, 50], [600, 600, 800, 900]], dtype=torch.bfloat16),
            masks=masks,
            scores=torch.tensor([0.5, 0.75], dtype=torch.bfloat16),
        ),
    )
    result = engine.predict(np.zeros((400, 640, 3), np.uint8), "pepper", "Home/floor")
    assert result["masks"].shape == (1, 1, 400, 640)
    assert result["masks"].dtype == bool and result["masks"].any()
    assert result["boxes_xyxy"].dtype == np.float32
    assert result["scores"].tolist() == [0.75]
    assert engine.inference_calls == 1


@pytest.mark.parametrize("scene", ["/tmp/elsewhere", "../other", "Home/../../other"])
def test_reference_scene_stays_inside_configured_root(tmp_path, scene):
    with pytest.raises(ValueError):
        scene_index(tmp_path, scene)
