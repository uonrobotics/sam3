import hashlib
import json

import pytest

from examples.label_real_reference import update_reference_index
from examples.real_object_catalog import (
    RealObjectCatalogError,
    resolve_real_object,
)


HEADER = "No,Object_name,Class_name,Old_name,Year\n"


def write_catalog(tmp_path, *rows, bom=False):
    text = HEADER + "".join(f"{row}\n" for row in rows)
    raw = text.encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path = tmp_path / "objects_metadata.csv"
    path.write_bytes(raw)
    return path, raw


@pytest.mark.parametrize(
    ("query", "matched_field"),
    [
        ("disposable_paper_cup", "Object_name"),
        (" PAPER_CUP ", "Old_name"),
        ("OBJ_120", "Class_name"),
    ],
)
def test_resolver_normalizes_all_namespaces_to_old_name(tmp_path, query, matched_field):
    catalog, raw = write_catalog(
        tmp_path,
        "120,disposable_paper_cup,obj_120,paper_cup,2025",
        bom=True,
    )

    identity = resolve_real_object(query, catalog)

    assert identity.applied_key == "paper_cup"
    assert identity.object_id == "obj_120"
    assert identity.catalog_object_name == "disposable_paper_cup"
    assert identity.old_name == "paper_cup"
    assert identity.key_namespace == "Old_name"
    assert identity.catalog_year == 2025
    assert identity.object_catalog_sha256 == hashlib.sha256(raw).hexdigest()
    assert identity.matched_field == matched_field


def test_resolver_falls_back_to_object_name_when_old_name_is_empty(tmp_path):
    catalog, _ = write_catalog(tmp_path, "999,new_object,obj_999,,2025")

    identity = resolve_real_object("obj_999", catalog)

    assert identity.applied_key == "new_object"
    assert identity.old_name is None
    assert identity.key_namespace == "Object_name"


def test_resolver_rejects_cross_namespace_alias_collision(tmp_path):
    catalog, _ = write_catalog(
        tmp_path,
        "1,current_a,obj_001,legacy_a,2025",
        "2,legacy_a,obj_002,legacy_b,2025",
    )

    with pytest.raises(RealObjectCatalogError, match="different objects"):
        resolve_real_object("legacy_a", catalog)


def test_resolver_rejects_no_and_class_name_mismatch(tmp_path):
    catalog, _ = write_catalog(
        tmp_path,
        "1,current_a,obj_001,legacy_a,2025",
        "2,current_b,obj_001,legacy_b,2025",
    )

    with pytest.raises(RealObjectCatalogError, match="expected 'obj_002'"):
        resolve_real_object("obj_001", catalog)


def test_reference_index_records_canonical_identity_metadata(tmp_path):
    catalog, _ = write_catalog(
        tmp_path,
        "120,disposable_paper_cup,obj_120,paper_cup,2025",
    )
    identity = resolve_real_object("disposable_paper_cup", catalog)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference_image = reference_dir / "reference.jpg"
    bbox_annotation = reference_dir / "reference.json"
    reference_image.write_bytes(b"image")
    bbox_annotation.write_text("{}", encoding="utf-8")
    index_path = reference_dir / "reference_index.json"

    update_reference_index(
        index_path=index_path,
        object_name=identity.applied_key,
        reference_image=reference_image,
        bbox_annotation=bbox_annotation,
        identity_metadata=identity.metadata(),
    )

    entry = json.loads(index_path.read_text(encoding="utf-8"))["objects"]["paper_cup"]
    assert entry["reference_image"] == "reference.jpg"
    assert entry["bbox_annotation"] == "reference.json"
    assert entry["object_name"] == "paper_cup"
    assert entry["object_id"] == "obj_120"
    assert entry["old_name"] == "paper_cup"
    assert entry["catalog_object_name"] == "disposable_paper_cup"
    assert entry["key_namespace"] == "Old_name"
    assert entry["catalog_year"] == 2025
    assert entry["object_catalog_sha256"] == identity.object_catalog_sha256


def test_reference_index_without_catalog_keeps_legacy_shape(tmp_path):
    reference_image = tmp_path / "reference.jpg"
    bbox_annotation = tmp_path / "reference.json"
    reference_image.write_bytes(b"image")
    bbox_annotation.write_text("{}", encoding="utf-8")
    index_path = tmp_path / "reference_index.json"

    update_reference_index(
        index_path=index_path,
        object_name="wire_tracker",
        reference_image=reference_image,
        bbox_annotation=bbox_annotation,
    )

    entry = json.loads(index_path.read_text(encoding="utf-8"))["objects"][
        "wire_tracker"
    ]
    assert entry == {
        "reference_image": "reference.jpg",
        "bbox_annotation": "reference.json",
    }


def test_reference_index_supports_object_scoped_reference_gen_paths(tmp_path):
    reference_root = tmp_path / "reference"
    object_dir = reference_root / "paper_cup"
    object_dir.mkdir(parents=True)
    reference_image = object_dir / "0000.png"
    bbox_annotation = object_dir / "0000.json"
    reference_image.write_bytes(b"image")
    bbox_annotation.write_text("{}", encoding="utf-8")
    index_path = reference_root / "reference_index.json"

    update_reference_index(
        index_path=index_path,
        object_name="paper_cup",
        reference_image=reference_image,
        bbox_annotation=bbox_annotation,
    )

    entry = json.loads(index_path.read_text(encoding="utf-8"))["objects"][
        "paper_cup"
    ]
    assert entry == {
        "reference_image": "paper_cup/0000.png",
        "bbox_annotation": "paper_cup/0000.json",
    }
