import json
import subprocess
import sys
from pathlib import Path

from examples.recover_no_detection_frames import (
    _current_status,
    _record_recovery_metadata,
    build_reference_pool,
    discover_pending_items,
    discover_scene_roots,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_manifest(scene_root: Path, frames: dict) -> None:
    _write_json(scene_root / "capture_manifest.json", {"frames": frames})


def test_build_reference_pool_excludes_self_and_scenes_without_reference(tmp_path):
    real_v1_root = tmp_path / "real_v1"
    scene_a = real_v1_root / "Home" / "floor"
    scene_b = real_v1_root / "Home" / "gray_sofa"
    scene_c = real_v1_root / "Manufactory" / "workbench"  # reference_index.json 없음

    for scene in (scene_a, scene_b, scene_c):
        _write_json(scene / "capture_manifest.json", {})
    _write_json(scene_a / "reference" / "reference_index.json", {"objects": {}})
    _write_json(scene_b / "reference" / "reference_index.json", {"objects": {}})
    # scene_c는 reference_index.json을 일부러 안 만든다.

    pool = build_reference_pool(real_v1_root, self_scene_root=scene_a)

    assert pool == [scene_b / "reference" / "reference_index.json"]


def test_build_reference_pool_sorts_by_scene_path(tmp_path):
    real_v1_root = tmp_path / "real_v1"
    scene_z = real_v1_root / "Z_site" / "z_scene"
    scene_a = real_v1_root / "A_site" / "a_scene"
    self_scene = real_v1_root / "Self" / "self_scene"

    for scene in (scene_z, scene_a, self_scene):
        _write_json(scene / "capture_manifest.json", {})
        _write_json(scene / "reference" / "reference_index.json", {"objects": {}})

    pool = build_reference_pool(real_v1_root, self_scene_root=self_scene)

    assert pool == [
        scene_a / "reference" / "reference_index.json",
        scene_z / "reference" / "reference_index.json",
    ]


def test_current_status_reads_status_field(tmp_path):
    scene = tmp_path / "scene"
    _write_json(
        scene / "inference_meta" / "sam3" / "top_view_camera" / "0000.json",
        {"status": "ok"},
    )

    assert _current_status(scene, "top_view_camera", "0000") == "ok"


def test_current_status_returns_none_when_not_yet_processed(tmp_path):
    scene = tmp_path / "scene"
    scene.mkdir()

    assert _current_status(scene, "top_view_camera", "0000") is None


def test_discover_pending_items_excludes_ok_and_never_moved_frames(tmp_path):
    scene = tmp_path / "no_det_scene"
    _write_manifest(
        scene,
        {
            "0000": {"object_name": "paper_cup", "views": ["side_view_camera", "top_view_camera"]},
            "0001": {"object_name": "paper_cup", "views": ["side_view_camera", "top_view_camera"]},
            # 0002: manifest는 원본 씬 전체의 복사본이라, 애초에 이 no_det
            # 폴더로 옮겨진 적 없는(원래 성공해서 real_v1에 남아있는) 프레임도
            # 그대로 들어있다. inference_meta가 아예 없으므로(status=None)
            # 재시도 대상에서 반드시 빠져야 한다 - 아니면 없는 이미지를
            # 대상으로 레퍼런스 후보를 전부 헛되이 시도하게 된다.
            "0002": {"object_name": "paper_cup", "views": ["side_view_camera", "top_view_camera"]},
        },
    )
    # 0000: top은 ok(재시도 대상 아님), side는 실패(재시도 대상)
    _write_json(
        scene / "inference_meta" / "sam3" / "top_view_camera" / "0000.json",
        {"status": "ok"},
    )
    _write_json(
        scene / "inference_meta" / "sam3" / "side_view_camera" / "0000.json",
        {"status": "no_predictions"},
    )
    # 0001: 둘 다 실패
    _write_json(
        scene / "inference_meta" / "sam3" / "top_view_camera" / "0001.json",
        {"status": "no_predictions"},
    )
    _write_json(
        scene / "inference_meta" / "sam3" / "side_view_camera" / "0001.json",
        {"status": "no_predictions"},
    )

    pending = discover_pending_items(scene)
    pending_pairs = {(item.capture_id, item.camera_name) for item in pending}

    assert pending_pairs == {
        ("0000", "side_view_camera"),
        ("0001", "top_view_camera"),
        ("0001", "side_view_camera"),
    }


def test_record_recovery_metadata_adds_field_without_disturbing_others(tmp_path):
    scene = tmp_path / "scene"
    path = scene / "inference_meta" / "sam3" / "side_view_camera" / "0009.json"
    _write_json(path, {"status": "ok", "prediction_count": 1})

    _record_recovery_metadata(
        scene,
        "side_view_camera",
        "0009",
        attempt_index=3,
        source_scene_relative="Home/gray_sofa",
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert record["prediction_count"] == 1
    assert record["recovery"] == {
        "reference_attempt_index": 3,
        "reference_source_scene": "Home/gray_sofa",
    }


def test_discover_scene_roots_returns_self_when_root_is_a_scene(tmp_path):
    scene = tmp_path / "scene"
    _write_json(scene / "capture_manifest.json", {})

    assert discover_scene_roots(scene) == [scene]


def test_discover_scene_roots_finds_nested_scenes(tmp_path):
    root = tmp_path / "real_v1_no_det"
    scene_a = root / "Home" / "floor"
    scene_b = root / "Manufactory" / "workbench"
    _write_json(scene_a / "capture_manifest.json", {})
    _write_json(scene_b / "capture_manifest.json", {})

    assert discover_scene_roots(root) == sorted([scene_a, scene_b])


def test_batch_dry_run_discovers_and_highlights_every_scene(tmp_path):
    """scene_relative를 생략하면 real_v1_no_det_root 밑의 씬 전부를 순서대로
    보여준다 - dry-run(--apply 없음)이라 GPU/모델은 필요 없다."""

    real_v1_root = tmp_path / "real_v1"
    no_det_root = tmp_path / "real_v1_no_det"

    for scene_relative in ("Home/floor", "Manufactory/workbench"):
        no_det_scene = no_det_root / scene_relative
        _write_json(no_det_scene / "capture_manifest.json", {"frames": {}})
        _write_json(real_v1_root / scene_relative / "capture_manifest.json", {})

    metadata_path = tmp_path / "objects_metadata.csv"
    metadata_path.write_text("No,Object_name\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "examples" / "recover_no_detection_frames.py"),
            str(real_v1_root),
            str(no_det_root),
            "--objects-metadata",
            str(metadata_path),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2개 씬 발견" in result.stdout
    assert "Home/floor" in result.stdout
    assert "Manufactory/workbench" in result.stdout
    # dry-run이라 GPU 추론 없이 대상 개수만 보여주고 끝나야 한다.
    assert "--apply를 주면 실제로 GPU 추론을 실행합니다." in result.stdout
