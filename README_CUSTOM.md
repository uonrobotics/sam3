# SAM3 Custom

이 fork는 upstream [SAM 3](https://github.com/facebookresearch/sam3)에 예제 3개를
추가합니다: 가상환경 단일 이미지 추론, 실환경 reference BBox 라벨링 툴, 실환경 scene
전체 batch 추론. 기본 설치는 upstream [`README.md`](README.md)를 따르고, 이 문서는
추가된 기능만 다룹니다.

## Installation

-

---
 
## Run

### 1. 가상환경
가상환경 단일 이미지 추론

```bash
python examples/cross_image_exemplar.py \
  --reference assets/images/0001.png \
  --target assets/images/0012.png \
  --object-name whiteboard_eraser \
  --bbox-dir assets/bbox \
  --scene-meta-dir assets/scene_meta \
  --output-dir outputs/cross_image_exemplar
```

### 2. 실환경

실환경 key 정책

- `<dataset-root>/objects_metadata.csv`의 `Old_name`을 real pipeline의
  reference/manifest key로 사용한다. `Old_name`이 비어 있는 행만 `Object_name`으로
  fallback한다.
- label 툴과 추론기 둘 다 `--dataset-root`만 받고 `objects_metadata.csv`를 그 아래에서 자동으로 찾는다.
- `Object_name`, `Old_name`, `Class_name` 중 어느 값을 입력해도 같은 real key로
  정규화된다.
- 추론기는 매 frame마다 CSV에서 identity를 새로 읽어 reference index/BBox
  annotation과 대조하고, 어긋나면(예: CSV 변경) 해당 frame만 실패로 기록하고
  나머지는 계속 처리한다. `capture_manifest.json`의 frame별 identity 필드는 이
  검증에 쓰지 않는다 — frame은 `object_name`만 신뢰해서 reference를 찾는다.
- catalog를 사용해 생성한 reference에는 `object_name`, `object_id`, `old_name`,
  `catalog_object_name`, `key_namespace`, `catalog_year`,
  `object_catalog_sha256`를 동일한 이름으로 기록한다.

실환경 reference BBox 라벨링 (`reference-gen`의 객체별 고정 slot은 `0000.png`).
`--dataset-root`/`--scene` 두 인자만 받는 Tkinter 대화형 GUI로,
`reference/` 아래 모든 object 폴더를 알파벳순으로 순회한다. 라벨링 안 된 object부터
자동으로 보여주고, 드래그로 BBox를 그리면 즉시 저장 후 다음 미라벨링 object로
넘어가며, 이미 라벨링된 object는 덮어쓰기 전에 확인 팝업을 띄운다.

```bash
python examples/label_real_reference.py \
  --dataset-root /media/uon/data1/gemini \
  --scene real_v1/home/LivingRoom_Kitchen/dining_table
```

실환경 이미지 전체 추론. `--dataset-root`/`--scene` 두 인자만 받는다.
`capture_manifest.json`의 `frames`를 capture_id 순서(0000 → 0001 → …)로 순회하면서 각 frame에 이미 기록된 object_name으로 reference를 자동 매칭하고, 그 frame에 등록된 카메라를 전부 처리한다.

```bash
python examples/cross_image_exemplar_real.py \
  --dataset-root /media/uon/data1/gemini \
  --scene real_v1/home/LivingRoom_Kitchen/dining_table
```

frame마다 `bbox/<camera>/<id>.json` + `inst_seg/<camera>/<id>.png` +
`inst_seg/<camera>/semantics_mapping_<id>.json` 세 파일이 모두 있으면 완료로 보고
건너뛴다. 그래서 같은 명령을 재실행하면 중단된 지점부터 자동으로 이어서 처리되고,
크래시나 Ctrl+C로 중단돼도 안전하다. object_name이 없는(unassigned) frame은 건너뛰고,
frame 하나가 실패해도 전체 batch는 멈추지 않고 `inference_meta/sam3/errors.jsonl`에
기록한 뒤 다음 frame으로 넘어간다. 특정 object만 처리하려면 `--object-name`으로
필터링할 수 있다(완료 판정은 그대로 적용된다).

Dataset mode는 다음 입력 위치를 고정 계약으로 사용한다.

```text
/media/uon/data1/gemini/
├── objects_metadata.csv
└── real_v1/home/LivingRoom_Kitchen/dining_table/
    ├── capture_manifest.json
    ├── reference/reference_index.json
    └── rgb/top_view_camera/0000.png
```

Manifest의 `frames.<capture_id>.views.<camera>.files.rgb`를 읽어 대상 이미지를 찾고,
같은 frame의 `object_name`으로 reference를 매칭한다. 입력 RGB의 stem을 바꾸지 않고
다음 canonical output을 저장한다.

```text
/media/uon/data1/gemini/real_v1/home/LivingRoom_Kitchen/dining_table/
├── bbox/top_view_camera/0000.json
├── inst_seg/top_view_camera/0000.png
├── inst_seg/top_view_camera/semantics_mapping_0000.json
├── inference_meta/sam3/top_view_camera/0000.json
├── inference_meta/sam3/errors.jsonl
└── diagnostics/sam3/top_view_camera/
    ├── overlay/0000.jpg
    └── stitched_prompt/0000.jpg
```

(FoundationPose가 같은 scene에 결과를 낼 때는 `diagnostics/foundationpose/...`,
`inference_meta/foundationpose/...`처럼 도구 이름으로 나뉜다 — 자세한 내용은
[`FoundationPose/readme_custom.md`](../FoundationPose/readme_custom.md) 참고.)

- `bbox`는 confidence가 가장 높은 prediction 하나만
  `{"obj_120": [x1, y1, x2, y2]}` 형태로 기록한다. Prediction이 없으면 `{}`이다.
- `inst_seg`는 입력 RGB와 같은 크기의 RGBA PNG다. 선택 mask는
  `(255, 25, 25, 255)`, 나머지는 opaque black `UNLABELLED`이다. 색과
  `Class_name`의 관계는 같은 stem의 `semantics_mapping_*.json`이 정의한다.
- `inference_meta/sam3/<camera>/NNNN.json`에는 선택 여부를 포함한 모든 candidate와
  score를 frame 단위로 기록한다(`bbox`/`inst_seg`엔 선택된 결과만 남으므로, 왜 이
  결과가 나왔는지 볼 때 참고). `inference_meta/sam3/errors.jsonl`은 처리 실패한
  frame을 append 방식으로 기록한다.
- overlay와 stitched prompt는 `bbox`/`inst_seg`처럼 capture ID 순서대로 저장되므로
  이미지 뷰어로 순서대로 넘겨보며 라벨링 결과를 검수할 수 있다. 선택적 diagnostic이라
  저장하지 않으려면
  `--no-save-diagnostics`를 추가한다.
- Dataset mode는 `mask_000.png` 같은 개별 binary mask를 생성하지 않는다.

라벨링과 추론은 RGB-D 수집이 끝난 뒤 사용자가 명시적으로 실행하는 별도의 오프라인
단계다 — `scene-gen`의 Set/Enter나 capture 버튼이 SAM3 추론을 자동으로 실행하지는
않는다.

---

## 테스트

AI가 코드 수정 후 회귀 확인용으로 만든 것. 평소 직접 실행할 필요 없음.

```bash
conda activate data_gen-inf && cd sam3 && python -m pytest tests/ -q
```

- `test_cross_image_exemplar_real.py`: 좌표 변환/frame 순회·필터링/완료 판정/에러 로그/결과 저장 로직 (모델 추론 없음, GPU 불필요)
- `test_real_object_catalog.py`: 객체 이름 해석, reference_index 갱신

---

## Reference

-
