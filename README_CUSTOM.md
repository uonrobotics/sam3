## 공통 기능

-


## 기능 차이

1. 가상환경 PCS 객체 분할 : 가상환경 기반 reference (image exemplar)로 단일 입력 이미지 객체 분할 (가능성 검증)
2. Image Exemplar 라벨링 툴 :  실환경 reference BBox 생성 도구 (label_real_reference.py)
-> 사용 방법 : dataset-root/scene 지정 후 실행하면 Tkinter GUI가 열림. 마우스
   드래그로 BBox를 그리면 즉시 저장되고 다음 미라벨링 object로 자동 이동. 이미
   라벨링된 object를 다시 드래그하면 덮어쓰기 확인 팝업이 뜸.
3. 실환경 PCS 객체 분할 : 실환경 기반 reference (image exemplar)로 다중 입력 이미지 객체 분할


## 향후 작업

1. Gemini 336L RGB-D 카메라 연동 및 SAM3 기반 실환경 객체 분할 적용성 검증
-> 카메라 종류를 인자로 설정 (갤럭시 / Gemini 등)
2. Image Exemplar 라벨링 툴 에러 처리
3. 가상환경 Image Stitching 기반 PCS 객체 분할 구현 - 다중 입력 지원
4. 가상환경/실환경 공통화 모듈 및 어댑터 생성


## install

-


## Run

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

실환경 key 정책

- `/media/uon/data1/gemini/objects_metadata.csv`의 `Old_name`을 real pipeline의
  reference/manifest key로 사용한다. `Old_name`이 비어 있는 행만 `Object_name`으로
  fallback한다.
- `--objects-metadata`를 지정하면 `--object-name`에는 CSV의 `Object_name`,
  `Old_name`, `Class_name` 중 어느 값을 입력해도 같은 real key로 정규화된다.
- reference index, BBox annotation, capture manifest가 모두 같은 catalog snapshot과
  real key를 사용해야 한다.
- catalog를 사용해 생성한 reference에는 `object_name`, `object_id`, `old_name`,
  `catalog_object_name`, `key_namespace`, `catalog_year`,
  `object_catalog_sha256`를 동일한 이름으로 기록한다.
- `--objects-metadata`를 지정한 추론은 위 identity field와 catalog SHA-256을
  reference index, BBox, capture manifest 사이에서 strict 검증한 뒤 시작한다.

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

실환경 이미지 전체 추론

```bash
python examples/cross_image_exemplar_real.py \
  --dataset-root /media/uon/data1/gemini/real_v1/home/LivingRoom_Kitchen/dining_table \
  --camera-name top_view_camera \
  --object-name paper_cup \
  --objects-metadata /media/uon/data1/gemini/objects_metadata.csv
```

Dataset mode는 다음 입력 위치를 고정 계약으로 사용한다.

```text
/media/uon/data1/gemini/real_v1/home/LivingRoom_Kitchen/dining_table/
├── capture_manifest.json
├── reference/reference_index.json
└── rgb/top_view_camera/0000.png
```

공유 catalog는 scene 밖의 `/media/uon/data1/gemini/objects_metadata.csv`를 사용한다.

Manifest의 `objects.<applied_object>.images_by_camera.top_view_camera`를 먼저 읽고,
해당 camera key가 없으면 기존 `objects.<applied_object>.images`를 fallback으로 읽는다.
입력 RGB의 stem을 바꾸지 않고 다음 canonical output을 저장한다.

```text
/media/uon/data1/gemini/real_v1/home/LivingRoom_Kitchen/dining_table/
├── bbox/top_view_camera/0000.json
├── inst_seg/top_view_camera/0000.png
├── inst_seg/top_view_camera/semantics_mapping_0000.json
├── inference_meta/sam3/top_view_camera/paper_cup/summary.json
└── diagnostics/sam3/top_view_camera/paper_cup/0000/
    ├── overlay.jpg
    └── stitched_prompt.jpg
```

- `bbox`는 confidence가 가장 높은 prediction 하나만
  `{"obj_120": [x1, y1, x2, y2]}` 형태로 기록한다. Prediction이 없으면 `{}`이다.
- `inst_seg`는 입력 RGB와 같은 크기의 RGBA PNG다. 선택 mask는
  `(255, 25, 25, 255)`, 나머지는 opaque black `UNLABELLED`이다. 색과
  `Class_name`의 관계는 같은 stem의 `semantics_mapping_*.json`이 정의한다.
- summary에는 선택 여부를 포함한 모든 candidate, score, 입력 provenance와 frame
  status를 항상 기록한다.
- overlay와 stitched prompt만 선택적 diagnostic이다. 저장하지 않으려면
  `--no-save-diagnostics`를 추가한다.
- Dataset mode는 `mask_000.png` 같은 개별 binary mask를 생성하지 않는다.

위 라벨링과 추론 명령은 RGB-D 수집이 끝난 뒤 사용자가 명시적으로 실행하는 오프라인 단계다.
현재 `scene-gen` viewer의 Set/Enter 동작이나 capture 버튼은 SAM3 추론을 자동으로
실행하지 않는다. 향후 `Scene Gen` 버튼은 이미 수집된 manifest 기반 batch PCS를
실행한다. 실증 camera stream을 즉시 처리하는 real-time PCS는 이 버튼과 별도의
미구현 경로다.


## 테스트

AI가 코드 수정 후 회귀 확인용으로 만든 것. 평소 직접 실행할 필요 없음.

```bash
conda activate data_gen-inf && cd sam3 && python -m pytest tests/ -q
```

- `test_cross_image_exemplar_real.py`: 좌표 변환/이미지 목록 선택/결과 저장 로직 (모델 추론 없음, GPU 불필요)
- `test_real_object_catalog.py`: 객체 이름 해석, reference_index 갱신


## Reference

-
