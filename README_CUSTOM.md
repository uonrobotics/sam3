## 공통 기능

-


## 기능 차이

1. 가상환경 PCS 객체 분할 : 가상환경 기반 reference (image exemplar)로 단일 입력 이미지 객체 분할 (가능성 검증)
2. Image Exemplar 라벨링 툴 :  실환경 reference BBox 생성 도구 (label_real_reference.py)
-> 사용 방법 : 마우스로 BBox 생성, 엔터로 등록, C로 취소
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

실환경 reference BBox 라벨링 (`0000.png`는 실제 reference frame ID로 교체)

```bash
python examples/label_real_reference.py \
  --image /media/uon/data1/gemini/reference/paper_cup/0000.png \
  --object-name paper_cup \
  --objects-metadata /media/uon/data1/gemini/objects_metadata.csv \
  --bbox-dir /media/uon/data1/gemini/reference/paper_cup \
  --reference-index /media/uon/data1/gemini/reference/reference_index.json
```

실환경 이미지 전체 추론

```bash
python examples/cross_image_exemplar_real.py \
  --object-name paper_cup \
  --objects-metadata /media/uon/data1/gemini/objects_metadata.csv \
  --image-dir /media/uon/data1/gemini/rgb \
  --reference-index /media/uon/data1/gemini/reference/reference_index.json \
  --scene-meta /media/uon/data1/gemini/scene_meta/capture_manifest.json \
  --output-dir outputs/cross_image_exemplar_real/paper_cup
```

위 두 명령은 RGB-D 수집이 끝난 뒤 사용자가 명시적으로 실행하는 오프라인 단계다.
현재 `scene-gen` viewer의 Set/Enter 동작이나 capture 버튼은 SAM3 추론을 자동으로
실행하지 않는다. 향후 `Scene Gen` 버튼은 이미 수집된 manifest 기반 batch PCS를
실행한다. 실증 camera stream을 즉시 처리하는 real-time PCS는 이 버튼과 별도의
미구현 경로다.


## Reference

-
