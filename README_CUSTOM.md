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

실환경 reference BBox 라벨링

```bash
python examples/label_real_reference.py \
  --image assets/reference/20260806_164933.jpg \
  --object-name wire_tracker \
  --bbox-dir assets/reference \
  --reference-index assets/reference/reference_index.json
```

실환경 이미지 전체 추론

```bash
python examples/cross_image_exemplar_real.py \
  --object-name wire_tracker \
  --image-dir assets/images2 \
  --reference-index assets/reference/reference_index.json \
  --scene-meta assets/scene_meta2/capture_manifest.json \
  --output-dir outputs/cross_image_exemplar_real
```


## Reference

-