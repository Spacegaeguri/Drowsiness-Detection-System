# 출처 및 변경 내역

## 웹 구현과 모델

- 원본: https://github.com/gdpwemf/driver-drowsiness-detection
- 가져온 커밋: `e492fe7750b3fe66adb6506e7a79c7a3aaade123`
- 가져온 날짜: 2026-10-21
- 대상: https://github.com/Spacegaeguri/Drowsiness-Detection-System/tree/main/browser-cnn-ear
- 원본 파일: `app.js`, `index.html`, `best_eye_model.keras`, `face_landmarker.task`, `model/model.json`, `model/group1-shard1of1.bin`, `read me`
- 원본 설명은 `reference/upstream-readme.txt`로 보존했습니다.

`app.js`의 원래 `/driver-drowsiness-detection` 절대 경로를 `import.meta.url` 기준의 모델 상대 경로로 변경했습니다. CNN·EAR 임계값, 최근 프레임 규칙, 2초 타이머, 알람 방식은 원본과 동일합니다. 모델 파일과 HTML도 원본을 유지합니다.

가져온 시점에 원본 저장소에 명시적인 LICENSE 파일이 없었습니다. 출처 표기는 별도의 재배포 허가나 오픈소스 라이선스 부여를 뜻하지 않습니다. 이 폴더에 임의의 라이선스를 덧붙이지 않았으며, 추가 재배포·상용 활용 시 원 저작자 및 모델·데이터 권리 조건을 확인해야 합니다.

## 학습·평가 내용

- 발표 때 확인한 Colab: https://colab.research.google.com/drive/1XnCXZxcoqrcnxo54aaQ3vkU-ULV2oery
- `training/*.py`는 당시 대화에서 확인한 Colab 셀을 명령행 도구로 다시 정리한 코드입니다. 원본 노트북을 그대로 복제한 파일이 아닙니다.
- 2026-08-31 대화에서 읽은 출력: 학습 81,675장, 평가 3,223장, 혼동행렬 `[[1566, 0], [273, 1384]]`.
- 기존 Colab 수치를 새 스크립트의 실행 결과나 가져온 체크포인트의 재검증 결과로 표기하지 않습니다.
- `reference/training-explanation.txt`, `reference/browser-code-explanation.txt`는 사용자가 제공한 학습·웹 코드 설명을 보존한 자료입니다. 줄바꿈을 LF로 통일하고 줄 끝 공백만 제거했습니다. 실제 분기의 정확한 설명은 `docs/IMPLEMENTATION.md`를 기준으로 읽으세요.

## 데이터와 라이브러리

- MRL Eye Dataset: https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset
- TensorFlow: https://www.tensorflow.org/
- Keras: https://keras.io/
- TensorFlow.js: https://www.tensorflow.org/js
- MediaPipe Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

데이터셋 원본 이미지는 이 폴더에 복제하지 않았습니다. `face_landmarker.task`와 학습 데이터는 각각의 원래 배포 조건을 따릅니다. 라이브러리 및 모델의 권리 조건을 이 저장소 전체에 대한 단일 라이선스로 대체하지 않습니다.
