# 구현별 비교

[프로젝트 홈](../README.md) · [문서 목차](README.md)

저장소의 세 실험은 서로 연결해 사용하는 단계가 아니라, 입력과 모델이 다른 구현입니다. 현재 GitHub Pages 실행 대상은 `browser-cnn-ear/`입니다.

| 구분 | 얼굴 전이학습 | 초기 얼굴 CNN | 눈 CNN + EAR |
| --- | --- | --- | --- |
| 위치 | `Drowsiness_Detection_CNN.ipynb` | `Drowsiness-System/Drowsiness_Detection_CNN.ipynb` | `browser-cnn-ear/` |
| 입력 | 128×128 RGB 얼굴 | 128×128 RGB 얼굴 | 64×64 RGB 눈 crop |
| 데이터 | DDD | DDD | MRL Eye Dataset |
| 구조 | 고정된 ImageNet MobileNetV2 + GAP + Dense128 + Dropout0.5 + sigmoid | Conv32/64/128 + Pooling + Flatten + Dense128 + Dropout0.5 + sigmoid | Conv32/64/128 + Pooling + GAP + Dense64 + Dropout0.4 + sigmoid |
| 라벨 0 | Drowsy | Drowsy | close eyes |
| 라벨 1 | Non Drowsy | Non Drowsy | open eyes |
| 정규화 | 입력에서 /255 | 입력에서 /255 | 모델 안 Rescaling(1/255) |
| 저장 파일 | `best_drowsiness_model.h5` | `drowsiness_face_model.h5` | `best_eye_model.keras`·TF.js JSON/bin |
| 추론 방식 | Colab 이미지 업로드 | Colab 이미지 업로드 | 웹캠 → MediaPipe → 눈 crop → CNN+EAR |

얼굴 모델 노트북의 두 출력 파일은 실행 시 생성되며 저장소에 배포 모델로 포함되어 있지 않습니다. 눈 CNN의 학습 모델·브라우저 모델·랜드마크 모델은 `browser-cnn-ear/`에 포함되어 있습니다.

## 기존 얼굴 실험

두 노트북은 `ismailnasri20/driver-drowsiness-dataset-ddd`를 다운로드하고 회전·이동·반전·확대·밝기 증강을 적용합니다. `validation_split=0.2`를 사용하며, 같은 ImageDataGenerator 설정을 validation에도 적용합니다. 저장된 출력의 데이터 수가 두 실험에서 다르므로 동일 조건의 모델 비교 실험으로 해석하지 않습니다.

루트 노트북에는 일반 CNN 코드가 주석으로 남아 있지만, 현재 실행되는 셀은 MobileNetV2입니다. MobileNetV2의 base 가중치를 고정하고 새 분류기만 학습하며, EarlyStopping과 ModelCheckpoint를 사용합니다. `Drowsiness-System/`의 초기 노트북은 일반 CNN을 15 epoch 학습하고 마지막 모델을 저장합니다.

노트북에는 `.gitignore`를 현재 디렉터리에 쓰는 셀과 Colab 이미지 업로드 셀이 있습니다. Colab의 별도 작업 디렉터리에서 확인하고, 로컬 저장소 루트에서 노트북 전체를 무작정 실행해 파일을 덮어쓰지 않도록 주의하세요.

## 웹 눈 상태 모델

TensorFlow.js 모델은 원본 Keras 모델을 변환한 별도 실행 산출물입니다. MediaPipe가 눈 위치를 찾고, 각 눈 CNN 점수와 EAR을 결합한 뒤 기록 보정·양안·2초 규칙으로 경고합니다. 하품이나 얼굴 전체의 졸음 상태를 직접 학습하는 모델은 아닙니다.

- [실행 안내](../browser-cnn-ear/README.md)
- [학습·평가·변환](../browser-cnn-ear/training/README.md)
- [실제 판정 규칙](../browser-cnn-ear/docs/IMPLEMENTATION.md)

## 결과를 비교할 때

기존 Colab의 91.53%는 눈 이미지 CNN의 검증 기록입니다. 얼굴 모델의 성능으로 사용하면 안 됩니다. 사람 단위로 분리한 독립 평가셋, 같은 입력 조건, 동일한 임계값을 준비하기 전에는 세 구현의 성능 우열을 단정하지 않습니다.
