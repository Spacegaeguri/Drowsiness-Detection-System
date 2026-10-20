# 초기 얼굴 CNN 실험

[프로젝트 홈](../README.md) · [구현 비교](../docs/PROJECT_VARIANTS.md) · [현재 웹 데모](https://spacegaeguri.github.io/Drowsiness-Detection-System/browser-cnn-ear/)

이 폴더는 얼굴 전체의 Drowsy / Non Drowsy 이진 분류를 실험한 초기 자료입니다. 기존 노트북과 샘플 이미지를 원래 경로에 보존합니다.

| 경로 | 내용 |
| --- | --- |
| [Drowsiness_Detection_CNN.ipynb](Drowsiness_Detection_CNN.ipynb) | 128×128 RGB 얼굴 입력의 일반 CNN 학습·이미지 업로드 추론 |
| `images/close/` | 기존에 보관된 감긴 눈 상태의 얼굴 샘플 |
| `images/open/` | 기존에 보관된 열린 눈 상태의 얼굴 샘플 |

**[Colab에서 초기 노트북 열기](https://colab.research.google.com/github/Spacegaeguri/Drowsiness-Detection-System/blob/main/Drowsiness-System/Drowsiness_Detection_CNN.ipynb)**

이 노트북은 `ismailnasri20/driver-drowsiness-dataset-ddd` 데이터를 내려받고 Conv2D 32·64·128, MaxPooling, Flatten, Dense128, Dropout0.5, sigmoid를 사용합니다. 학습 결과 `drowsiness_face_model.h5`가 생성되지만 해당 모델 파일은 이 폴더에 포함되어 있지 않습니다.

폴더 안 샘플 이미지가 학습 전체 데이터이거나 독립 테스트셋이라는 의미는 아닙니다. 루트의 MobileNetV2 노트북, `browser-cnn-ear/`의 64×64 눈 모델과 입력 및 결과를 혼동하지 마세요.
