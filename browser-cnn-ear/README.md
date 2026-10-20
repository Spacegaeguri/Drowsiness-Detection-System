# CNN + EAR 브라우저 졸음 감지

웹캠에서 좌우 눈을 찾아 작은 CNN으로 눈 상태를 분류하고, EAR·최근 프레임·지속 시간을 결합해 경고음을 내는 프로토타입입니다. TensorFlow.js와 MediaPipe가 사용자 브라우저 안에서 동작합니다.

이 폴더는 기존 `Drowsiness_Detection_CNN.ipynb`, `Drowsiness-System/`과 독립된 웹 구현입니다. 기존 128×128 얼굴 모델과 이 폴더의 64×64 눈 모델은 입력·라벨·전처리가 다르므로 교체해서 사용하면 안 됩니다.

## 실행

**[브라우저에서 바로 실행하기](https://spacegaeguri.github.io/Drowsiness-Detection-System/browser-cnn-ear/)**

링크를 열어 모델 상태가 `READY`가 되면 `Start Camera`를 누르고 카메라 권한을 허용합니다. 별도 설치가 필요 없습니다. GitHub Pages는 `main` 브랜치의 저장소 루트를 배포하며, 앞으로 이 폴더 변경을 `main`에 올리면 사이트에도 반영됩니다.

### 로컬 실행

저장소 루트에서 정적 서버를 실행합니다. Python은 개발용 파일 서버 역할만 하며 추론 서버가 아닙니다.

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

`http://localhost:8000/browser-cnn-ear/`를 열고 `READY`를 확인한 뒤 `Start Camera`를 누릅니다. 카메라와 경고음은 버튼 클릭 이후 시작합니다. 종료하려면 탭을 닫거나 브라우저의 카메라 권한을 중지합니다.

HTTPS 정적 호스팅에도 폴더째 올릴 수 있습니다. 모델 경로는 `app.js` 위치를 기준으로 계산하므로 저장소명이나 하위 경로에 의존하지 않습니다. GitHub Pages는 2026-10-21에 활성화했습니다.

- 카메라는 HTTPS 또는 localhost 보안 컨텍스트가 필요합니다. `file://`로 직접 열지 마세요.
- TensorFlow.js와 MediaPipe JavaScript/WASM은 CDN에서 내려받으므로 처음 로드할 때 인터넷이 필요합니다.
- 앱은 영상 업로드·녹화·서버 전송 기능을 구현하지 않습니다. 외부 라이브러리 다운로드와 영상 전송은 다른 동작입니다.
- 실제 경고는 기기 음량, 브라우저 권한과 카메라 상태에 영향을 받습니다.

## 구성

| 경로 | 역할 |
| --- | --- |
| `index.html`, `app.js` | 웹 화면, 카메라, CNN·EAR 판단, 시간 측정, 경고 |
| `model/model.json`, `model/*.bin` | 바로 실행하는 TensorFlow.js 모델 |
| `best_eye_model.keras` | 원본 Keras 눈 상태 모델 |
| `face_landmarker.task` | 얼굴 랜드마크 모델 |
| `training/train_eye_model.py` | Colab 학습 구성을 옮긴 재학습 스크립트 |
| `training/evaluate_eye_model.py` | 분류 보고서·혼동행렬을 JSON으로 저장 |
| `training/export_tfjs.py` | Keras 모델을 TensorFlow.js로 변환 |
| `docs/IMPLEMENTATION.md` | EAR 공식, 실제 분기, 프레임 보정과 오류 상태 |
| `docs/EVALUATION.md` | 기존 Colab 수치와 검증 데이터 재사용 한계 |
| `docs/PRESENTATION_QA.md` | 기술 선택·차별성·예상 질문과 답변 |
| `reference/` | 제공받은 원본 설명 문서 |
| `SOURCES.md` | 원본 저장소·커밋·수정 범위·출처 |

## 실제 판정 규칙

한쪽 눈에서 다음 중 하나를 만족하면 그 분석 프레임을 `CLOSED`로 처리합니다.

1. `EAR < 0.15`
2. `openScore < 0.55`이면서 `EAR < 0.18`
3. `closedConfidence = 1 - openScore >= 0.85`이면서 CNN 기본 판정도 `CLOSED`

EAR이 높아도 3번 조건으로 `CLOSED`가 될 수 있습니다. `closedConfidence`는 시그모이드 출력에서 계산한 점수이며, 확률 보정을 검증한 신뢰도는 아닙니다.

각 눈에서 최근 **최대 5회 분석 결과** 중 CLOSED 비율이 60% 이상이면 보정 후 CLOSED가 됩니다. 시작 시 5개가 쌓일 때까지 기다리지 않고 현재 기록 길이로 나눕니다. 두 눈의 보정 결과가 모두 CLOSED인 상태가 2초 이상 유지되면 `SLEEPING`과 경고음을 출력합니다. 원본 센서 프레임 수나 일정 FPS를 보장하는 알고리즘은 아닙니다.

## 평가와 범위

기존 Colab 출력은 3,223장에 대해 정확도 **91.53%**, 감긴 눈 recall **100%**, 열린 눈 recall 약 **83.52%**입니다. 같은 `test_ds`를 EarlyStopping과 최적 모델 선택에도 사용했으므로 독립 테스트 성능으로 해석할 수 없습니다. 이 수치는 CNN 단독·임계값 0.5 평가이며, 웹의 CNN+EAR·0.55·시간 조건 전체 시스템 성능이 아닙니다. 가져온 모델을 이번 작업에서 재학습하거나 이 수치를 재측정하지 않았습니다.

현재 한 명의 얼굴과 눈 감김을 대상으로 합니다. 하품·머리 자세·실차 주행 검증은 구현 범위 밖입니다. 자세한 내용은 [평가 문서](docs/EVALUATION.md)를 참고하세요.

## 출처

웹 코드와 모델은 [gdpwemf/driver-drowsiness-detection](https://github.com/gdpwemf/driver-drowsiness-detection)에서 가져왔고 출처와 원본 커밋을 보존했습니다. [SOURCES.md](SOURCES.md)에 변경 내역과 사용 조건 확인 사항을 정리했습니다.
