# 학습·평가·변환

이 스크립트는 기존 Colab 구성을 명령행으로 정리한 추가 코드입니다. 기존 루트의 얼굴 분류 노트북을 실행하거나 수정하지 않습니다. 이미 포함된 모델로 웹을 실행할 때는 Python 학습 환경이 필요 없습니다.

## 환경과 데이터

Python 가상환경에서 설치합니다. 저장된 원본 모델의 metadata에는 Keras 3.13.2가 기록되어 있고, 배포된 JSON에는 TensorFlow.js Converter 4.22.0이 기록되어 있습니다. 당시 전체 TensorFlow·Python 버전은 기록으로 확정할 수 없어 재현 환경을 임의로 고정하지 않았습니다. 새 학습과 변환은 설치한 버전에 따라 호환성 확인이 필요합니다.

```bash
cd browser-cnn-ear
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r training/requirements.txt
```

[MRL 데이터셋](https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset)을 별도로 내려받아 준비합니다. 각 분할은 다음 두 폴더를 가져야 합니다.

```text
train/close eyes/...
train/open eyes/...
validation/close eyes/...
validation/open eyes/...
test/close eyes/...
test/open eyes/...
```

Colab의 기존 배포본은 `data/train`, `data/test`만 사용했습니다. 이 폴더 이름이 독립 검증을 보장하지는 않습니다. 새 실험에서는 인물 기준으로 validation/test를 별도 구성하고 분할 정보를 기록하세요.

## 학습

```bash
python training/train_eye_model.py \
  --train-dir eye_dataset/data/train \
  --validation-dir eye_dataset/data/validation \
  --output training/runs/experiment-01
```

출력 디렉터리가 이미 있으면 중지합니다. 최대 15 epoch·batch 32·64×64 RGB·Adam·binary crossentropy·EarlyStopping patience 3은 기존 구성과 같습니다. 결과는 `best_eye_model.keras`, `history.json`, `run.json`입니다. 저장된 배포 모델을 자동 교체하지 않습니다. 이번 가져오기 작업에서는 재학습을 실행하지 않았습니다.

## 평가

```bash
python training/evaluate_eye_model.py \
  --data-dir eye_dataset/data/test \
  --model training/runs/experiment-01/best_eye_model.keras \
  --threshold 0.5 \
  --output training/runs/experiment-01/evaluation.json
```

`--threshold 0.5`는 기존 Colab 평가 기준, `--threshold 0.55`는 웹의 CNN 기본 분류 기준입니다. 어느 것도 EAR·시간 조건까지 포함한 웹 전체 평가가 아닙니다. 별도 `--model`을 생략하면 이 폴더의 제공된 `best_eye_model.keras`를 평가합니다.

## TensorFlow.js 변환

```bash
python -m pip install -r training/requirements-export.txt
python training/export_tfjs.py \
  --model training/runs/experiment-01/best_eye_model.keras \
  --output training/runs/experiment-01/tfjs
```

변환기는 Keras·TensorFlow 버전에 영향을 받으므로 필요하면 별도 가상환경에서 실행합니다. Keras 3의 InputLayer `batch_shape`는 TF.js가 읽는 `batch_input_shape`로 변환합니다. 변환 후 JSON만 바꾸고 이전 bin을 재사용하지 마세요. 새 모델의 구조와 가중치는 반드시 한 세트로 테스트한 뒤 `model/`에 배치해야 합니다.

제공된 웹 모델은 이미 TensorFlow.js 4.22.0에서 로드·형상 확인·합성 입력 추론을 확인했습니다. 새로 재학습/변환한 모델의 정확도와 브라우저 호환성은 별도 검증 대상입니다.
