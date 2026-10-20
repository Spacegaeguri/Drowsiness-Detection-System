# 프로젝트 문서

**[라이브 데모](https://spacegaeguri.github.io/Drowsiness-Detection-System/browser-cnn-ear/)** · [프로젝트 홈](../README.md)

처음 방문했다면 웹 데모를 실행한 뒤 구현 설명과 평가 문서를 차례로 읽으면 됩니다. 재학습이 목적이라면 먼저 모델별 차이를 확인하세요.

## 목적별 안내

| 목적 | 문서 | 담긴 내용 |
| --- | --- | --- |
| 저장소 전체 이해 | [프로젝트 홈](../README.md) | 데모·구현 비교·폴더 구성 |
| 모델 선택 | [구현별 비교](PROJECT_VARIANTS.md) | 얼굴 MobileNetV2·초기 얼굴 CNN·눈 CNN+EAR |
| 웹 데모 실행 | [웹 README](../browser-cnn-ear/README.md) | 카메라·소리·로컬 실행·CDN |
| 코드 이해 | [구현 설명](../browser-cnn-ear/docs/IMPLEMENTATION.md) | 데이터 흐름·EAR·판정·시간·오류 |
| 결과 해석 | [평가 문서](../browser-cnn-ear/docs/EVALUATION.md) | 혼동행렬·Precision·Recall·F1·한계 |
| 모델 재학습 | [학습 안내](../browser-cnn-ear/training/README.md) | 환경·데이터 분할·학습·평가·TF.js 변환 |
| 발표 준비 | [발표 Q&A](../browser-cnn-ear/docs/PRESENTATION_QA.md) | CNN 선택·차별성·임계값·질의응답 |
| 검증 상태 확인 | [검증 기록](../browser-cnn-ear/docs/VERIFICATION.md) | 확인한 범위와 미실행 항목 |
| 출처 확인 | [출처 및 변경 내역](../browser-cnn-ear/SOURCES.md) | 원본 코드·모델·Colab·데이터 |

## 보존 자료

- [초기 얼굴 CNN 실험](../Drowsiness-System/README.md): 기존 노트북과 샘플 이미지 안내.
- [기존 README 원문](archive/README-original.txt): 정리 이전의 프로젝트 소개와 역할 설명. 현재 실행 방법·구현 상태는 프로젝트 홈을 따릅니다.
- [원본 웹 설명](../browser-cnn-ear/reference/upstream-readme.txt), [학습 설명](../browser-cnn-ear/reference/training-explanation.txt), [코드 설명](../browser-cnn-ear/reference/browser-code-explanation.txt): 당시 제공된 설명 자료.

## 문서 수정 기준

성능 수치를 추가할 때는 대상 모델, 평가 데이터 분할, 임계값, 표본 수를 함께 기록합니다. 얼굴 분류 정확도를 눈 상태 모델이나 실제 경고 성능으로 옮겨 쓰지 않습니다. 아직 구현하지 않은 기능은 다음 단계로 구분합니다.
