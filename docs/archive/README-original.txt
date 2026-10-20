# 🚗 실시간 운전자 졸음 감지 시스템 (Real-time Driver Drowsiness Detection System)

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)]()
[![MediaPipe](https://img.shields.io/badge/MediaPipe-00A98F?style=flat-square&logo=google&logoColor=white)]()

단일 카메라 프레임을 활용하여 운전자의 상태를 실시간으로 분석하고, 졸음운전을 선제적으로 예방하는 비침습적(Non-invasive) 딥러닝 기반 모니터링 시스템입니다.

**[🚀 Google Colab에서 모델 학습 결과 확인하기](https://colab.research.google.com/drive/1x-pFrTud4KEFYgPbIZSUA2UrSXeD-tjf?usp=sharing)**

---

## 💡 프로젝트 배경 (Background)
졸음운전은 운전자의 대처 능력을 상실하게 만들어 막대한 인명 피해와 사회적 비용을 발생시키는 주요 원인입니다. 기존의 졸음 방지 방식은 운전자의 주관적 자각에 의존하거나 웨어러블 장비의 불편함을 동반했습니다.

본 프로젝트는 **다양한 주행 환경 변수(주야간 조명 변화, 선글라스 착용, 운전자의 다양한 얼굴형)** 속에서도 운전자의 의지에만 의존하는 위험한 상황을 해결하고, 연산 자원이 제한된 차량 내부 환경(Edge)에서도 원활하게 구동되는 경량화된 비전 시스템을 구축하고자 기획되었습니다.

## ✨ 주요 기능 (Key Features)
- **실시간 이진 분류 (Real-time Binary Classification):** 실시간 영상 프레임에서 추출된 얼굴 특징(눈·코·입)을 입력받아 즉각적인 졸음 상태 탐지 및 경고 출력.
- **경량화 모델링:** 단일 프레임 분석을 통해 계산량을 최소화하여 실시간 처리 속도(FPS) 최적화.
- **환경 변수 적응형 시스템:** 주간/야간 조명 및 선글라스 착용 등 주행 환경별 조건 구분 적용.

## 👨‍💻 팀원 역할 및 기여도 (Role & Contribution)

### 🟢 Model Optimization & Data Pipeline (본인)
- **데이터 증강(Augmentation) 파이프라인 구축:** - 회전, 크기 조정, 대칭 반전 기법 적용을 통한 데이터 다양성 확보.
  - 조명 조건이 달라지는 주야간 현실 도로 상황을 시뮬레이션하여 모델 적응력(Robustness) 향상.
- **실시간 처리용 CNN 아키텍처 설계:**
  - 단일 프레임 단위 얼굴 특징 분석에 맞춰 모델 구조를 단순화하고 연산 병목 현상 제거.
- **모델 학습 성과 시각화:**
  - Accuracy 및 Loss 변화 추이 비교 히트맵 및 그래프 시각화를 통해 극단적 데이터를 제어하고 예측 성능 개선.

### 👥 System Design & UI/UX (팀원)
- 졸음운전 한계점 분석 및 얼굴 표정 데이터(눈 깜빡임, 특징점) 추출 기준 정의.
- 운전 환경 변수 연구 (조명 변화, 얼굴형 조건 비교).
- 사용자 친화적 시스템 구동 및 경고 알림 프로세스 설계.
- 글로벌 안전 표준에 부합하는 서비스 확장 기획.

## 📈 AI 활용 사례 및 기술 변화 (Before & After)

| 구분 | 도입 전 (Manual) | 도입 후 (AI-Powered) |
| :--- | :--- | :--- |
| **탐지 방식** | 피로도 자각 및 동승자 관찰 등 주관적 판단 | CNN 모델 기반 데이터 중심의 객관적/선제적 탐지 |
| **환경 취약성** | 터널 진입, 심야 주행 시 시야 확보 및 판단 불가 | Data Augmentation을 통해 학습된 강건한 인식 성능 |
| **안전 보장** | 껌 씹기, 환기 등 운전자의 자발적 물리적 행동 필요 | 카메라를 통한 백그라운드 구동으로 비침습적 안전 보장 |

## 🚀 향후 고도화 로드맵 (Future Work)
[Image of MediaPipe face mesh landmarks]
1. **조명 변수 대응 강화:** 밝기 조절 및 색상 보정을 적용한 전처리 모델 추가 최적화.
2. **MediaPipe 랜드마크 추적 도입:** 3D 얼굴 회전(Yaw, Pitch) 시 발생하는 인식률 저하 한계를 극복하기 위해 눈 좌표(EAR) 정밀 추출 및 크롭 기능 적용 예정.
3. **YOLOv8 기반 복합 탐지:** 단순 얼굴 분류를 넘어, 전체 화각 내 객체 탐지를 통해 스마트폰 조작 등 산만 운전(Distracted Driving) 요소를 동시에 잡아내는 통합 예방 시스템으로 스케일업.

## ⚙️ 로컬 실행 방법 (Getting Started)
```bash
# 1. 저장소 클론
git clone [https://github.com/내아이디/Drowsiness-Detection-System.git](https://github.com/내아이디/Drowsiness-Detection-System.git)

# 2. 디렉토리 이동
cd Drowsiness-Detection-System

# 3. 필요 라이브러리 설치
pip install -r requirements.txt

# 4. 실시간 웹캠 테스트 코드 실행 (MediaPipe 랜드
