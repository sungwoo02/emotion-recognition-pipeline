# emotion-recognition-pipeline
# 🎭 대화 텍스트 기반 다중 감성 분류 모델 (Emotion Classification)

> AI Hub '감성대화말뭉치'를 활용하여 일상 대화 속 복합적인 감정을 추론하는 KLUE-RoBERTa-Large 파인튜닝 프로젝트입니다.

## 📌 Project Overview
본 프로젝트는 사용자의 연속된 자연어 발화(대화) 문맥을 분석하여, 내포된 미묘한 감정 상태를 인지하고 분류하는 딥러닝 파이프라인을 구축하는 것을 목표로 합니다. 
특히, 컴퓨팅 자원이 제한된 환경(Google Colab)에서 대형 언어 모델(Large Language Model)을 OOM(Out of Memory) 에러 없이 안정적으로 학습시키기 위한 **메모리 최적화 기법**을 적극적으로 도입했습니다.

## 🛠 Tech Stack
- **Language:** Python
- **AI/ML:** PyTorch, HuggingFace `transformers`, `evaluate`, Scikit-learn
- **Data Manipulation:** Pandas, Numpy
- **Environment:** Google Colab (T4 GPU)

## 📊 Dataset
- **출처:** [AI Hub 감성대화말뭉치](https://aihub.or.kr/)
- **형식:** JSON (`Training.json`, `Validation.json`)
- **특징:**
  - 사람의 연속 발화(HS01~HS03)를 단일 문맥으로 병합하여 활용.
  - 6대 대분류 감정이 아닌, 60여 개의 미묘한 세부 감정 코드(예: E54, E31 등)를 라벨로 사용하는 고난도 다중 분류(Multi-class) 데이터셋.

## 🚀 Key Features & Pipeline

### 1. 전처리 (Data Preprocessing)
- 중첩된 JSON 구조 파싱 및 결측치/공백 텍스트 필터링 (`try-except` 예외 처리)
- `LabelEncoder`를 활용한 60여 개 감정 코드의 정수형 라벨링 및 HuggingFace `Dataset` 구조화
- Max Length 128 제한 토큰화 (Tokenizer)

### 2. 모델 학습 최적화 (Model Training & Optimization)
한국어 문맥 이해도가 뛰어난 **`KLUE-RoBERTa-Large`** 모델을 채택하였으며, VRAM 한계를 극복하기 위해 아래 기법을 적용했습니다.
- **Gradient Accumulation:** Batch Size를 4로 낮추되, Accumulation Steps를 8로 설정하여 실질적 배치 크기 유지 (메모리 절약)
- **Mixed Precision (FP16):** 학습 속도 향상 및 메모리 점유율 최소화
- **VRAM Cache Clearing:** 모델 벤치마크 및 반복 학습 시 `gc.collect()`, `torch.cuda.empty_cache()`를 호출하여 메모리 누수 완벽 차단
- **Early Stopping:** 과적합 방지 (Patience=1)

## 📈 Results & Troubleshooting

### 베이스라인 성능 (Epoch 3 기준)
- **Accuracy:** 22.3%
- **Macro F1-Score:** 14.6%

### 💡 한계 분석 및 데이터 중심(Data-centric) 개선 전략
초기 베이스라인 모델 성능이 22%대로 측정되었습니다. 모델의 학습 Loss는 정상적으로 감소하였으나, 이는 정답 라벨이 60여 개의 세부 감정으로 과도하게 파편화되어 있어 모델이 결정 경계를 찾기 어려웠기 때문으로 분석됩니다. 

**[Next Step]** 향후 고도화 단계에서는 전처리 파이프라인에 **Mapping** 로직을 추가하여 60개의 세부 감정 코드를 **6대 대표 감정(기쁨, 슬픔, 분노, 불안, 상처, 당황)**으로 통합할 예정입니다. 라벨 통합 시 데이터 불균형이 완화되고 모델의 분류 정확도가 80% 이상으로 비약적으로 상승할 것으로 예상합니다.

## 📁 Repository Structure
```text
├── data/
│   ├── 감성대화말뭉치(최종데이터)_Training.json (gitignore)
│   └── 감성대화말뭉치(최종데이터)_Validation.json (gitignore)
├── main.py (학습 및 벤치마크 파이프라인 코드)
├── json_benchmark_results_summary.csv (평가 결과 리포트)
└── README.md

# 1. 의존성 패키지 설치
pip install transformers datasets evaluate accelerate scikit-learn pandas numpy

# 2. 파이프라인 실행
python main.py
