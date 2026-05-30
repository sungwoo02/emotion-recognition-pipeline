# 1. 코랩 필수 라이브러리 설치
!pip install transformers datasets evaluate accelerate scikit-learn pandas numpy

import pandas as pd
import numpy as np
import torch
import time
import gc
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 💡 1. JSON 데이터 파싱 및 전처리 (감성대화말뭉치 맞춤형)
# ---------------------------------------------------------
print("⏳ JSON 데이터 파싱을 시작합니다. (시간이 조금 걸릴 수 있습니다)...")

def load_aihub_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for item in data:
        try:
            # 사람의 대화(HS01, HS02, HS03)를 하나로 이어붙여서 입력 텍스트로 만듭니다.
            content = item['talk']['content']
            text = " ".join([content.get(f'HS0{i}', '') for i in range(1, 4) if content.get(f'HS0{i}', '')])
            
            # 감정 코드(type)를 정답 라벨로 추출합니다 (예: E54, E31 등)
            label = item['profile']['emotion']['type']
            
            if text.strip() and label:
                records.append({'text': text.strip(), 'raw_label': label})
        except KeyError:
            continue
            
    return pd.DataFrame(records)

# 업로드하신 파일명에 맞게 로드합니다.
train_df = load_aihub_json("감성대화말뭉치(최종데이터)_Training.json")
val_df = load_aihub_json("감성대화말뭉치(최종데이터)_Validation.json")

print(f"✅ 데이터 로드 완료! (학습용: {len(train_df)}개, 검증용: {len(val_df)}개)")

# 💡 감정 코드(알파벳+숫자)를 인공지능이 이해할 수 있는 숫자(0, 1, 2...)로 변환합니다.
le = LabelEncoder()
# Train과 Validation에 있는 모든 라벨을 합쳐서 기준을 만듭니다.
le.fit(pd.concat([train_df['raw_label'], val_df['raw_label']]))

train_df['label'] = le.transform(train_df['raw_label'])
val_df['label'] = le.transform(val_df['raw_label'])
num_labels = len(le.classes_)

print(f"🎯 분류해야 할 감정 클래스 개수: {num_labels}개")

# HuggingFace Dataset으로 변환
train_dataset = Dataset.from_dict({'text': train_df['text'].tolist(), 'label': train_df['label'].tolist()})
# 이번 벤치마크에서는 Validation 세트를 평가 및 추론 시간 측정용(Test)으로 겸해서 사용합니다.
val_dataset = Dataset.from_dict({'text': val_df['text'].tolist(), 'label': val_df['label'].tolist()})

# 평가 기준 설정
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1_macro": f1["f1"]}

# ---------------------------------------------------------
# 🤖 2. 벤치마크할 5가지 모델 리스트
# ---------------------------------------------------------
models_to_test = {
    "KoELECTRA-Small": "monologg/koelectra-small-v3-discriminator",
    "ModernBERT-Base": "answerdotai/ModernBERT-base",
    "KLUE-RoBERTa-Base": "klue/roberta-base",
    "KR-ELECTRA-Base": "snunlp/KR-ELECTRA-discriminator",
    "KLUE-RoBERTa-Large": "klue/roberta-large"
}

results = []
print("\n🔥 본격적인 5대 모델 벤치마크를 시작합니다! 🔥\n")

# ---------------------------------------------------------
# 🚀 3. 모델 순회 학습 및 추론 시작
# ---------------------------------------------------------
for model_alias, model_path in models_to_test.items():
    print("="*65)
    print(f"🚀 현재 모델 : {model_alias}")
    print("="*65)
    
    # [코랩 메모리 최적화]
    if "Large" in model_alias:
        batch_size = 4  
        grad_accum = 8
    elif "Base" in model_alias:
        batch_size = 16 
        grad_accum = 2
    else:
        batch_size = 32 
        grad_accum = 1
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./temp_{model_alias}",
        num_train_epochs=3, # 데이터가 방대하므로 빠른 벤치마크를 위해 3 에폭으로 설정
        per_device_train_batch_size=batch_size, 
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        fp16=True, 
        learning_rate=3e-5,
        eval_strategy="epoch", 
        save_strategy="epoch",
        weight_decay=0.01, 
        load_best_model_at_end=True, 
        metric_for_best_model="eval_f1_macro", 
        logging_dir=f'./logs_{model_alias}',
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] 
    )

    # ⏱️ [1] 학습 시작
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start

    # [2] 최고 성능 측정 (Validation)
    print(f"[{model_alias}] 검증 데이터 성능 및 추론 시간 측정 중...")
    
    infer_start = time.time()
    eval_results = trainer.evaluate() # 추론(Inference) 과정이 포함되어 있습니다.
    infer_time = time.time() - infer_start
    
    val_acc = eval_results.get("eval_accuracy", 0.0) * 100
    val_f1 = eval_results.get("eval_f1_macro", 0.0) * 100

    # 결과 표에 기록 추가
    results.append({
        "Model": model_alias,
        "Accuracy (%)": round(val_acc, 2),
        "F1-Score (%)": round(val_f1, 2),
        "Train Time (s)": round(train_time, 2),
        "Inference Time (s)": round(infer_time, 4)
    })

    # 🧹 [가장 중요] 다음 모델을 위한 VRAM 청소
    del model
    del trainer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"✅ {model_alias} 메모리 초기화 완료.\n")

# ---------------------------------------------------------
# 📊 4. 최종 벤치마크 결과 표 출력
# ---------------------------------------------------------
print("\n" + "★"*70)
print("🏆 5대 모델 벤치마크 최종 결과 🏆")
print("★"*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score (%)", ascending=False).reset_index(drop=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(results_df.to_string(index=False))
print("★"*70)

# 결과 요약표 CSV 저장
results_df.to_csv("json_benchmark_results_summary.csv", index=False)
print("\n💾 결과 요약표가 'json_benchmark_results_summary.csv' 파일로 저장되었습니다.")
