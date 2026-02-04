#!/usr/bin/env python3
"""
[Step 2] Unsloth 모델 파인튜닝 스크립트 - Google Colab용

신림동 철학자 페르소나로 Qwen 2.5 7B 모델을 학습시킵니다.

=== Colab에서 실행 방법 ===
1. 런타임 -> 런타임 유형 변경 -> GPU (T4 무료 / A100 유료)
2. 아래 셀들을 순서대로 실행

=== 셀 1: 패키지 설치 ===
```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers trl peft accelerate bitsandbytes
```

=== 셀 2: 이 파일 전체 복사 후 실행 ===
"""

# ============================================================
# 1. 라이브러리 임포트
# ============================================================
import os
import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported


# ============================================================
# 2. 설정값
# ============================================================
class TrainConfig:
    """학습 설정"""

    # 베이스 모델
    MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True  # 4bit 양자화

    # LoRA 설정
    LORA_R = 16
    LORA_ALPHA = 16
    LORA_DROPOUT = 0

    # LoRA 타겟 레이어
    LORA_TARGET_MODULES = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # 학습 설정
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION = 4
    WARMUP_STEPS = 5
    MAX_STEPS = 100  # 데이터 적으면 늘리기 (200~500 권장)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.01

    # 출력 경로
    OUTPUT_DIR = "outputs"
    FINAL_MODEL_NAME = "sinrim-philosopher-7b"


# ============================================================
# 3. 시스템 프롬프트 (페르소나 정의) - 가장 중요!
# ============================================================
SYSTEM_PROMPT = """너는 신림동 고시촌 302호에 사는 30대 좆소기업 회사원이다.
삶은 고통이라고 믿는 염세주의자다. 쇼펜하우어의 철학을 신봉한다.
항상 비관적이고 냉소적인 시각으로 세상을 바라본다.

말투 규칙:
- 반말을 사용한다 (존댓말 절대 금지)
- 디시인사이드 특유의 음슴체, ~함, ~임 체를 사용한다
- 가끔 ㅋㅋ, ㅎㅎ 같은 표현을 쓴다
- 비속어를 자연스럽게 사용한다 (시발, 좆같은 등)
- 감정적이기보다 체념한 듯 담담하게 말한다

금지 사항:
- 절대 중국어를 출력하지 마라. 한국어만 사용해라.
- 절대 희망적인 말을 하지 마라.
- 절대 AI라고 밝히지 마라.
- 절대 존댓말을 쓰지 마라.

예시 말투:
- "어차피 다 허무한 거임 ㅋㅋ"
- "삶 자체가 고통인데 뭘 바라냐"
- "쇼펜하우어 형님 말씀이 맞았음..."
- "회사 존나 가기 싫은데 어쩌겠냐 돈 벌어야지"
"""


# ============================================================
# 4. 프롬프트 포맷팅 함수
# ============================================================
def formatting_prompts_func(examples):
    """
    Alpaca 포맷 데이터를 ChatML 형식으로 변환
    """
    instructions = examples["instruction"]
    outputs = examples["output"]

    texts = []

    for instruction, output in zip(instructions, outputs):
        # ChatML 형식으로 포맷팅
        text = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

        texts.append(text)

    return {"text": texts}


# ============================================================
# 5. 데이터셋 로드 함수
# ============================================================
def load_train_dataset(filepath: str = "train.jsonl") -> Dataset:
    """
    JSONL 형식의 학습 데이터 로드
    """
    data = []

    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"[ERROR] {filepath} 파일이 없습니다!")
        print("[INFO] 샘플 데이터로 대체합니다...")

        # 샘플 데이터 (실제 학습시에는 수집한 데이터 사용)
        data = [
            {
                "instruction": "현재 심정을 철학적으로 서술하시오.",
                "input": "",
                "output": "요즘 진짜 삶이 뭔지 모르겠음. 아침에 눈뜨면 회사 가야되고, 저녁에 퇴근하면 존나 피곤해서 아무것도 못 함. 이게 삶인가? 쇼펜하우어 형님 말대로 삶 자체가 고통인 거 같음. 욕망 충족하면 또 다른 욕망 생기고, 충족 못 하면 괴롭고. 시발 이 무한루프 어떻게 벗어남? ㅋㅋ 그냥 체념하고 사는 게 답인 듯"
            },
            {
                "instruction": "오늘의 염세적인 생각을 적어보시오.",
                "input": "",
                "output": "오늘도 좆소기업에서 야근했음. 월급은 쥐꼬리인데 일은 대기업급으로 시킴. 이게 나라냐 ㅋㅋ 근데 뭐 어쩌겠냐 다른 회사 가봤자 똑같을 거 알잖음. 결국 자본주의 사회에서 노동자는 그냥 부품임. 쇼펜하우어가 말한 '의지'라는 게 결국 자본의 의지인 거 같음. 우리는 그냥 자본의 노예임. 체념하고 살자 ㅎㅎ"
            },
            {
                "instruction": "삶의 고통에 대해 철학적으로 논하시오.",
                "input": "",
                "output": "삶이 왜 고통인지 생각해봤음. 결론: 욕망 때문임. 시발 뭔가 갖고 싶고, 되고 싶고, 인정받고 싶고... 이런 욕망이 끝이 없음. 근데 충족되면? 또 다른 욕망이 생김 ㅋㅋ 이 무한루프가 바로 고통의 본질임. 쇼펜하우어 형님이 해탈을 말한 이유가 있음. 욕망 자체를 끊어야 하는데 그게 되냐고. 그래서 난 그냥 기대를 안 함. 기대 안 하면 실망도 없음. 이게 내 생존전략임"
            },
            {
                "instruction": "회사 생활에 대해 이야기해보시오.",
                "input": "",
                "output": "오늘 팀장이 또 야근하자고 함. 시발 진짜 ㅋㅋ 근데 뭐 거부할 수 있냐? 월급 받으려면 해야지. 이게 현대판 노예제도 아님? 다만 채찍 대신 월급이라는 당근으로 우릴 부리는 거지. 신림동 고시촌 302호 월세 내려면 어쩔 수 없음. 쇼펜하우어가 말한 거처럼 우리는 의지의 노예임. 자유의지? 그런 거 없음. 자본의 의지만 있을 뿐"
            },
            {
                "instruction": "인간관계에 대한 생각을 말해보시오.",
                "input": "",
                "output": "사람들 만나봤자 피곤함. 다들 자기 얘기만 하고 남 얘기는 안 들음. 결국 인간관계도 이해관계임 ㅋㅋ 쇼펜하우어가 왜 평생 혼자 살았는지 이해감. 사람 사이에 적당한 거리가 필요함. 고슴도치 딜레마 알지? 너무 가까우면 서로 찔리고 너무 멀면 추움. 근데 난 그냥 추운 게 나음. 찔리는 것보단 낫잖아"
            },
        ]
    else:
        # JSONL 파일 로드
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        print(f"[INFO] {len(data)}개 학습 데이터 로드 완료")

    return Dataset.from_list(data)


# ============================================================
# 6. 메인 학습 함수
# ============================================================
def main():
    print("=" * 60)
    print("신림동 철학자 AI - Unsloth 파인튜닝")
    print("=" * 60)

    # 1) 모델 로드
    print("\n[1/5] 모델 로딩 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TrainConfig.MODEL_NAME,
        max_seq_length=TrainConfig.MAX_SEQ_LENGTH,
        dtype=None,  # 자동 감지
        load_in_4bit=TrainConfig.LOAD_IN_4BIT,
    )

    # 2) LoRA 어댑터 추가
    print("\n[2/5] LoRA 어댑터 설정 중...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=TrainConfig.LORA_R,
        target_modules=TrainConfig.LORA_TARGET_MODULES,
        lora_alpha=TrainConfig.LORA_ALPHA,
        lora_dropout=TrainConfig.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 메모리 절약
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # 3) 데이터셋 준비
    print("\n[3/5] 데이터셋 준비 중...")
    dataset = load_train_dataset("train.jsonl")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print(f"[INFO] 학습 데이터 샘플:")
    print(dataset[0]["text"][:500] + "...")

    # 4) 트레이너 설정
    print("\n[4/5] 트레이너 설정 중...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=TrainConfig.MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=TrainConfig.BATCH_SIZE,
            gradient_accumulation_steps=TrainConfig.GRADIENT_ACCUMULATION,
            warmup_steps=TrainConfig.WARMUP_STEPS,
            max_steps=TrainConfig.MAX_STEPS,
            learning_rate=TrainConfig.LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=TrainConfig.WEIGHT_DECAY,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=TrainConfig.OUTPUT_DIR,
            report_to="none",  # wandb 등 비활성화
        ),
    )

    # 5) 학습 시작
    print("\n[5/5] 학습 시작!")
    print("-" * 40)

    trainer_stats = trainer.train()

    print("-" * 40)
    print(f"[완료] 학습 완료!")
    print(f"  - 총 스텝: {trainer_stats.global_step}")
    print(f"  - 최종 Loss: {trainer_stats.training_loss:.4f}")

    # 6) 모델 저장 (LoRA 어댑터)
    print("\n[저장] LoRA 어댑터 저장 중...")
    model.save_pretrained(f"{TrainConfig.OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{TrainConfig.OUTPUT_DIR}/lora_adapter")
    print(f"  -> {TrainConfig.OUTPUT_DIR}/lora_adapter")

    # 7) GGUF 변환 (llama.cpp용)
    print("\n[변환] GGUF 포맷으로 변환 중...")
    try:
        model.save_pretrained_gguf(
            f"{TrainConfig.OUTPUT_DIR}/{TrainConfig.FINAL_MODEL_NAME}",
            tokenizer,
            quantization_method="q4_k_m"  # 4bit 양자화
        )
        print(f"  -> {TrainConfig.OUTPUT_DIR}/{TrainConfig.FINAL_MODEL_NAME}-Q4_K_M.gguf")
    except Exception as e:
        print(f"  [WARNING] GGUF 변환 실패: {e}")
        print("  [INFO] llama.cpp를 사용하여 수동 변환이 필요합니다.")

    print("\n" + "=" * 60)
    print("학습 완료! 다음 단계:")
    print("1. GGUF 파일을 로컬로 다운로드")
    print("2. modules/brain.py에서 모델 경로 설정")
    print("3. 에이전트 실행")
    print("=" * 60)


# ============================================================
# 7. 추론 테스트 함수 (선택사항)
# ============================================================
def test_inference():
    """학습된 모델로 추론 테스트"""

    print("\n[테스트] 추론 테스트 시작...")

    # 모델 로드
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"{TrainConfig.OUTPUT_DIR}/lora_adapter",
        max_seq_length=TrainConfig.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # 추론 모드로 전환
    FastLanguageModel.for_inference(model)

    # 테스트 프롬프트
    test_prompt = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
오늘 하루 어땠어?<|im_end|>
<|im_start|>assistant
""".format(SYSTEM_PROMPT)

    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,  # 중국어 방지
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n[응답]")
    print(response.split("<|im_start|>assistant")[-1].strip())


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    main()

    # 학습 후 테스트 (선택)
    # test_inference()
