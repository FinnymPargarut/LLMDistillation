from unsloth import FastLanguageModel


DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_student_with_lora(
    model_id: str = "unsloth/gemma-3-1b-it",
    max_seq_length: int = 256,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    load_in_4bit: bool = False,
    target_modules: list[str] | None = None,
    seed: int = 42,
):
    """
    Загружает студента через Unsloth и вешает LoRA-адаптеры.

    По железу:
    На T4 нет bf16. Если dtype=None — Unsloth сам выберет fp16 и применит свои касты для gemma-3.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or DEFAULT_LORA_TARGETS,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    return model, tokenizer