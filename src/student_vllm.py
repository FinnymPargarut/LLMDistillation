import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.validate import extract_equation_from_llm_response


def build_student_llm(model_path: str,
                      tensor_parallel_size: int = 1,
                      gpu_memory_utilization: float = 0.90,
                      max_model_len: int = 512,
                      dtype: str = "auto") -> LLM:
    """
    Загружает merged-модель студента в vLLM.

    model_path: путь к локальной merged-модели (например outputs/exp03/merged)
                или HF hub id. Если в пути есть adapter_config.json — это LoRA-
                адаптеры, и надо грузить иначе: этот случай тут не обрабатывается,
                merged-only.
    """
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=True,
    )


def generate_student_responses(llm: LLM, tokenizer, dataset, prompt_builder,
                               max_tokens: int = 128,
                               seed: int = 42) -> list[dict]:
    """
    Прогоняет студента на тестовом датасете. Возвращает list[dict] со всем нужным
    для сабмита + дебага: id (если есть), nums, target, response, equation.
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # greedy
        seed=seed,
    )

    messages_list = [prompt_builder(ex) for ex in dataset]
    prompts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        for msgs in messages_list
    ]

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"[vLLM student] {len(outputs)} responses in {elapsed:.1f}s "
          f"({total_tokens} tokens, {total_tokens/max(elapsed,1e-6):.0f} tok/s)")

    results = []
    for idx, (ex, out) in enumerate(zip(dataset, outputs)):
        response_text = out.outputs[0].text
        equation = extract_equation_from_llm_response(response_text)
        rec = {
            "idx": idx,
            "nums": list(ex["nums"]),
            "target": ex["target"],
            "response": response_text,
            "response_n_tokens": len(out.outputs[0].token_ids),
            "equation": equation,
        }
        if "id" in ex:
            rec["id"] = ex["id"]
        results.append(rec)

    return results


def load_student_tokenizer(model_path: str):
    """Токенизатор из той же папки что и модель — важно при merge."""
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)