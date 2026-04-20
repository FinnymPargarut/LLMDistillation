import json
import time
from pathlib import Path
from vllm import LLM, SamplingParams

from src.validate import extract_equation_from_llm_response, validate_equation


QWEN3_THINKING_SAMPLING = dict(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
)

QWEN3_NOTHINKING_SAMPLING = dict(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
)


def build_llm(model_id: str, tensor_parallel_size: int = 1,
              gpu_memory_utilization: float = 0.90,
              max_model_len: int = 10240,
              dtype: str = "auto") -> LLM:
    return LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=dtype,
        trust_remote_code=True,
    )


def format_prompts(tokenizer, messages_list: list[list[dict]],
                   enable_thinking: bool = True) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for msgs in messages_list
    ]


def generate_teacher_responses(llm: LLM, tokenizer, dataset, prompt_builder,
                               enable_thinking: bool = True,
                               max_tokens: int = 8192,
                               sampling_override: dict | None = None,
                               seed: int = 42,
                               ) -> list[dict]:
    """
    Прогоняем vLLM по всему датасету разом.
    """
    base_sampling = QWEN3_THINKING_SAMPLING if enable_thinking else QWEN3_NOTHINKING_SAMPLING
    sampling_kwargs = dict(base_sampling)
    if sampling_override:
        sampling_kwargs.update(sampling_override)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        seed=seed,
        **sampling_kwargs,
    )

    messages_list = [prompt_builder(ex) for ex in dataset]
    prompts = format_prompts(tokenizer, messages_list, enable_thinking=enable_thinking)

    # vLLM сам делает continuous batching
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    total_out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"[vLLM] generated {len(outputs)} responses in {elapsed:.1f}s "
          f"({total_out_tokens} out_tokens, {total_out_tokens/elapsed:.0f} tok/s)")

    results = []
    for idx, (ex, out) in enumerate(zip(dataset, outputs)):
        response_text = out.outputs[0].text
        equation = extract_equation_from_llm_response(response_text)
        if equation is not None:
            val_res = validate_equation(equation, ex["nums"], ex["target"])
            correct = val_res.ok
            reason = val_res.reason
        else:
            correct = False
            reason = "Failed to extract equation from response"
        results.append({
            "idx": idx,
            "nums": list(ex["nums"]),
            "target": ex["target"],
            "response": response_text,
            "response_n_tokens": len(out.outputs[0].token_ids),
            "equation": equation,
            "correct": correct,
            "reason": reason,
        })

    return results


def summarize_results(results: list[dict]) -> dict:
    n = len(results)
    extracted = sum(1 for r in results if r["equation"] is not None)
    correct = sum(1 for r in results if r["correct"])
    avg_tokens = sum(r["response_n_tokens"] for r in results) / n if n else 0
    return {
        "n_examples": n,
        "equation_extracted": extracted,
        "equation_correct": correct,
        "extracted_rate": extracted / n if n else 0,
        "accuracy": correct / n if n else 0,
        "avg_response_tokens": avg_tokens,
    }


def save_results(results: list[dict], summary: dict, out_path: Path):
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_example": results}, f,
                  indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")