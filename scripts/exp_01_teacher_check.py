import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from src.data import load_splits
from src.prompting import build_prompt
from src.teacher_vllm import (
    build_llm,
    generate_teacher_responses,
    summarize_results,
    save_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B-AWQ",
                        help="Qwen/Qwen3-8B-AWQ для T4, Qwen/Qwen3-8B для A100.")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--dtype", type=str, default="float16",
                        help="float16 для T4, bfloat16 для A100.")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Qwen3 thinking требует длинных ответов.")
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--examples-limit", type=int, default=None,
                        help="Ограничить dev для быстрого теста.")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--exp-name", type=str, default="exp01_teacher_check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()

    _, dev = load_splits()
    if args.examples_limit is not None:
        dev = dev.select(range(min(args.examples_limit, len(dev))))
    print(f"Dev: {len(dev)} examples")
    print(f"Model: {args.model_id} (tp={args.tp}, dtype={args.dtype})")

    # Токенайзер нужен отдельно для apply_chat_template с enable_thinking
    from transformers import AutoTokenizer
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    llm = build_llm(
        model_id=args.model_id,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )
    print(f"Loaded LLM+tokenizer in {time.time() - t0:.1f}s")

    t0 = time.time()
    results = generate_teacher_responses(
        llm=llm,
        tokenizer=tokenizer,
        dataset=dev,
        prompt_builder=build_prompt,
        enable_thinking=not args.no_thinking,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    summary = summarize_results(results)

    print("\n=== First 3 responses (eyeball) ===")
    for r in results[:3]:
        print(f"\n--- idx={r['idx']} nums={r['nums']} target={r['target']}")
        print(f"equation={r['equation']}, correct={r['correct']} ({r['reason']})")
        resp = r["response"]
        if "</think>" in resp:
            tail = resp.split("</think>", 1)[-1].strip()
            print(f"[post-think] {tail[:400]}")
        else:
            print(f"[raw, no </think>] {resp[:400]}...")

    per_example_s = elapsed / len(dev)
    print(f"\n--- Throughput ---")
    print(f"Total: {elapsed:.1f}s, per example: {per_example_s:.2f}s")
    for n in (1000, 5000, 10000, 20000):
        print(f"  {n:>6} examples → {per_example_s * n / 3600:.1f}h")

    print(f"\n--- Summary ---")
    print(summary)

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    save_results(results, summary, out_dir / f"{args.exp_name}.json")


if __name__ == "__main__":
    main()