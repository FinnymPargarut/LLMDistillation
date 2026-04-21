import argparse
import json
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.data import load_splits
from src.prompting import build_prompt
from src.teacher_vllm import (
    build_llm,
    generate_teacher_responses,
    summarize_results,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16",
                        help="float16 для AWQ, bfloat16 для обычного Qwen3-8B на A100.")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)

    parser.add_argument("--start-idx", type=int, default=0,
                        help="С какого индекса начать в train-сплите.")
    parser.add_argument("--n-examples", type=int, required=True,
                        help="Сколько примеров обработать.")

    parser.add_argument("--output-name", type=str, default="teacher_data",
                        help="Имя файла вывода (без расширения).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv()

    train, _ = load_splits()
    end_idx = min(args.start_idx + args.n_examples, len(train))
    subset = train.select(range(args.start_idx, end_idx))
    print(f"Processing train[{args.start_idx}:{end_idx}] = {len(subset)} examples")
    print(f"Model: {args.model_id} (tp={args.tp}, dtype={args.dtype})")

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
        dataset=subset,
        prompt_builder=build_prompt,
        enable_thinking=True,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    # Добавим глобальный train-идекс (чтобы потом можно было matched обратно)
    for i, r in enumerate(results):
        r["train_idx"] = args.start_idx + i

    summary = summarize_results(results)
    print(f"\n=== Summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nTotal generation time: {elapsed:.1f}s ({elapsed/len(subset):.2f}s/example)")

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(results)
    parquet_path = out_dir / f"{args.output_name}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved parquet: {parquet_path} ({len(df)} rows)")

    summary["start_idx"] = args.start_idx
    summary["end_idx"] = end_idx
    summary["elapsed_s"] = elapsed
    summary["model_id"] = args.model_id
    summary_path = out_dir / f"{args.output_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()