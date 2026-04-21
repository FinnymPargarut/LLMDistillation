import argparse
import json
import time
from pathlib import Path

import kagglehub
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from transformers import set_seed

from src.prompting import build_prompt
from src.student_vllm import (
    build_student_llm,
    generate_student_responses,
    load_student_tokenizer,
)
from src.submission import build_submission_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True,
                   help="Путь к merged-модели (например outputs/exp03/merged) "
                        "или HF hub id.")
    p.add_argument("--exp-name", required=True,
                   help="Имя для файлов сабмита (без .csv).")

    # vLLM
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--dtype", type=str, default="auto",
                   help="bfloat16 на A100, float16 на T4, auto обычно ок.")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Студент генерит короткий ответ — 128 с запасом.")
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--gpu-mem-util", type=float, default=0.90)

    # Data
    p.add_argument("--competition", default="distillation-challenge-2026")
    p.add_argument("--test-file", default="test_public.csv",
                   help="Имя файла с тестом внутри данных соревнования.")
    p.add_argument("--sample-submission", default="sample_submission.csv")
    p.add_argument("--examples-limit", type=int, default=None,
                   help="Для отладки: взять только первые N примеров.")

    p.add_argument("--out-dir", default="outputs/submissions")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    load_dotenv()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = kagglehub.competition_download(args.competition)
    print(f"Competition files: {path}")

    test = Dataset.from_csv(f"{path}/{args.test_file}")
    if args.examples_limit is not None:
        test = test.select(range(args.examples_limit))
    print(f"Test examples: {len(test)} (columns: {list(test.column_names)})")

    sample_submission_path = f"{path}/{args.sample_submission}"

    print(f"Loading student: {args.model_path}")
    tokenizer = load_student_tokenizer(args.model_path)
    llm = build_student_llm(
        model_path=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    t0 = time.time()
    results = generate_student_responses(
        llm=llm,
        tokenizer=tokenizer,
        dataset=test,
        prompt_builder=build_prompt,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    inference_time = time.time() - t0
    print(f"Inference time: {inference_time:.1f}s")

    submission_df, stats = build_submission_df(results, sample_submission_path)
    stats["model_path"] = args.model_path
    stats["inference_time_s"] = round(inference_time, 1)

    print("\n=== Submission stats ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if stats["n_fallback"]:
        print(f"\nWARNING: {stats['n_fallback']} задач получили fallback equation "
              f"('{'0'}'). Это потери на LB — модель не сгенерила валидный ответ.")

    # Сохранение
    submission_path = out_dir / f"{args.exp_name}_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSaved submission: {submission_path}")

    # Сырые ответы для дебага
    raw_df = pd.DataFrame(results)
    raw_path = out_dir / f"{args.exp_name}_raw.parquet"
    raw_df.to_parquet(raw_path, index=False)
    print(f"Saved raw responses: {raw_path}")

    stats_path = out_dir / f"{args.exp_name}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved stats: {stats_path}")


if __name__ == "__main__":
    main()