import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from src.prompting import build_prompt


def normalize_equation(eq):
    if not eq: return None
    eq = eq.split('=')[0].strip()
    return " ".join(eq.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-glob", type=str, default="outputs/teacher_chunk_*.parquet")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--small-n", type=int, default=2000,
                        help="Размер маленького датасета для быстрых итераций.")
    parser.add_argument("--student-id", type=str, default="google/gemma-3-1b-it",
                        help="Токенизатор студента — для подсчёта длин.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / args.out_dir
    out_dir.mkdir(exist_ok=True)

    # 1. Load all teacher chunks
    chunk_paths = sorted(Path(".").glob(args.chunks_glob))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks found at {args.chunks_glob}")
    print(f"Found {len(chunk_paths)} chunk files:")
    for p in chunk_paths:
        print(f"  {p}")

    dfs = [pd.read_parquet(p) for p in chunk_paths]
    df = pd.concat(dfs, ignore_index=True)
    n_raw = len(df)
    print(f"\nLoaded {n_raw} total teacher responses")
    print(f"Columns: {list(df.columns)}")

    # 2. Filter correct
    df = df[df["correct"] == True].copy()
    df = df[df["equation"].notna()].copy()
    n_correct = len(df)
    print(f"After keeping correct+extracted: {n_correct} ({n_correct/n_raw:.1%})")

    # 3. Normalize equations, show samples
    df["equation_norm"] = df["equation"].apply(normalize_equation)
    print("\n--- Sample equations (raw -> normalized) ---")
    for _, row in df.head(10).iterrows():
        print(f"  {row['equation']!r:45s} -> {row['equation_norm']!r}")

    # Dedup on
    df["_dedup_key"] = df.apply(
        lambda r: (tuple(sorted(r["nums"])), r["target"], r["equation_norm"]),
        axis=1,
    )
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["_dedup_key"]).drop(columns=["_dedup_key"])
    n_dedup = len(df)
    print(f"\nAfter dedup on (sorted_nums, target, equation): {n_dedup} "
          f"(removed {before_dedup - n_dedup})")

    # 4. Build SFT fields
    def build_row(r):
        messages = build_prompt({"nums": list(r["nums"]), "target": r["target"]})
        completion = f"<answer> {r['equation_norm']} </answer>"
        return pd.Series({
            "messages": messages,
            "completion": completion,
        })

    sft = df.apply(build_row, axis=1)
    sft["nums"] = df["nums"].values
    sft["target"] = df["target"].values
    sft["train_idx"] = df["train_idx"].values if "train_idx" in df.columns else range(len(df))

    # 5. Length stats (student tokenizer)
    print(f"\nLoading student tokenizer: {args.student_id}")
    tok = AutoTokenizer.from_pretrained(args.student_id, trust_remote_code=True)

    # Prompt рендерим через chat template студента — это то, что реально увидит SFTTrainer
    def render_prompt(messages):
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    sft["prompt_text"] = sft["messages"].apply(render_prompt)
    sft["prompt_tokens"] = sft["prompt_text"].apply(lambda t: len(tok.encode(t)))
    sft["completion_tokens"] = sft["completion"].apply(lambda t: len(tok.encode(t)))
    sft["total_tokens"] = sft["prompt_tokens"] + sft["completion_tokens"]

    def stats(s):
        return {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": int(s.min()), "p50": int(s.quantile(0.5)),
            "p95": int(s.quantile(0.95)), "max": int(s.max()),
        }

    len_stats = {
        "prompt_tokens": stats(sft["prompt_tokens"]),
        "completion_tokens": stats(sft["completion_tokens"]),
        "total_tokens": stats(sft["total_tokens"]),
    }
    print("\n--- Length stats (student tokenizer) ---")
    print(json.dumps(len_stats, indent=2))

    # 6. Save
    save_cols = ["messages", "completion", "nums", "target", "train_idx",
                 "prompt_tokens", "completion_tokens"]
    full_path = out_dir / "sft_train.parquet"
    sft[save_cols].to_parquet(full_path, index=False)
    print(f"\nSaved full dataset:  {full_path} ({len(sft)} rows)")

    small_path = out_dir / "sft_train_small.parquet"
    small = sft[save_cols].sample(
        n=min(args.small_n, len(sft)), random_state=args.seed
    ).reset_index(drop=True)
    small.to_parquet(small_path, index=False)
    print(f"Saved small dataset: {small_path} ({len(small)} rows)")

    summary = {
        "n_raw": int(n_raw),
        "n_correct": int(n_correct),
        "n_after_dedup": int(n_dedup),
        "length_stats": len_stats,
        "chunks_used": [str(p) for p in chunk_paths],
        "student_tokenizer": args.student_id,
        "example_completion": sft["completion"].iloc[0],
    }
    summary_path = out_dir / "sft_train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary:       {summary_path}")


if __name__ == "__main__":
    main()