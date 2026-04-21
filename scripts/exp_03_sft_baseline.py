import argparse
import json
import os
import time
from pathlib import Path

from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer

from src.data import load_splits
from src.sft_data import load_sft_dataset
from src.sft_eval import CountdownAccuracyCallback, evaluate_accuracy_on_dev
from src.sft_model import build_student_with_lora


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="unsloth/gemma-3-1b-it")
    p.add_argument("--data", default="outputs/sft_train.parquet")
    p.add_argument("--run-name", default="exp03_sft_baseline")
    p.add_argument("--out-dir", default="outputs/exp03")

    p.add_argument("--max-seq-length", type=int, default=256,
                   help="p95 у нас ~176, берём 256 с запасом.")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    # Training
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)

    # Eval (dev accuracy через генерацию)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--eval-n-quick", type=int, default=50,
                   help="Размер быстрой dev-выборки для промежуточных eval'ов.")
    p.add_argument("--eval-n-final", type=int, default=200,
                   help="Размер полной dev-выборки для финального eval (<=200).")
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=2)

    # Logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="countdown-distill")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def load_dev_examples(n: int):
    _, dev = load_splits()
    dev = dev.select(range(min(n, len(dev))))
    return [dict(ex) for ex in dev]


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    adapters_dir = out_dir / "adapters"
    merged_dir = out_dir / "merged"

    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ.setdefault("WANDB_LOG_MODEL", "false")

    print(f"Loading model + LoRA: {args.model_id}")
    model, tokenizer = build_student_with_lora(
        model_id=args.model_id,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )
    model.print_trainable_parameters()


    train_ds = load_sft_dataset(args.data, tokenizer)
    print(f"Train dataset: {len(train_ds)} examples")
    print(f"Sample prompt (last 200 chars):\n  {train_ds[0]['prompt'][-200:]!r}")
    print(f"Sample completion: {train_ds[0]['completion']!r}")

    dev_examples = load_dev_examples(args.eval_n_final)
    print(f"Dev examples: {len(dev_examples)} (quick slice: {args.eval_n_quick})")

    sft_config = SFTConfig(
        output_dir=str(ckpt_dir),
        run_name=args.run_name,

        max_length=args.max_seq_length,
        packing=False,
        dataset_num_proc=2,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_8bit",
        max_grad_norm=1.0,

        fp16=True,
        bf16=False,

        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",

        seed=args.seed,
        data_seed=args.seed,
        report_to=("wandb" if args.wandb else "none"),
    )

    callback = CountdownAccuracyCallback(
        dev_examples=dev_examples,
        tokenizer=tokenizer,
        eval_steps=args.eval_steps,
        eval_n_quick=args.eval_n_quick,
        gen_batch_size=args.gen_batch_size,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[callback],
    )

    print("\n=== Starting training ===")
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    print(f"\nTraining done in {train_time/60:.1f} min")

    print(f"\n=== Final dev eval (n={len(dev_examples)}) ===")
    final_metrics, final_samples = evaluate_accuracy_on_dev(
        model, tokenizer, dev_examples,
        gen_batch_size=args.gen_batch_size,
        return_samples_n=5,
    )
    print(json.dumps(final_metrics, indent=2))
    for s in final_samples:
        print(f"  nums={s['nums']} target={s['target']} -> eq={s['equation']!r}")

    model.save_pretrained(str(adapters_dir))
    tokenizer.save_pretrained(str(adapters_dir))
    print(f"Saved LoRA adapters: {adapters_dir}")

    try:
        model.save_pretrained_merged(
            str(merged_dir), tokenizer, save_method="merged_16bit",
        )
        print(f"Saved merged model: {merged_dir}")
    except Exception as e:
        print(f"Merge failed (non-fatal, adapters saved separately): {e}")

    summary = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "train_examples": len(train_ds),
        "dev_examples": len(dev_examples),
        "final_metrics": final_metrics,
        "train_time_minutes": round(train_time / 60, 2),
        "args": vars(args),
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary: {out_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()