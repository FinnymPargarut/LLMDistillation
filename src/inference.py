"""
Инференс на задаче Countdown.
Запуск: python src/inference.py --config configs/inference.yaml

Baseline (дефолтная Gemma без обучения):
    python src/inference.py --config configs/inference.yaml

После обучения (передаём путь к весам):
    python src/inference.py --config configs/inference.yaml --model_path outputs/checkpoints/run1
"""

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import wandb
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
import os

from evaluate import parse_equation, check_equation


load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_prompt(tokenizer, nums: list[int], target: int) -> str:
    messages = [{
        "role": "user",
        "content": (
            f"Using the numbers {nums}, create an equation that equals {target}. "
            f"You can use basic arithmetic operations (+, -, *, /) and parentheses. "
            f"Each number must be used exactly once. "
            f"Think step by step, then write the final equation on the last line."
        )
    }]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def run_inference(config: dict, model_path: str | None = None):
    model_id = model_path or config["model_id"]

    if config.get("use_wandb"):
        wandb.init(
            project=config["wandb_project"],
            name=config["wandb_run_name"],
            config={**config, "model_path": model_id},
        )

    # -- Модель
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # -- Данные
    dataset = load_dataset(
        config["dataset_id"],
        config["dataset_config"],
        split=config["split"]
    )
    if config.get("num_samples"):
        dataset = dataset.select(range(config["num_samples"]))
    print(f"Evaluating on {len(dataset)} samples")
    if args.dry_run:
        dataset = dataset.select(range(1))
        config["max_new_tokens"] = 10
        config["use_wandb"] = False
        print("=== DRY RUN MODE ===")

    # -- Инференс
    results = []
    correct = 0

    for i, example in enumerate(tqdm(dataset)):
        nums = example["nums"]
        target = example["target"]

        prompt = build_prompt(tokenizer, nums, target)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        equation = parse_equation(generated)
        check = check_equation(equation, nums, target)
        if check["is_correct"]:
            correct += 1

        results.append({
            "target": target,
            "nums": nums,
            "raw_output": generated,
            "parsed_equation": equation,
            **check,
        })

        if (i + 1) % 10 == 0:
            acc = correct / (i + 1)
            print(f"[{i+1}/{len(dataset)}] Accuracy: {acc:.3f}")
            if config.get("use_wandb"):
                wandb.log({"running_accuracy": acc, "step": i + 1})

    # -- Итоги
    accuracy = correct / len(dataset)
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{len(dataset)})")

    print("\n=== Sample outputs ===")
    for r in results[:5]:
        status = "✓" if r["is_correct"] else "✗"
        print(f"{status} target={r['target']} nums={r['nums']}")
        print(f"  equation: {r['parsed_equation']}")
        print(f"  raw: {r['raw_output'][:150].strip()}\n")

    Path("outputs").mkdir(exist_ok=True)
    out_path = f"outputs/results_{Path(model_id).name}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    if config.get("use_wandb"):
        wandb.log({"final_accuracy": accuracy})
        wandb.finish()

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Путь к обученным весам. Если не указан — берётся model_id из конфига")
    parser.add_argument("--dry_run", action="store_true",
                        help="Загружает модель и прогоняет 1 пример для проверки кода")
    args = parser.parse_args()

    config = load_config(args.config)
    run_inference(config, model_path=args.model_path)