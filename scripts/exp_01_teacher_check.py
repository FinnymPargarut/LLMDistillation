import argparse
import time
from dotenv import load_dotenv
from transformers import set_seed

from src.data import load_splits
from src.prompting import build_prompt
from src.inference import load_model_and_tokenizer, run_inference_on_dataset, evaluate_model


QWEN3_THINKING_GEN_KWARGS = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--exp-name", type=str, default="exp01_teacher_check")
    parser.add_argument("--max-new-tokens", type=int, default=1536,
                        help="Qwen3 thinking любит длинные ответы. 1536 — компромисс.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--examples-limit", type=int, default=None,
                        help="Ограничить dev для быстрой проверки (например 20-50).")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="NF4 квантизация через bitsandbytes (для T4).")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Отключить thinking-режим Qwen3 (быстрее, но хуже качество).")
    args = parser.parse_args()

    load_dotenv()
    set_seed(42)

    print(f"Loading model {args.model_id} (4bit={args.load_in_4bit})...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model_id, load_in_4bit=args.load_in_4bit)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    _, dev = load_splits()
    if args.examples_limit is not None:
        dev = dev.select(range(min(args.examples_limit, len(dev))))
    print(f"Running on {len(dev)} examples from dev split")

    # Qwen3-специфика
    thinking_on = not args.no_thinking
    chat_template_kwargs = {"enable_thinking": thinking_on}
    gen_kwargs = dict(QWEN3_THINKING_GEN_KWARGS) if thinking_on else {
        "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    }
    print(f"thinking={thinking_on}, gen_kwargs={gen_kwargs}")
    print(f"chat_template_kwargs={chat_template_kwargs}")

    t0 = time.time()
    generator = run_inference_on_dataset(
        model, tokenizer, dev, build_prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        generation_kwargs=gen_kwargs,
        chat_template_kwargs=chat_template_kwargs,
    )
    summary, _ = evaluate_model(generator, len(dev), args.exp_name, print_first_n=3)
    elapsed = time.time() - t0

    per_example_s = elapsed / len(dev)
    print(f"\n--- Throughput ---")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Per example: {per_example_s:.2f}s")
    print(f"Projected for 10k examples: {per_example_s * 10000 / 3600:.1f}h")
    print(f"Projected for 20k examples: {per_example_s * 20000 / 3600:.1f}h")
    print(f"\n--- Summary ---")
    print(summary)


if __name__ == "__main__":
    main()