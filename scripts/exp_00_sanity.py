from dotenv import load_dotenv
import argparse
from transformers import set_seed

from src.data import load_splits
from src.prompting import build_prompt
from src.inference import load_model_and_tokenizer, run_inference_on_dataset, evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--exp-name", type=str, default="gemma_sanity")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples-limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    set_seed(42)

    model, tokenizer = load_model_and_tokenizer(args.model_id)
    _, dev = load_splits()
    if args.examples_limit is not None:
        dev = dev.select(range(args.examples_limit))
    generator = run_inference_on_dataset(model, tokenizer, dev, build_prompt,
                                         max_new_tokens=args.max_new_tokens,
                                         batch_size=args.batch_size
                                         )
    summary, _ = evaluate_model(generator, len(dev), args.exp_name)
    print("Summary:\n", summary)


if __name__ == "__main__":
    main()
