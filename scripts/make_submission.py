import pandas as pd
import argparse
import kagglehub
from transformers import set_seed
from datasets import Dataset
from dotenv import load_dotenv
from pathlib import Path

from src.prompting import build_prompt
from src.inference import load_model_and_tokenizer, run_inference_on_dataset, evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--exp-name", type=str, default="submit_results")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples-limit", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    set_seed(42)

    path = kagglehub.competition_download('distillation-challenge-2026')
    print("Path to competition files:", path)

    model, tokenizer = load_model_and_tokenizer(args.model_id)
    test = Dataset.from_csv(f"{path}/test_public.csv")
    if args.examples_limit is not None:
        test = test.select(range(args.examples_limit))
    generator = run_inference_on_dataset(model, tokenizer, test, build_prompt,
                                         greedy=False,
                                         max_new_tokens=args.max_new_tokens,
                                         batch_size=args.batch_size
                                         )
    summary, results = evaluate_model(generator, len(test), args.exp_name)
    print("Summary:\n", summary)

    equations = [d["equation"] for d in results]
    submission = pd.read_csv(f"{path}/sample_submission.csv")
    submission["equation"] = equations

    submission_path = Path(__file__).resolve().parent.parent / "outputs"
    submission.to_csv(submission_path / f"{args.exp_name}_submission.csv", index=False)


if __name__ == "__main__":
    main()