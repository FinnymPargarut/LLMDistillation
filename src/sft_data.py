import pandas as pd
from datasets import Dataset


def load_sft_dataset(parquet_path: str, tokenizer) -> Dataset:
    df = pd.read_parquet(parquet_path)

    eos = tokenizer.eos_token or ""

    def render(row):
        messages = list(row["messages"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        completion = row["completion"]
        if eos and not completion.endswith(eos):
            completion = completion + eos
        return {"prompt": prompt, "completion": completion}

    records = [render(r) for _, r in df.iterrows()]
    return Dataset.from_list(records)