from datasets import load_dataset, Dataset


def load_countdown_dataset():
    return load_dataset("HuggingFaceTB/Countdown-Task-GOLD", "all", split="train")


def make_splits(ds, dev_size=200, seed=42):
    shuffled = ds.shuffle(seed=seed)
    train = shuffled.select(range(dev_size, len(shuffled)))
    dev = shuffled.select(range(dev_size))
    return train, dev


def load_splits(path="../data/splits"):
    train = Dataset.from_parquet(f"{path}/train.parquet")
    dev = Dataset.from_parquet(f"{path}/dev.parquet")
    return train, dev


if __name__ == "__main__":
    # ds = load_countdown_dataset()
    # print(ds.to_pandas())
    # print(make_splits(ds))

    print(load_splits())