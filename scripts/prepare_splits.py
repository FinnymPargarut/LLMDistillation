from dotenv import load_dotenv
from pathlib import Path
from src.data import load_countdown_dataset, make_splits


def main():
    load_dotenv()
    DEV_SIZE = 200
    SEED = 42
    OUT_PATH = "data/splits"

    ds = load_countdown_dataset()
    train, dev = make_splits(ds, DEV_SIZE, SEED)
    out = Path(__file__).parent.parent / OUT_PATH
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet")
    dev.to_parquet(out / "dev.parquet")
    # dev.to_csv(out / "dev.csv")
    print(f"Save split to {out}")


if __name__ == "__main__":
    main()