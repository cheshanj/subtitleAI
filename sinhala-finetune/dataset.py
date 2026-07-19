"""Dataset helpers for NLLB-200 Sinhala fine-tuning."""
import pandas as pd
from datasets import Dataset, DatasetDict

LANG_CODES = {
    "english": "eng_Latn",
    "tamil": "tam_Taml",
    "hindi": "hin_Deva",
    "sinhala": "sin_Sinh",
}


def load_pairs(filepath, src_lang="english"):
    df = pd.read_csv(filepath, sep="\t", dtype=str).dropna()
    if {"source", "target"}.issubset(df.columns):
        df = df[["source", "target"]].copy()
    else:
        df = df.iloc[:, [1, 2]].copy()
    df.columns = ["src", "tgt"]
    df["src"] = df["src"].str.strip()
    df["tgt"] = df["tgt"].str.strip()
    df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)]
    df["src_lang"] = LANG_CODES[src_lang]
    df["tgt_lang"] = LANG_CODES["sinhala"]
    df = df.reset_index(drop=True)
    print(f"  Loaded {len(df):,} pairs ({src_lang})")
    return df


def split(df, val_size=2000):
    val_size = min(val_size, max(1, len(df) // 10))
    val_df = df.iloc[:val_size].reset_index(drop=True)
    train_df = df.iloc[val_size:].reset_index(drop=True)
    print(f"  Train: {len(train_df):,}   Val: {len(val_df):,}")
    return DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "val": Dataset.from_pandas(val_df, preserve_index=False),
    })


def make_tokenise_fn(tokenizer, max_length=128):
    def tokenise(batch):
        tokenizer.src_lang = batch["src_lang"][0]
        tokenizer.tgt_lang = LANG_CODES["sinhala"]
        model_inputs = tokenizer(batch["src"], max_length=max_length, truncation=True)
        labels = tokenizer(text_target=batch["tgt"], max_length=max_length, truncation=True)
        tokenizer.src_lang = batch["src_lang"][0]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return tokenise


def build(filepath, tokenizer, src_lang="english", val_size=2000, max_length=128, num_proc=1, sample=None):
    df = load_pairs(filepath, src_lang)
    if sample:
        df = df.head(sample)
        print(f"  [sample] Using {len(df):,} rows")
    dd = split(df, val_size=val_size)
    tok = make_tokenise_fn(tokenizer, max_length=max_length)
    print("  Tokenising...")
    return dd.map(
        tok,
        batched=True,
        batch_size=256,
        num_proc=max(1, int(num_proc)),
        remove_columns=["src", "tgt", "src_lang", "tgt_lang"],
    )
