"""Hardened NLLB-200 fine-tuning runner for Colab."""
import argparse
import glob
import inspect
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import dataset as ds
import metrics as ev

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
TARGET_LANG = "sin_Sinh"


def get_target_token_id(tokenizer, lang_code):
    tid = tokenizer.convert_tokens_to_ids(lang_code)
    if tid is not None and tid != tokenizer.unk_token_id:
        return tid
    tid = tokenizer.get_vocab().get(lang_code)
    if tid is not None:
        return tid
    raise ValueError(f"Could not find {lang_code} in tokenizer vocab.")


def disable_rng_state_restore(checkpoint):
    """Avoid PyTorch 2.6 weights_only failure on Trainer RNG-state files.

    This preserves model, optimizer, scheduler, and trainer-state resume. It only
    skips exact random-generator restoration, which is acceptable for continuing
    training from a trusted checkpoint when torch.load blocks rng_state.pth.
    """
    if not checkpoint:
        return
    for rng_file in glob.glob(os.path.join(checkpoint, "rng_state*.pth")):
        disabled = rng_file + ".disabled"
        if not os.path.exists(disabled):
            os.rename(rng_file, disabled)
            print(f"  Disabled RNG restore file for PyTorch 2.6 compatibility: {os.path.basename(rng_file)}")


def checkpoint_step(path):
    try:
        return int(Path(path).name.split("-")[-1])
    except Exception:
        return -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--val-size", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--src-lang", default="english", choices=["english", "tamil", "hindi"])
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--num-proc", type=int, default=1)
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--resume", default=None, help="Path to checkpoint-NNN folder")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=2500)
    p.add_argument("--logging-steps", type=int, default=100)
    p.add_argument("--disable-rng-restore", action="store_true", default=True)
    p.add_argument("--no-metrics", action="store_true", help="Disable BLEU/chrF during training eval")
    return p.parse_args()


def training_args_kwargs(args, out_dir, device):
    kwargs = {
        "output_dir": out_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "warmup_steps": args.warmup,
        "weight_decay": 0.01,
        "fp16": args.fp16 and device == "cuda",
        "predict_with_generate": not args.no_metrics,
        "generation_max_length": args.max_length,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 5,
        "load_best_model_at_end": False,
        "logging_dir": os.path.join(out_dir, "logs"),
        "logging_steps": args.logging_steps,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "push_to_hub": False,
    }
    sig = inspect.signature(Seq2SeqTrainingArguments)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return kwargs


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 60)
    print("  NLLB-200 Sinhala Fine-tuning")
    print("=" * 60)
    print(f"  Device  : {device}  |  FP16: {args.fp16 and device == 'cuda'}")
    print(f"  Batch   : {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    if args.sample:
        print(f"  Sample  : {args.sample} rows")
    if args.resume:
        print(f"  Resume  : {args.resume}")
    print("=" * 60 + "\n")

    print("[1/4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.src_lang = ds.LANG_CODES[args.src_lang]
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    sin_id = get_target_token_id(tokenizer, TARGET_LANG)
    model.config.forced_bos_token_id = sin_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.eos_token_id
    model.generation_config.forced_bos_token_id = sin_id
    model.generation_config.max_new_tokens = args.max_length
    if model.generation_config.decoder_start_token_id is None:
        model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    if model.generation_config.bos_token_id is None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    print(f"  sin_Sinh id: {sin_id}")

    print("\n[2/4] Building dataset...")
    data = ds.build(
        filepath=args.data,
        tokenizer=tokenizer,
        src_lang=args.src_lang,
        val_size=args.val_size,
        max_length=args.max_length,
        num_proc=args.num_proc,
        sample=args.sample,
    )

    print("\n[3/4] Configuring trainer...")
    resume_ckpt = args.resume if args.resume and os.path.isdir(args.resume) else None
    if resume_ckpt:
        out_dir = str(Path(resume_ckpt).resolve().parent)
        if args.disable_rng_restore:
            disable_rng_state_restore(resume_ckpt)
    else:
        run_name = f"nllb-sinhala-{args.src_lang}-{datetime.now().strftime('%Y%m%d-%H%M')}"
        out_dir = os.path.join(args.output_dir, run_name)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding=True)
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs(args, out_dir, device))
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=None if args.no_metrics else ev.make_compute_metrics(tokenizer),
    )

    print("\n[4/4] Training...\n")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = os.path.join(out_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "src_lang": args.src_lang, "epochs": args.epochs, "final_dir": final_dir, "out_dir": out_dir}, f, indent=2)
    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Model saved to: {final_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
