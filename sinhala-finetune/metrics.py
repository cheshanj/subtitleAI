"""Metrics using sacrebleu directly. No import of Hugging Face evaluate."""
import numpy as np
import sacrebleu as sb


def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        bleu = sb.corpus_bleu(decoded_preds, [decoded_labels])
        chrf = sb.corpus_chrf(decoded_preds, [decoded_labels])
        return {"bleu": round(bleu.score, 4), "chrf": round(chrf.score, 4)}
    return compute_metrics


def benchmark_checkpoint(model, tokenizer, pairs, src_lang="eng_Latn", tgt_lang="sin_Sinh", max_new_tokens=128, num_beams=4, device="cpu"):
    import torch
    model.eval()
    model.to(device)
    tokenizer.src_lang = src_lang
    preds, refs = [], []
    for pair in pairs:
        inputs = tokenizer(pair["src"], return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=3,
            )
        preds.append(tokenizer.decode(ids[0], skip_special_tokens=True).strip())
        refs.append(pair["ref"].strip())
    bleu = sb.corpus_bleu(preds, [refs])
    chrf = sb.corpus_chrf(preds, [refs])
    samples = [{"src": pairs[i]["src"], "ref": refs[i], "pred": preds[i]} for i in range(min(5, len(pairs)))]
    return {"bleu": round(bleu.score, 4), "chrf": round(chrf.score, 4), "samples": samples}
