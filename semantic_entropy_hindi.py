"""
Semantic entropy hallucination detection: English vs Hindi (same questions).
Run on Kaggle/Colab with a GPU:
    pip install transformers datasets accelerate scikit-learn sentencepiece sacremoses
    python semantic_entropy_hindi.py --n_questions 300 --lang en
    python semantic_entropy_hindi.py --n_questions 300 --lang hi
"""
import argparse, json, random, itertools
import numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, pipeline)

GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
TRANSLATOR = "Helsinki-NLP/opus-mt-en-hi"  # quick baseline; swap for IndicTrans2 later

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def load_questions(n, lang, seed):
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.shuffle(seed=seed).select(range(n))
    qs = [{"q": r["question"], "gold": r["answer"]["value"]} for r in ds]
    if lang == "hi":
        tr = pipeline("translation", model=TRANSLATOR, device=0 if torch.cuda.is_available() else -1)
        out = tr([x["q"] for x in qs], max_length=128, batch_size=16)
        for x, o in zip(qs, out):
            x["q"] = o["translation_text"]
    return qs

def build_prompt(tok, q, lang):
    inst = ("Answer briefly in one short phrase." if lang == "en"
            else "संक्षेप में हिंदी में एक छोटे वाक्यांश में उत्तर दीजिए।")
    msgs = [{"role": "user", "content": f"{inst}\n\n{q}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate(model, tok, prompt, n, temperature):
    inp = tok(prompt, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=32, pad_token_id=tok.eos_token_id)
    if temperature == 0:
        out = model.generate(**inp, do_sample=False, **kw)
    else:
        out = model.generate(**inp, do_sample=True, temperature=temperature,
                             top_p=0.95, num_return_sequences=n, **kw)
    return [tok.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]

@torch.no_grad()
def entails(nli_tok, nli, a, b):
    x = nli_tok(a, b, return_tensors="pt", truncation=True).to(nli.device)
    pred = nli(**x).logits.argmax(-1).item()
    return nli.config.id2label[pred].lower() == "entailment"

def semantic_entropy(question, answers, nli_tok, nli):
    clusters = []  # list of lists
    for a in answers:
        for c in clusters:
            rep = c[0]
            if (entails(nli_tok, nli, f"{question} {rep}", f"{question} {a}") and
                    entails(nli_tok, nli, f"{question} {a}", f"{question} {rep}")):
                c.append(a); break
        else:
            clusters.append([a])
    p = np.array([len(c) for c in clusters]) / len(answers)
    return float(-(p * np.log(p)).sum()), len(clusters)

def judge_correct(model, tok, gold, pred):
    """LLM judge: is predicted answer the same as the English gold answer?"""
    msg = (f"Gold answer (English): {gold}\nPredicted answer: {pred}\n"
           "Does the predicted answer mean the same as the gold answer? Reply only yes or no.")
    p = tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
    return generate(model, tok, p, 1, 0)[0].lower().startswith("yes")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "hi"], default="en")
    ap.add_argument("--n_questions", type=int, default=300)
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    seed_all(a.seed)

    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float16, device_map="auto")
    nli_tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(model.device).eval()

    qs = load_questions(a.n_questions, a.lang, a.seed)
    rows = []
    for i, x in enumerate(qs):
        prompt = build_prompt(tok, x["q"], a.lang)
        main_ans = generate(model, tok, prompt, 1, 0)[0]
        samples = generate(model, tok, prompt, a.n_samples, 1.0)
        se, k = semantic_entropy(x["q"], samples, nli_tok, nli)
        correct = judge_correct(model, tok, x["gold"], main_ans)
        rows.append({**x, "main": main_ans, "samples": samples, "se": se,
                     "n_clusters": k, "correct": correct})
        if i % 20 == 0:
            print(i, x["q"][:60], "|", main_ans, "| SE", round(se, 2), "| correct", correct, flush=True)

    y = [0 if r["correct"] else 1 for r in rows]  # 1 = hallucination
    se = [r["se"] for r in rows]
    print(f"\nLang={a.lang} N={len(rows)} accuracy={1 - np.mean(y):.3f}")
    if len(set(y)) > 1:
        print(f"AUROC (semantic entropy -> detects wrong answers): {roc_auc_score(y, se):.3f}")
    with open(f"results_{a.lang}_seed{a.seed}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
