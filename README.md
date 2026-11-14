# Semantic Entropy for Hallucination Detection: English vs Hindi

Does semantic-entropy hallucination detection work for Hindi questions with a small open LLM?

## Summary

I implemented semantic entropy (Farquhar et al., *Nature* 2024) and evaluated it with **Qwen2.5-3B-Instruct** on 300 TriviaQA questions in three settings: English, Hindi questions answered in English, and Hindi questions answered in Hindi. Hindi questions were machine-translated from the same English questions with NLLB, so all settings use identical content.

Main findings (single seed, N = 300 per setting):

1. **English:** semantic entropy detects wrong answers well (AUROC 0.80, 95% CI 0.73-0.85).
2. **Hindi question, English answer:** accuracy falls from 27.7% to 4.7%. AUROC is 0.71, but the confidence interval (0.51-0.88) is wide because only 14 answers were correct, so this run cannot show whether detection quality differs from English.
3. **Hindi question, Hindi answer:** only 1 of 300 answers was judged correct, so AUROC cannot be measured. In my manual reading of the first 70 answers, most Hindi answers were incoherent or off-topic, so the near-zero accuracy mainly reflects this model's weak Hindi generation.
4. **Entropy is much higher in Hindi.** The mean number of distinct meaning-clusters among 10 samples rises from 4.9 (English) to 7.2 (Hindi question, English answer) and 7.8 (Hindi answer). I did not test whether this comes from real disagreement between samples or from the NLI model being less reliable on Hindi.

## Method

1. Sample 300 TriviaQA validation questions (`rc.nocontext`, seed 0). Hindi versions are translated with `facebook/nllb-200-distilled-600M`.
2. For each question, generate one greedy answer (the one being judged) and 10 samples at temperature 1.0 (top-p 0.95), max 64 new tokens, with Qwen2.5-3B-Instruct.
3. Cluster the samples by meaning: two answers join a cluster if a multilingual NLI model (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) predicts entailment in both directions, given the question.
4. Semantic entropy = entropy of the cluster-size distribution.
5. Correctness of the greedy answer is judged by the same Qwen model (yes/no) against the English gold answer.
6. Metric: AUROC of semantic entropy for predicting incorrect answers, with a 95% bootstrap confidence interval (2,000 resamples).

## Results

| Setting | Question | Answer | Correct / N | Accuracy | AUROC (95% CI) | Mean entropy (correct / wrong) | Mean clusters (of 10) |
|---|---|---|---|---|---|---|---|
| `en` | English | English | 83 / 300 | 0.277 | 0.796 (0.728-0.854) | 0.456 / 1.411 | 4.87 |
| `hi_ansen` | Hindi | English | 14 / 300 | 0.047 | 0.707 (0.513-0.876) | 1.055 / 1.773 | 7.18 |
| `hi` | Hindi | Hindi | 1 / 300 | 0.003 | not reported (1 correct answer) | - / 1.903 | 7.83 |

Raw outputs are in `results/` (`results_en_seed0.jsonl`, `results_hi_ansen_seed0.jsonl`, `results_hi_seed0.jsonl`).

## Error analysis (Hindi answers, `hi` setting)

- 297 of 300 answers are in Devanagari script; 13 end in a broken character (cut off at the 64-token limit).
- By a rough word-overlap heuristic, about 192 of 300 answers mostly reuse words from the question instead of giving an answer.
- In a manual read of the first 70 answers, the judge's "wrong" labels were almost always right. A few were borderline (for example, the answer "9 x 9 squares" for gold "81").

## Limitations

- **One model, one seed, 300 questions.** No claim is made beyond Qwen2.5-3B on TriviaQA.
- **The LLM judge is noisy.** In English, the gold string appears in the model's answer in 99 cases, but the judge accepted only 74 of them, so accuracy figures are approximate. In Hindi, the judge compares Devanagari answers with English gold answers.
- **Too few correct answers in Hindi.** With 14 and 1 correct answers, AUROC is unreliable or unavailable. A larger model or Indic-focused model is needed for a fair comparison.
- **Machine-translated questions** can lose or distort names.
- **NLI clustering was not validated for Hindi**, so the higher Hindi entropy cannot be attributed to the model alone.

## How to reproduce

```bash
pip install -r requirements.txt
python semantic_entropy_hindi.py --n_questions 300 --n_samples 10 --lang en
python semantic_entropy_hindi.py --n_questions 300 --n_samples 10 --lang hi --answer_lang en
python semantic_entropy_hindi.py --n_questions 300 --n_samples 10 --lang hi
```

Run on a GPU (a free Colab T4 was used).

## Future work

- Use a larger or Indic-focused model (for example Sarvam-1) so Hindi accuracy is high enough to measure AUROC.
- Use a stronger judge and answer aliases for correctness.
- Validate the NLI clustering on Hindi answer pairs by hand.
- Use IndicTrans2 for translation, and add a small Meitei evaluation set.

## References

- Farquhar et al., "Detecting hallucinations in large language models using semantic entropy", *Nature*, 2024
- Kuhn et al., "Semantic Uncertainty", ICLR 2023
- Manakul et al., "SelfCheckGPT", 2023
