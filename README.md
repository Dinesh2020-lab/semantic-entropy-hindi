# Semantic Entropy for Hallucination Detection: English vs Hindi

**Status:** In progress. Pipeline implemented; experiments are being run.

## Research question

Does semantic-entropy-based hallucination detection work as well in Hindi as in English for a small open-source LLM?

## Motivation

Large language models often produce fluent but incorrect answers (hallucinations). Semantic entropy detects them without labelled data: the model answers the same question many times, the answers are grouped by meaning, and high disagreement between meanings signals a likely error. The method has been studied almost entirely in English, and the NLI model used to group answers may behave differently in Indian languages.

## Method

1. Sample 300 TriviaQA questions. Create a Hindi version by machine translation (`Helsinki-NLP/opus-mt-en-hi`), so both languages use the same questions.
2. For each question, generate one greedy answer and ten high-temperature samples with `Qwen2.5-3B-Instruct`.
3. Cluster the samples by meaning using bidirectional entailment from a multilingual NLI model (`mDeBERTa-v3-base-xnli`).
4. Compute the entropy over cluster sizes as the uncertainty score.
5. Label each greedy answer correct or incorrect with an LLM judge against the gold answer.
6. Measure how well the entropy score predicts incorrect answers (AUROC), per language.

## Planned analysis

- Multiple random seeds (0, 1, 2)
- Ablation over the number of samples (3, 5, 10)
- Manual error analysis of about 30 Hindi failure cases (translation errors, clustering errors, factual errors, script mixing)

## How to run

```bash
pip install -r requirements.txt
python semantic_entropy_hindi.py --n_questions 300 --n_samples 10 --lang en
python semantic_entropy_hindi.py --n_questions 300 --n_samples 10 --lang hi
```

Use a GPU (for example a free Kaggle T4). Results are saved as `results_<lang>_seed<seed>.jsonl`.

## Results

To be added after the experiments finish.

| Language | Accuracy | AUROC |
|---|---|---|
| English | - | - |
| Hindi | - | - |

## Limitations

- Machine-translated Hindi questions may contain errors
- Only one small model is tested
- Modest sample size
- The LLM judge adds noise to correctness labels

## Future work

- Use IndicTrans2 for better translation
- Test an Indic-focused model
- Build a small Meitei (Manipuri) evaluation set

## References

- Farquhar et al., "Detecting hallucinations in large language models using semantic entropy", *Nature*, 2024
- Kuhn et al., "Semantic Uncertainty", ICLR 2023
- Manakul et al., "SelfCheckGPT", 2023
