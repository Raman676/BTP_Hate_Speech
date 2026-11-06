# Runbook — what to run, in what order

Every command runs from the project root (`cmi/`).

---

## Step 0 — one-time setup

```bash
pip install pandas numpy scikit-learn joblib indic-transliteration \
            english-words google-api-python-client requests
python src/build_lexicon.py      # builds out/lex_*.json
python src/char_model.py         # trains out/char_lid.joblib
```

Verify the measurement pipeline still passes:

```bash
python src/eval_gold.py          # expect token accuracy 0.972
python src/exp_cmi_structure.py  # expect r = 0.976, separation 0.954
```

---

## Step 1 — collect comments  ← **you are here**

Your key already lives in `config_local.py` (gitignored, so it won't travel if you share the project).

**Find videos automatically** using the built-in body-shaming queries:

```bash
python src/collect_youtube.py --search
```

This searches 14 Hindi/Hinglish queries for appearance-related content, writes the candidates to `data/videos_found.csv`, then collects threaded comments from all of them into `data/comments.csv`.

**Do a small run first**, skim what comes back, then scale:

```bash
python src/collect_youtube.py --search --per-query 3 --max-threads 25
```

**Or hand-pick videos** if you'd rather choose yourself:

```bash
python src/collect_youtube.py --videos ID1 ID2 ID3
```

**Quota.** 10,000 units/day. Each search costs 100 units (14 queries ≈ 1,400); comment collection costs ~1 unit per 100 comments. You can afford roughly 80 searches a day, not 800.

**What you get.** Every comment is flagged `is_appearance` if it mentions appearance vocabulary. This is a flag, **not a filter** — the base rate of body-shaming is itself a finding, and hard-filtering would miss implicit cases like `pehle better thi` that carry no appearance vocabulary at all. Use `--appearance-only` to concentrate labelling effort, but declare it in the paper if you do.

**Target.** 3,000–5,000 comments across 500+ threads. Check the summary: if `replies (have a parent)` is small, those videos aren't generating argument and you need different ones. If `mentioning appearance` is under 200, the videos aren't drawing body talk at all.

---

## Step 2 — label pass L0 (no context)

```bash
python src/make_sheets.py l0 data/comments.csv
```

Writes `labels/label_L0.csv` with four columns: `uid, text, label, note`.

Open it in Excel or Google Sheets. Put `3`, `2`, `1`, `0`, or `X` in the **label** column for every row, per `ANNOTATION_GUIDELINE.md` §2–§4. Use **note** freely — those notes become your error analysis.

**Do not compute or look at any IV while labelling.** The sheet has no IV columns for exactly this reason.

```bash
python src/make_sheets.py lock labels/label_L0.csv
```

Refuses to lock if anything is blank or invalid. Records a SHA-256 so later tampering is caught.

---

## Step 3 — wait 48 hours, then label pass L1 (with context)

```bash
python src/make_sheets.py l1 data/comments.csv
```

Refuses to run until L0 is locked. Fresh random order so you aren't recalling your own L0 answers. Label the same way, then lock it.

---

## Step 4 — reliability

```bash
python src/make_sheets.py retest labels/label_L0.csv   # 10% blind re-label
# label labels/label_L0_retest.csv, then:
python src/make_sheets.py agree labels/label_L0.csv labels/label_L0_retest.csv
```

Gives Cohen's κ (you vs yourself) and Krippendorff's α, with Rani et al.'s 0.47 as the benchmark.

For inter-annotator α, have Adhiraj and Arpita label the same 300-row subset and run `agree` on each pair.

---

## Step 5 — compute the IVs (only now)

Labels are locked, so looking at IVs can no longer contaminate them.

```python
import sys, json, joblib, pandas as pd
sys.path.insert(0, 'src')
from lid import LanguageTagger, describe, TOKEN_RE
from char_model import as_tagger_fn
import profanity as P

tagger = LanguageTagger(json.load(open('out/lex_en.json')),
                        json.load(open('out/lex_hi.json')),
                        char_model=as_tagger_fn(joblib.load('out/char_lid.joblib')))
pm = P.load()

df = pd.read_csv('data/comments.csv')
iv = pd.DataFrame([{**describe(t, tagger), **pm.measure(TOKEN_RE.findall(t))}
                   for t in df.text])
pd.concat([df, iv], axis=1).to_csv('out/comments_with_ivs.csv', index=False)
```

Gives you **IV-1** `cmi`, **IV-2** `switch_points`, **IV-3** `swear_freq` plus `profanity_valence`.

---

## Step 6 — Mica (Kaggle, GPU on)

Not runnable in this project's sandbox: no GPU, and huggingface.co is blocked. Run it in a Kaggle notebook.

Start the llama.cpp server per the setup block in `src/mica_classify.py`, then:

```bash
python src/mica_classify.py health
python src/mica_classify.py run --in data/comments.csv --condition L0
python src/mica_classify.py run --in data/comments.csv --condition L1
python src/mica_classify.py compare --l0 out/mica_L0.csv --l1 out/mica_L1.csv
```

---

## Step 7 — the diagnostic you must not skip

Join your human labels to Mica's predictions and the IVs, then:

```bash
python src/mica_classify.py degrade --in out/merged.csv
```

If this prints **DEGRADATION DETECTED**, Mica is weaker on code-mixed text, and any IV→DV relationship drawn from model labels is confounded. Human labels stay primary, and the degradation itself becomes a reportable finding about off-the-shelf models on Hinglish.

---

## File map

| File | Purpose |
|---|---|
| `src/lid.py` | tokenise, tag language, **IV-1 CMI**, **IV-2 switch points** |
| `src/profanity.py` | **IV-3 swear frequency** + valence split |
| `src/build_lexicon.py` | build Hindi/English word lists |
| `src/char_model.py` | train the character n-gram language tagger |
| `src/collect_youtube.py` | **Step 1** — video search + threaded comment collection |
| `config_local.py` | your API key (gitignored) |
| `src/make_sheets.py` | **Steps 2–4** — blinded sheets, locking, κ/α |
| `src/mica_classify.py` | **Steps 6–7** — Mica + degradation check |
| `src/eval_gold.py` | validation: tagger vs gold → 0.972 |
| `src/exp_cmi_structure.py` | validation: CMI construct → r = 0.976 |
| `src/exp_classifier.py` | Pilot 1 classifier + the leakage result |
| `ANNOTATION_GUIDELINE.md` | label scheme, decision rules, blinding protocol |
| `RESULTS.md` | Pilot 1 write-up |
