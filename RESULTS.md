# Pilot 1 — Code-Mixing Measurement for Hinglish Aggression Research

**Date:** 27 October 2026
**Status:** Measurement instrument built and validated. Hypothesis not yet testable — pilot corpus lacks the required variance.

---

## 1. What we set out to do

Our independent variable is **the code-mixing pattern of a parent comment**, operationalised as three sub-variables:

| | Variable | Type |
|---|---|---|
| IV-1 | Code-Mixing Index (CMI), Gambäck & Das (2016) | continuous 0–100 |
| IV-2 | Switch points | integer |
| IV-3 | Matrix language | English / Hindi / Balanced |

All three depend on one upstream step: **token-level language identification** on Roman-script text. We flagged this as the project's primary risk. This pilot tests whether that risk is manageable before we commit to data collection.

---

## 2. Data

Public corpora for Roman-script Hinglish are harder to obtain than the literature suggests.

- **Bohra et al. (2018)** — the dataset used by Rani et al. Distributed as **tweet IDs only**; text must be requested by email. Hydration via the X API is no longer viable on a student budget.
- **Hinglish Offensive Text (Mathur et al. 2018)** — the public repo contains only a 206-word profanity lexicon, not the corpus.
- **HateCheckHIn (Das et al.)** — fully public with text. **This is what we used.**

| Subset | n | Script | Labels |
|---|---|---|---|
| multilingual | 1,130 (460 Roman-only) | Roman + Devanagari | all hateful |
| monolingual | 4,754 | Devanagari | 3,338 hateful / 1,416 nonhateful |

**Finding 0 — report this to the professor.** Data access, not modelling, is the binding constraint. We should email the Bohra and TRAC authors for corpus access **this week**; replies take time.

---

## 3. The model we built

A three-stage token-level language tagger:

1. **Lexicon lookup.** English: 234,422 words. Romanized Hindi: 2,683 forms, built by transliterating Devanagari vocabulary and expanding spelling variants (`nafarata → nafrat`), plus a 206-word Hinglish profanity lexicon.
2. **Closed-class override.** Hindi function words (pronouns, postpositions, auxiliaries) and English function words are tagged by list, not by model.
3. **Character n-gram backoff** for out-of-vocabulary tokens — TF-IDF char 1–4 grams + logistic regression, trained on lexicon membership so it costs no annotation.

`univ` (punctuation, emoji, *lol*, *omg*) and `ne` (named entities) are tagged and **excluded** from all three IVs.

---

## 4. Numbers

### 4.1 Character-level LID backoff
Held-out word types: **96.3% accuracy**, macro-F1 **0.952**.

### 4.2 Token-level tagger vs hand-tagged gold
20 real sentences, 215 tokens, hand-tagged by one annotator.

| Version | Token accuracy | Macro-F1 |
|---|---|---|
| v1 — hand-listed ambiguity set | 0.870 | 0.880 |
| **v2 — ambiguity derived from lexicon overlap** | **0.944** | **0.950** |

**Finding 1 — the diagnosis that produced the gain.** 241 surface forms appear in *both* lexicons. In v1 these fell through to the character model, which tagged frequent Hindi postpositions (*se*, *ka*, *ki*) as English every time. Treating lexicon overlap as ambiguity and resolving it by context recovered **+7.4 points of accuracy**.

**Remaining 12 errors, and what they mean:**

| Token | Issue |
|---|---|
| *to* ×2 | Hindi discourse particle vs English preposition — irreducibly ambiguous, needs context |
| *Hindu* | Is it English or Hindi? **An annotation-guideline question, not a model bug** |
| *lower, cast* | English embedded in a Hindi clause; context resolver over-pulled to Hindi |
| *sah, turant, angrezon, log* | Genuine lexicon gaps |

### 4.3 Construct validation of CMI (Experiment A)

HateCheckHIn's Roman-script block has two *designed* conditions:
- **MF1** — fully romanized Hindi → predicted low CMI
- **MF6** — English frame + romanized Hindi → predicted high CMI

Ground truth comes from the test-suite design, **independent of our tagger**.

| | n | mean CMI | SD | mean switch points |
|---|---|---|---|---|
| MF1 | 310 | **5.53** | 9.46 | 0.70 |
| MF6 | 150 | **39.74** | 7.52 | 1.87 |

Mann-Whitney U = 671, z = −17.72, p < 0.001, rank-biserial **r = 0.97**.

Matrix language: MF1 classified **99.7% Hindi** — correct by design.
CMI alone separates the two conditions at **95.4% accuracy** (threshold CMI = 29).

**Finding 2 — the instrument works.** The IV measures what it claims to measure.

### 4.4 Classifier (Experiment B)

Devanagari monolingual set transliterated to Roman script. Character 2–5 gram TF-IDF + logistic regression, 5-fold CV.

| Condition | Macro-F1 |
|---|---|
| Majority baseline | 0.413 |
| A-naive — text only, **random folds** | **0.988** |
| A — text only, **template-grouped folds** | **0.655** |
| B — CMI features only | 0.455 |
| C — text + CMI | 0.660 |

**Finding 3 — the most important result here.** HateCheck generates ~39 sentences from each of 121 templates, varying only the target group. With random folds the model memorises templates and scores 0.988. Grouping by template drops it to 0.655.

> **Leakage inflation: +0.333 macro-F1 — a third of the apparent performance was memorisation.**

Any paper in this area reporting random-split CV on template-generated data is reporting an inflated number. This is directly relevant to how we read the two papers we reviewed.

**Finding 4 — an honest null.** CMI adds **+0.005** over text alone. This is *not* evidence against our hypothesis. Mean CMI in this corpus is 0.55 (hateful) vs 0.76 (nonhateful) — the data is monolingual by construction, so there is **no code-mixing variance to exploit**. The hypothesis is untestable on this corpus. That is precisely why we need threaded, naturally code-mixed data.

---

## 5. Limitations — state these before being asked

1. **Gold tags are single-annotator.** 215 tokens, one annotator. Not a reportable number until the team re-annotates independently and we compute Krippendorff's α. Rani et al. got α = 0.47; we should beat it.
2. **Lexicon overlaps the test corpus.** Our Hindi lexicon derives from the Devanagari half of the same suite. In-suite accuracy is optimistic; the gold evaluation is the honest figure.
3. **Transliterated ≠ naturalistic.** Transliteration gives one canonical spelling per word. Real Hinglish has many. The 0.655 is an *upper bound* for scraped comments, not an estimate.
4. **HateCheck is a functional test suite,** not scraped social media. Template-generated, no conversational structure, no thread context — so it cannot test the parent-reply hypothesis at all.
5. **CMI is corpus-dependent.** The 2,683-form Hindi lexicon came from 1,263 Devanagari types in a narrow template set. It will need substantial expansion for real comment data.

---

## 6. What this changes about the plan

- The tagger risk is **manageable** — 94.4% on real text, with a clear error taxonomy. Proceed.
- *Hindu*, *lower cast* and similar cases are **guideline problems, not engineering problems**. Write the annotation guideline before scraping, and decide explicitly how borrowed and community-name tokens are tagged.
- Adopt **template- or thread-grouped splits as the default** for every evaluation from here on. Never report random-split CV.
- The CMI null confirms the design requires **threaded YouTube data with genuine code-mixing variance**. Nothing short of that tests the hypothesis.

## 7. Next steps

1. Email Bohra et al. and the TRAC-2 organisers for corpus access — this week.
2. Register for a YouTube Data API key; pilot `commentThreads.list` on 3 videos to confirm reply nesting.
3. Three annotators independently re-tag the 215 gold tokens; compute Krippendorff's α.
4. Write the annotation guideline, including the *Hindu* / borrowed-token decision.
5. Expand the romanized Hindi lexicon from naturalistic text once collected.

---

### Reproducing

```
src/build_lexicon.py        build lexicons
src/char_model.py           train character n-gram LID
src/eval_gold.py            tagger vs hand-tagged gold      -> 0.944
src/exp_cmi_structure.py    CMI construct validation        -> r = 0.97
src/exp_classifier.py       classifier + leakage comparison -> 0.655 vs 0.988
```
Full console output: `out/full_run.log`

**Sources:** [HateCheckHIn](https://github.com/hate-alert/HateCheckHIn) · [Hinglish Profanity List](https://github.com/pmathur5k10/Hinglish-Offensive-Text-Classification) · [Bohra et al. dataset](https://github.com/deepanshu1995/HateSpeech-Hindi-English-Code-Mixed-Social-Media-Text)
