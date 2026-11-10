# Learn Everything — Body-Shaming in Hinglish YouTube Comments

**BTP · Raman Dubey, Adhiraj Singh Brar, Arpita Pawar · November 2026**

This is the one file to read before explaining or presenting the project. It covers every idea, step, number and decision so far. It has three parts: Part A teaches the concepts, Part B tells the story of what we did, and Part C helps you present and defend it.

Numbers come from `out/RESULTS_L0.md`, which `python src/analyze_l0.py` regenerates.

---

## Contents

- **Part A — Concepts**
  1. The research question in one paragraph
  2. Code-mixing and Hinglish
  3. Language identification (LID)
  4. The three independent variables
  5. The dependent variable: the label scheme
  6. Why we label twice (L0 / L1)
  7. Blinding and why it matters
  8. Statistics used, explained simply
- **Part B — What we did, step by step**
  9. Pilot 1: building and validating the instrument
  10. Collecting the data
  11. Labelling
  12. Problems we hit and how we fixed them
  13. Results
- **Part C — Presenting it**
  14. The story in 60 seconds
  15. Slide-by-slide outline
  16. Questions the professor will ask, with answers
  17. Limitations — say them before you're asked
  18. What comes next
  19. Glossary
  20. Reproducing everything

---

# Part A — Concepts

## 1. The research question in one paragraph

Indian YouTube comment sections are full of Hinglish: Hindi and English mixed in one sentence and typed in Roman letters. Under videos about weight, skin colour and appearance, much of that talk is body-shaming. We ask: **does the way people mix languages predict whether a comment is body-shaming?** Specifically, does a *parent* comment's code-mixing predict whether the *reply* to it is hostile? Second question: **does a human judge a comment differently when they can see what it was replying to?**

It's a **linguistics** study, not an engineering one. The goal is a defensible finding, not a high accuracy score.

## 2. Code-mixing and Hinglish

**Code-mixing** means using two languages inside one utterance:

> *bhai ye **body** bilkul **fit** hai*: "brother, this body is totally fit"

Hindi provides the frame (*bhai, ye, bilkul, hai*) and English words are slotted in (*body, fit*).

Two things that trip people up:

- **Script ≠ language.** *bhai ye to bilkul kharab lag rahi hai* is written in Roman letters but is **100% Hindi**. We keep only Roman-script comments (we drop Devanagari like *मोटी*), but that's a filter on the writing system, not the language.
- **No standard spelling.** One Hindi word appears as *zyada / jyada / jiyada / zada*. Every tool we built has to cope with this.

## 3. Language identification (LID)

Before you can measure mixing, you have to know which language each word is in. This is **token-level language identification**, and it's the riskiest part of the project. If it's wrong, every code-mixing number is wrong.

Our tagger works in three stages:

| Stage | What it does | Example |
|---|---|---|
| 1. Dictionary lookup | 234,422 English words; 2,683 romanized Hindi forms | *body* → en, *bilkul* → hi |
| 2. Function-word override | Closed lists of very common words always win | *the, is* → en; *hai, ka, se* → hi |
| 3. Character model | For unknown words, a classifier looks at letter patterns (1–4 letter chunks) | *angrezon* → hi |

The tagger also marks two kinds of token that are excluded from the scores:

- `univ`: emoji, punctuation, numbers, *lol, haha*
- `ne`: names (people, brands, films)

**The key insight from Pilot 1:** 241 words exist in *both* dictionaries (*se, ka, ki, are, age*). The first version tagged *se, ka, ki* as English every time. Treating any word in both dictionaries as ambiguous and deciding it from the surrounding words raised accuracy by **+7.4 points**.

**How good is it?** On 215 hand-tagged words from real sentences: **97.2% accuracy**. The 6 errors are words like *to* (Hindi particle vs English preposition) and *Hindu* (which language is it?), which are partly questions for the guideline rather than bugs.

## 4. The three independent variables (IVs)

All three are calculated in `src/lid.py` and `src/profanity.py`, on the comment **and** on its parent.

### IV-1: Code-Mixing Index (CMI), Gambäck & Das (2016)

$$\text{CMI} = 100 \times \left(1 - \frac{\text{count of the majority language}}{\text{number of hi + en words}}\right)$$

- 0 = completely one language
- 50 = perfectly balanced (the maximum for two languages)

**Worked example:** *bhai ye body bilkul fit hai*

| bhai | ye | body | bilkul | fit | hai |
|---|---|---|---|---|---|
| hi | hi | en | hi | en | hi |

Hindi = 4, English = 2, n = 6 → CMI = 100 × (1 − 4/6) = **33.3**

### IV-2: Switch points

How many times the language changes between adjacent words. In the example: hi→hi→**en**→**hi**→**en**→**hi** = **4 switches**.

CMI says *how much* of each language is used; switch points say *how interleaved* they are. "Five Hindi words then five English words" and "alternating word by word" have the same CMI but very different switch counts.

### IV-3: Swear frequency + valence

- **Frequency** = swear words ÷ total words. Swears are matched against the Hinglish Profanity List (Mathur et al. 2018) plus a short English list.
- **Valence** is whether the swear is aimed at someone:
  - *directed*: near *tu, tera, you*, or a threat verb (*mar, nikal*)
  - *intensifier*: near praise (*mast, badhiya*)
  - *ambiguous*: neither

The central idea, project rule 7: **profanity ≠ aggression.** *kya mast h ye bhenchod* ("this is bloody awesome") is praise. A system that just counts swear words calls it hate speech.

### Also measured: matrix language

The dominant language of the comment: **Hindi** if ≥ 60% Hindi words, **English** if ≥ 60% English, otherwise **Balanced**.

## 5. The dependent variable (DV): the label scheme

Every comment is hand-labelled by a human. **Labels from a model are never used as the DV** (rule 1), because a model that's worse on code-mixed text would create a fake link between code-mixing and the labels.

| Code | Meaning | Example |
|---|---|---|
| **3** | Body-shaming: hostility about **appearance** | *kitni moti ho gayi hai ye* |
| **2** | Other aggression, not about appearance | *tu bilkul bewakoof hai* |
| **1** | Profane but not aggressive | *kya mast h ye bhenchod* |
| **0** | Neither | *ye video bakwaas hai* (criticises a thing, not a person) |
| **X** | Can't judge without context | *pehle better thi* ("she was better before": looks? acting?) |

**Why 3 and 2 are separate:** otherwise the study is about "aggression" in general and can't say anything about body-shaming specifically.

**Why X counts as data:** both papers we reviewed named missing context as their main source of error. How often X is used, and how much that drops when context is shown, is a headline result.

**Decision rules (R1–R10)** are applied in order, and the first one that fits decides. The most important:
- **R1:** criticising a *thing* isn't aggression.
- **R5:** label sarcasm by what is meant, not what is literally said.
- **R8:** shaming a celebrity who isn't in the conversation still counts.
- **R9:** talking about your own body isn't body-shaming.

Full rules are in `ANNOTATION_GUIDELINE.md`.

## 6. Why we label twice (L0 / L1)

| Pass | The annotator sees | Purpose |
|---|---|---|
| **L0** | The comment alone | Judgement without context |
| **L1** | The comment **plus its parent** | Judgement with context |

The same comments, labelled twice, in a different random order each time. **Context is the manipulated variable**: any label that changes between L0 and L1 shows that context changed the judgement. L0 must be finished and **locked** (its SHA-256 fingerprint recorded) before L1 is created, because you can't un-see context.

## 7. Blinding and why it matters

The annotator (Raman) also designed the hypothesis. If you can see "CMI = 40" while labelling, you might unconsciously label in favour of your hypothesis. This is **experimenter bias**. The protections:

1. The labelling sheet contains only `uid, text, label, note`. No scores.
2. Comments are in random order (seed recorded).
3. L0 is locked before L1.
4. At least 48 hours between L0 and L1.
5. Reliability checks: re-label 10% yourself (test–retest), and have teammates label 300 (inter-annotator agreement).

**What actually happened:** protections 1–3 were followed for L0. The scores were calculated *before* L1, so **L1 will not be blind to them**. That has to be disclosed in the paper (see §17).

## 8. Statistics used, explained simply

| Tool | What it answers | How to read it |
|---|---|---|
| **Percent with 95% CI (Wilson)** | How common is each label, with uncertainty? | "6.2% (4.7–8.1%)": the true rate is very likely in that range |
| **Kruskal–Wallis** | Do the labels differ in CMI? (no normal-distribution assumption) | p > 0.05 → no evidence of a difference |
| **χ² (chi-square)** | Is matrix language related to the label? | Same reading |
| **Logistic regression** | Does X raise the *odds* of label Y, holding other things constant? | **Odds ratio (OR)**: 1 = no effect, > 1 raises the odds, < 1 lowers them |
| **"Per +10 CMI"** | Scaling | OR 1.10 means each 10 extra CMI points multiplies the odds by 1.10 |
| **Clustered standard errors** | Comments on one video aren't independent | Widens CIs honestly |
| **Conditional logit (within-video)** | Compares comments **only within the same video** | Removes everything that differs *between* videos (topic, audience) |
| **Fisher exact test** | A 2×2 association when counts are tiny | Used for swearing (only 14 cases) |
| **Spearman ρ** | Rank correlation between two numbers | 0 = none, 1 = perfect |

### The most important idea: Simpson's paradox

A trend that shows up in pooled data can disappear or **reverse** within groups. Here, videos differ in *both* their language mix *and* how hostile their comment sections are, for reasons unrelated to code-mixing (topic, audience). If you pool all videos, you're partly comparing *videos*, not comments. The **within-video** model compares comments under the same video, so topic and audience are held fixed. **The within-video number is the one to report.**

### Why we never use random train/test splits

Pilot 1 got **0.988** F1 with random splits but **0.655** when whole templates were held out. A third of the "performance" was memorisation. Always group by video, thread or template.

---

# Part B — What we did, step by step

## 9. Pilot 1: building and validating the instrument (Oct 2026)

| Check | Result |
|---|---|
| Character-model accuracy on held-out words | 96.3% (macro-F1 0.952) |
| Tagger vs 215 hand-tagged real tokens | **97.2%** (v1 87.0% → v2 after the ambiguity fix) |
| CMI construct validity: HateCheckHIn MF1 (all Hindi) vs MF6 (English + Hindi) | mean CMI 5.0 vs 39.8; rank-biserial **r = 0.976**; CMI alone separates them with **95.4%** accuracy |
| Classifier: random vs template-grouped folds | 0.988 vs **0.655** → leakage of +0.333 |
| CMI added to the classifier | +0.005 (no effect, but that corpus had no code-mixing variation, so it couldn't test the hypothesis) |

**Lesson:** the measuring tool works, and the hypothesis needs real, threaded, naturally code-mixed data. Hence YouTube.

## 10. Collecting the data (6 Nov 2026)

**Why YouTube:** its comment API returns each top-level comment *with its replies nested*. That parent→reply link is the basis of the main IV. X/Twitter costs $200–5,000/month and doesn't return clean threads.

**Privacy:** usernames and channel IDs are never stored. Each author becomes a salted hash, so "same person replied twice" is still detectable without storing identity.

**How:**
1. Validation re-run on the new machine: 0.972 accuracy and r = 0.976, matching the originals.
2. 14 built-in search queries (e.g. *skinny shaming india*, *celebrity body transformation bollywood*, *dark skin beauty india hindi*), 3 videos each → 42 videos.
3. Raman's 4 hand-picked videos added (`fgHN5ErCAdk`, `_ypeyzfD4no`, `FWe5d5Qcf04`, `u4gEBRSKi2E`). One returned no comments.
4. Cleaning:
   - **Dropped 4 Malayalam/Tamil/Telugu videos:** the tagger only knows Hindi and English, so romanized Malayalam would get meaningless CMI scores.
   - **Dropped 1 Pakistani video:** Roman Urdu can't be told apart from Hindi; we keep the population Indian.
   - **Dropped 4 influencer-scandal videos:** off-topic, with only 2–4% of comments mentioning appearance.
5. **Pool: 15,488 comments.** Saved in `data/comments_pool.csv`.
6. **Sample:** at most **30 comments per video**, taking whole threads in a fixed random order → **786 comments, 172 threads, 29 videos, 627 replies**.

The per-video cap stops one video from dominating. Before it, three videos supplied half of all comments.

## 11. Labelling (6–10 Nov 2026)

- `make_sheets.py l0` made `labels/label_L0.csv`: 786 rows, random order, seed 20261106.
- Raman labelled all 786 over four sessions, pushing to GitHub after each.
- Locked on 10 Nov. SHA-256 `7a464665…`, recorded in `labels/locks.json`.

## 12. Problems we hit and how we fixed them

| Problem | Cause | Fix | Why it matters |
|---|---|---|---|
| Collector crashed | Windows console can't print Devanagari search queries | Run with UTF-8 output | — |
| Hinglish and emoji garbled in Excel | CSV saved without a UTF-8 marker | Sheets now written as UTF-8 with BOM | Wrong text = wrong labels |
| **Row 474's ID became `7.83E+11`** | ID `782617386685` is all digits, so Excel turned it into a number | Rebuilt the ID from the comment text; checked before every commit | That comment would have silently disappeared when joining labels to scores |
| Romanized Malayalam etc. | Search returned South Indian troll videos | Removed those videos | The tagger would have produced invalid CMI |
| **Swear counter was badly wrong** | See below | Rewrote the matcher | IV-3 was invalid |

### The swear-counter fix, in detail

The first version reported about 80 comments with swears. Almost all were false hits:

- **English words folded into Hindi swears:** *cut* → *chut*, *count* → *cunt*, *both* → *bhoot*
- **Spelling folding plus "one letter off" matching:** *jyada* → *jihadi*, *dharam* → *haraam*, *patle* → *pathe*, *chhod* ("leave") → *chod*
- **The list itself contains everyday words:** *pasine* (sweat), *lassan* (garlic), *haathi* (elephant), *jamai* (son-in-law), *kali* (black), *sadi*, *toota*

Fixes, each documented in `src/profanity.py`:
1. **English words are only ever matched exactly**, never spelling-folded.
2. **The "one letter off" fuzzy matching was removed.** It gave 7 false hits for every real one. Real variants (*chtiya, bhenchod, behenchod, bsdk*…) are now explicit aliases.
3. **`NOT_SWEAR`:** list entries that are ordinary vocabulary are excluded. IV-3 measures *taboo words*; whether an ordinary word was used as an insult is for the human label to decide.
4. **Colour terms (*kali/kala*) are excluded.** They're the core vocabulary of skin-tone shaming, so counting them as "swearing" would make IV-3 partly a copy of the DV. *kaalu* (a slur-only nickname) is kept.
5. **English swears added** (*fuck, shit, bitch*…). The original list had none.
6. Fixed a bug where *bkl/bsdk* pointed to a word that wasn't in the list.

**After the fix:** 16 real swears in 14 comments: *shit ×8, bc ×2, chtiya ×2, wtf, fuck, asshole, mc*. Regression tests in the module check both directions: *cut, jyada, chhod* score 0, and *bhenchod, chtiya, bc* score > 0.

## 13. Results (L0, comment-only labels)

### 13.1 Labels

| | 0 Neither | 1 Profane | 2 Other aggr. | 3 Body-shaming | X |
|---|---|---|---|---|---|
| n | 604 | 19 | 48 | 49 | 66 |
| % | 76.8 | 2.4 | 6.1 | **6.2** (4.7–8.1) | **8.4** (6.7–10.5) |

### 13.2 H1: does a comment's own code-mixing predict its label? **No.**

| | OR per +10 CMI | 95% CI | p |
|---|---|---|---|
| Body-shaming, within video | 1.11 | 0.83–1.47 | 0.49 |
| Any aggression, within video | 1.08 | 0.88–1.33 | 0.46 |
| Switch points → body-shaming, within video | 1.02 | 0.95–1.10 | 0.59 |

Median CMI barely differs across labels (Kruskal–Wallis p = 0.96), and neither does matrix language (p = 0.29). Hostile comments *look* more code-mixed (73% vs 57%) only because they're about 3× longer.

### 13.3 H2: does the *parent's* code-mixing predict the reply? **Inconclusive, and the raw trend is a confound.**

| Parent CMI | low | mid | high |
|---|---|---|---|
| % reply aggression | 17.2 | 15.3 | 7.2 |

This looks like "code-mixed parents get calmer replies". But:

| Reply aggression, parent code-mixed (yes/no) | OR | p |
|---|---|---|
| Pooled across videos | **0.37** | 0.043 |
| **Within video** | **2.22** | 0.22 |

**The sign flips.** That's Simpson's paradox: comment sections that are more English are also calmer, for reasons unrelated to code-mixing. Within a video, all estimates lean the *other* way (per +10 parent CMI, aggression OR 1.47, p = 0.08), but none is significant.

### 13.4 H3: swearing. **Too sparse to test.**

Only 14 of 786 comments (1.8%) contain a swear. Swearing does raise the odds of aggression (Fisher OR 9.2, p < 0.001), but 6 of the 14 swearing comments were **not** aggressive. That supports *profanity ≠ aggression*, but the sample is tiny.

### 13.5 Secondary findings, which are the solid ones

1. **Accommodation:** reply CMI tracks parent CMI (ρ = 0.12, p = 0.003). People partly match the language mix of the comment they're replying to.
2. **Context, not code-mixing, makes comments hard to judge:** replies are **4.1×** more likely to be X (p = 0.002), and CMI has no effect (p = 0.23). This is the motivation for L1.
3. **Keyword search misses body-shaming:** **31%** of body-shaming comments contain no appearance word (*pehle better thi*), and only 17% of appearance-word comments are body-shaming. Filtering by keyword would lose a third of the phenomenon and mostly catch harmless comments.
4. **Video topic drives prevalence:** shaming and trolling videos had about 15% body-shaming; fitness-transformation videos had 0–2%. 14 of 29 videos had none.

---

# Part C — Presenting it

## 14. The story in 60 seconds

> We asked whether the way Hinglish speakers mix Hindi and English predicts body-shaming in YouTube comments. We built a language tagger that's 97% accurate on real Hinglish and confirmed that our code-mixing measure tracks real differences in mixing (r = 0.98). We collected 15,000 threaded comments, sampled 786, and hand-labelled them with a five-way scheme separating body-shaming from other aggression and from harmless swearing. **Code-mixing does not predict body-shaming.** An apparent effect, where code-mixed parents get calmer replies, is a between-video confound that reverses once we compare within videos. What *does* matter is context: replies are four times harder to judge than top-level comments, and a third of body-shaming uses no appearance words at all. That points the next phase at context, which our second labelling pass tests directly.

## 15. Slide-by-slide outline (about 12 slides)

| # | Slide | Content | Visual |
|---|---|---|---|
| 1 | Title | Project, team, date | — |
| 2 | Problem | Body-shaming in Hinglish comments; why code-mixing might matter | 2–3 real example comments, labelled |
| 3 | Research questions | H1 own CMI, H2 parent CMI, H3 swearing; context (L0 vs L1) | — |
| 4 | Measuring code-mixing | CMI formula, the worked *bhai ye body bilkul fit hai* example, switch points | Token-by-token colour strip |
| 5 | The tagger | 3 stages; the dictionary-overlap insight (+7.4 pts) | Bar: 87.0% → 97.2% |
| 6 | Validation | MF1 vs MF6 CMI 5 vs 40, r = 0.976; leakage 0.988 vs 0.655 | Two bars |
| 7 | Data | YouTube threads, privacy hashing, 46 candidate videos → 29, 30 per video, 786 comments | Funnel: 15,488 → 786 |
| 8 | Labelling | 5-way scheme, X as data, L0/L1 design, locking | Label table |
| 9 | Label results | Distribution with CIs | Bar chart |
| 10 | **Main result** | H1 null; H2 sign flip, pooled vs within video | **Pooled OR 0.37 vs within OR 2.22, side by side** |
| 11 | What does matter | Replies 4× more X; 31% of body-shaming without appearance words; accommodation | 3 stat tiles |
| 12 | Limits & next | Power, single annotator, blinding, L1, scale-up | — |

Slide 10 is the one to rehearse. Explaining *why* the pooled and within-video numbers disagree shows you understand the method.

## 16. Questions the professor will ask, with answers

**"Isn't a null result a failure?"**
No. It's a well-measured null: the instrument is validated, the labels are blind to the scores, and the within-video model removes the obvious confound. A *wrong positive* (reporting the pooled OR 0.37) would have been the failure.

**"How do you know your CMI is measuring code-mixing?"**
Construct validation on HateCheckHIn, where the conditions are designed to differ: all-Hindi (MF1) mean CMI 5.0 vs mixed (MF6) 39.8, r = 0.976. That ground truth comes from how the test suite was designed, not from our tagger.

**"Your tagger is 97% accurate, but on what?"**
215 tokens, hand-tagged by one person. That's a weakness: it needs three annotators and Krippendorff's α. The figure is honest but small.

**"Why should we trust one person's labels?"**
A written guideline with decision rules made *before* labelling, random order, a locked file with a hash, and no scores on screen. Pending: test–retest κ and three-annotator α on 300 comments. Rani et al. got α = 0.47; that's the bar.

**"You saw the results before L1."**
Yes, and it's disclosed. L1 is therefore not blind to the IVs. The L0 results are unaffected because L0 was locked first.

**"Why not just train a model?"**
That's Study 2. A model's labels can't be the DV: if the model is worse on code-mixed text, its errors would *create* a CMI→label relationship. We'll run Mica-v0.5-4B against the human labels and check specifically for that degradation.

**"Why exclude kali from swear words? Isn't it offensive?"**
It's offensive *as body-shaming*, which is the DV. Counting it in IV-3 as well would make the predictor partly a copy of the outcome.

**"Only 49 body-shaming cases. Is that enough?"**
No, not for H2. The interval can't rule out a moderate effect, so we call H2 *inconclusive*. The 15,488-comment pool is already collected; the fix is to label more from the high-yield videos.

**"Why did you drop videos?"**
South Indian-language videos: the tagger can't measure them. The Pakistani video: Roman Urdu ≠ the target population. Influencer scandals: off-topic. Each decision is recorded.

**"What's the 30-per-video cap for?"**
So that no single audience dominates. Before the cap, three videos supplied half the data.

## 17. Limitations — say them before you're asked

1. **Power:** 49 body-shaming comments. H2 is inconclusive, not negative.
2. **Single annotator:** no reliability figure yet.
3. **Blinding broken for L1:** IVs were seen before the context pass.
4. **Gold LID sample:** 215 tokens, one annotator.
5. **Sample:** keyword-searched videos and 30 per video; not representative of Hinglish YouTube.
6. **Lexicon coverage:** the Hindi lexicon (2,683 forms) came from a narrow template set and needs expanding from real comments.
7. **IV-3 has almost no variance** in this data (1.8% contain swears).
8. **Mica's model card doesn't mention Hindi or Hinglish.**

## 18. What comes next

1. **L1 pass:** label the same 786 with the parent visible, then lock. Key result: how many X labels resolve, and which labels change.
2. **Reliability:** 10% blind self re-label (Cohen's κ); Adhiraj and Arpita label 300 on a *blank* sheet (Krippendorff's α).
3. **Scale up:** label more comments from shaming/trolling videos to reach 150+ body-shaming cases, then re-test H2 within video.
4. **Study 2:** Mica on Kaggle (GPU), L0 vs L1 prompts, plus the degradation check (`mica_classify.py degrade`).
5. **Re-tag the LID gold sample** with three people.

## 19. Glossary

| Term | Meaning |
|---|---|
| **Hinglish** | Hindi–English mixing, here in Roman script |
| **Code-mixing** | Two languages in one utterance |
| **Token** | One word or symbol after splitting the text |
| **LID** | Language identification: tagging each token hi / en / univ / ne |
| **CMI** | Code-Mixing Index, 0 (monolingual) to 50 (balanced) |
| **Switch point** | A change of language between adjacent tokens |
| **Matrix language** | The dominant language providing the grammatical frame |
| **IV / DV** | Independent variable (predictor) / dependent variable (outcome) |
| **L0 / L1** | Labelling pass without / with parent context |
| **Lock** | Freezing a label file and recording its SHA-256 fingerprint |
| **Blinding** | Hiding information that could bias the annotator |
| **OR** | Odds ratio; 1 = no effect |
| **CI** | Confidence interval: the range the true value plausibly lies in |
| **p-value** | How surprising the data would be if there were no effect; < 0.05 is the convention |
| **Clustered SE** | Standard errors that account for comments on one video being related |
| **Within-video / conditional logit** | A model that compares only comments under the same video |
| **Simpson's paradox** | A trend in pooled data that reverses within groups |
| **Leakage** | Test data resembling training data, inflating scores |
| **κ / α** | Agreement statistics: Cohen's kappa (2 raters), Krippendorff's alpha (any number) |
| **Accommodation** | Speakers adjusting their language toward their interlocutor |
| **Implicature (Grice)** | Meaning implied beyond what's literally said; why sarcasm is labelled by intent |

## 20. Reproducing everything

```bash
pip install pandas numpy scikit-learn joblib indic-transliteration \
            english-words google-api-python-client requests scipy statsmodels
python src/build_lexicon.py          # lexicons
python src/char_model.py             # character LID model
python src/eval_gold.py              # -> 0.972
python src/exp_cmi_structure.py      # -> r = 0.976, separation 0.954
python src/profanity.py              # swear-matcher self-test
python src/analyze_l0.py             # -> out/RESULTS_L0.md
```

`data/comments.csv` isn't on GitHub (it holds other people's comments). Ask Raman for it, or re-collect with `src/collect_youtube.py` using your own API key. Note that a re-collection will produce different comments.

| File | What it is |
|---|---|
| `CLAUDE.md` | Project rules: read the 7 rules |
| `ANNOTATION_GUIDELINE.md` | Label scheme and decision rules |
| `RUNBOOK.md` | Step-by-step procedure |
| `RESULTS.md` | Pilot 1 write-up |
| `out/RESULTS_L0.md` | **Current results** |
| `labels/label_L0.csv` + `locks.json` | Locked L0 labels |
| `src/analyze_l0.py` | The analysis that produces the results |
