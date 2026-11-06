# Annotation Guideline — Body-Shaming in Hindi-English Code-Mixed Comments

**Version 2.0 · 6 November 2026**
Read fully before labelling a single comment. The protocol in §5 is not optional — it is what makes single-annotator labels defensible.

**Changed in v2.0:** the scheme now separates **body-shaming aggression** from **other aggression**. v1.0 collapsed them, which would have made the study about aggression in general rather than about body-shaming.

---

## 1. What you are labelling

The **pragmatic force** of a comment: what it *does* to its addressee, not what words it contains.

This is the project's theoretical core. Kumar et al. (2021) found profanity alone does not make a comment offensive. Grice explains why: speakers routinely flout a maxim to generate an implicature, so the literal content of an utterance underdetermines its force.

> **The test question for every comment:** would a competent Hinglish speaker, reading this, feel it was aimed at hurting someone — and if so, was it aimed at how they *look*?
>
> Not: does it contain a swear word. Not: is it rude.

---

## 2. The label scheme

| Code | Label | Definition |
|---|---|---|
| **3** | **Body-shaming** | Hostility aimed at a person's **appearance** — weight, shape, size, face, skin colour, hair, ageing, perceived surgery or editing. |
| **2** | **Other aggression** | Hostility aimed at a person or group, but **not about appearance** — their character, intelligence, caste, religion, gender, family. |
| **1** | **Profane / charged, not aggressive** | Profanity, crude or strong negative language, **not directed hostilely**. Intensifiers, solidarity markers, venting, self-directed. |
| **0** | **Neither** | Ordinary comment. May be critical or negative about a *thing* without hostility toward a person. |
| **X** | **Cannot judge** | You genuinely cannot determine force from what you can see. |

### Why 3 and 2 are separate

A body-shaming video's comment section carries both kinds of hostility. If you collapse them, your dependent variable is "aggression," and the study is no longer about body-shaming — you would be unable to say whether code-mixing predicts *body-shaming* specifically or hostility generally. Keeping them apart lets you test both, and lets you report whether the code-mixing effect differs between them. That contrast is likely your most interesting result.

### Why 1 exists

Binary hate/not-hate forces `kya mast h ye bhenchod` into "not hate," which is correct but discards the information that it is *profane and positive*. That combination is what breaks lexicon-based systems, and it is what IV-3 measures. Collapsing it loses your result.

### The X code is data

`X` is **not a failure**. Rani et al. could not label hashtag-only posts; both papers you reviewed named missing context as their main error source. The rate of `X` in the comment-only pass, and how far it falls when context is added, is **a headline result of Study 1**. Use it honestly — do not guess to avoid it.

---

## 3. Decision rules

Apply in order. The first rule that applies, decides.

**R1 — Target test.** Is hostility aimed at a person or group at all?
No → **0** or **1** (see R3). Criticism of a *thing* is not aggression: "ye video bakwaas hai" is **0**.

**R2 — Appearance test.** If hostile, is it about how someone **looks**?
Yes → **3**. No → **2**.
Covers weight, shape, size, face, skin tone, hair, teeth, ageing, "plastic surgery", "filter laga ke". Mixed cases where appearance is one of several attacks → **3**, since appearance is the phenomenon under study.

**R3 — Valence test.** Profanity next to positive evaluation (*mast, badhiya, sahi, zabardast*) is an intensifier → **1**.
`kya mast h ye bhenchod` = **1**. Enthusiasm, in-group register. Flouting Manner to signal solidarity.

**R4 — Addressee test.** Profanity plus a second-person marker (*tu, tera, tujhe, you, your*) is usually directed → **2** or **3** by R2.
Profanity with no addressee and no target → **1**.

**R5 — Sarcasm test.** If the literal reading is positive but the obvious intended reading is hostile, label the **intended** reading.
Flouting Quality. "Wah kya figure hai 😂" on a weight-gain video is **3**, not 0.

**R6 — Concern vs mockery.** Appearance comments framed as concern are **3** only if mocking.
"bhai sehat ka dhyan rakho" said warmly is **0**. The same words with 😂😂 or in a pile-on are **3**. If you cannot tell from the comment alone → **X**. Do not guess.

**R7 — Severity is not force.** A severity-10 swear used as an intensifier is still **1**. A mild word used to demean someone's body is **3**. Do not let the profanity lexicon label for you.

**R8 — Third-party body talk.** Appearance hostility about someone *not* in the conversation (a celebrity, the video's subject) is still **3**. The target need not be the addressee.

**R9 — Self-directed.** "main kitna mota hoon yaar" is **1**, not 3. Body-shaming requires a target other than the speaker.

**R10 — When torn between 2 and 3,** ask what the insult is *about*. If removing the appearance content leaves nothing, it is **3**.

---

## 4. Worked examples

| Comment | Label | Why |
|---|---|---|
| `kitni moti ho gayi hai ye` | **3** | Weight, third party (R2, R8) |
| `tera chehra kitna kala hai` | **3** | Skin tone, directed. No profanity needed (R2) |
| `wah kya figure hai 😂` | **3** | Sarcastic appearance mockery (R5) |
| `bhai itna weight gain kyu kiya` | **X** | Concern or mockery? Can't tell alone (R6) |
| `tu bilkul bewakoof hai` | **2** | Hostile, but about intelligence (R2) |
| `lower caste log aise hi hote hain` | **2** | Group-directed, not appearance (R2) |
| `kya mast h ye bhenchod` | **1** | Profane, positive. Intensifier (R3) |
| `yaar mera din kharab tha, bc` | **1** | Venting, no target (R4) |
| `main kitna mota ho gaya hoon` | **1** | Self-directed (R9) |
| `ye video bakwaas hai` | **0** | Criticises a thing (R1) |
| `sehat ka dhyan rakhna bhai ❤️` | **0** | Genuine concern (R6) |
| `pehle better thi` | **X** | Implicit appearance jab, or about her acting? (R6) |
| `#bodypositivity` | **X** | No proposition to judge |
| `moti bhains kahin ki, tujhe sharam nahi aati` | **3** | Appearance + directed + profane (R2, R10) |

Note `pehle better thi` — no appearance vocabulary at all, yet it may be the most common form of body-shaming in these threads. The appearance-term flag in `comments.csv` **will miss these**. Your labels must not.

---

## 5. Protocol — the part that makes solo labelling defensible

You are both the researcher and the only annotator. You know the hypothesis, and the code-mixing is visible while you label. That is **experimenter bias by construction**, and your professor will ask about it. These five rules neutralise it.

### P1 — Never look at an IV before labels are locked
Do not compute, display, or sort by CMI, switch points, or swear frequency until the label file is frozen. If you see CMI 40 next to a comment while deciding, that label is contaminated. The sheet generator strips those columns.

### P2 — Comment-only pass first, for the *entire* set, then lock
L0 must be complete and written to disk before you look at a single parent comment. You cannot un-see context. Reverse or interleaved order destroys the comparison.

Lock = write the file, record its SHA-256, never edit it again. Corrections go in a separate file with reasons.

### P3 — Randomise order, re-randomise between passes
Thread-adjacent comments bias each other. L1 uses a different random order from L0, or it becomes a memory test.

### P4 — Leave a gap between passes
Minimum 48 hours between L0 and L1. Longer is better.

### P5 — Reliability, two ways

**Intra-annotator (test–retest).** Re-label a random 10% of L0 after the gap, blind. Report Cohen's κ between your two passes. If you don't agree with yourself, nobody else will.

**Inter-annotator.** Have Adhiraj and Arpita each label the same 300-comment subset using this guideline. Report Krippendorff's α across all three. Rani et al. reported **α = 0.47** with six annotators and mid-study guideline revisions — that is the bar, and it is beatable because your guideline exists *before* labelling starts, which theirs did not.

> A single-annotator corpus with no reliability figure leaves a hole in the methodology section. Both numbers are cheap and close it.

---

## 6. Labelling mechanics

```bash
python src/make_sheets.py l0 data/comments.csv     # makes labels/label_L0.csv
# label every row 3/2/1/0/X in Excel or Sheets, then:
python src/make_sheets.py lock labels/label_L0.csv
```

The sheet has four columns: `uid, text, label, note`. You fill **label** and **note**. `uid` is an opaque hash carrying no thread or position information, so you cannot infer context during L0.

Use **note** freely — "unclear if sarcastic", "needs thread", "appearance implied not stated". These become your error analysis and the raw material for the qualitative half of the paper.

---

## 7. Language-tagging conventions

These affect the IVs, not the labels, but settle them now and keep them consistent.

- **Established English borrowings keep English**: *body, figure, size, time, video, comment*. English-origin, English-tagged even in a Hindi frame.
- **Community and religion names follow the frame**: *Hindu, Muslim, Dalit* tagged by the language of the surrounding clause.
- **Named entities are `ne` and excluded** from CMI: people, channels, films, brands.
- **Discourse markers are `univ` and excluded**: *lol, omg, haha, hmm*.
- **Emoji, punctuation, digits are `univ`** and excluded.

---

## 8. Limitations to state in the paper

1. Single primary annotator; reliability from test–retest plus a three-way subset, not full double annotation.
2. Annotator is not blind to the hypothesis. Mitigated by P1–P4; not eliminated.
3. Guideline written before labelling and **not revised during it**. If you find it inadequate mid-pass, stop, revise, version it, and re-label from the start. Rani et al. revised mid-study and dropped annotators; that is why their α is 0.47.
4. Videos were found by keyword search, so the sample is not representative of YouTube Hindi comment sections generally.
5. If you used `--appearance-only`, say so: it concentrates labelling effort but destroys the base rate and biases toward explicit vocabulary, missing implicit cases like `pehle better thi`.
