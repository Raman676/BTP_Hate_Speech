# Project context for Claude Code

## What this project is

An academic study of **body-shaming hate speech in Hindi-English code-mixed
(Hinglish) YouTube comments, Roman script only**.

It is a **linguistics study**, not an engineering project. The deliverable is
a defensible research finding, not a high F1 score. Methodological correctness
beats performance every time. If a choice would raise a metric but weaken the
inference, take the weaker metric.

## The research design

Two separate studies. Do not merge them.

**Study 1 — the research question.** Do code-mixing features predict
*human-judged* body-shaming, and does that judgment change when the annotator
can see the parent comment?
- DV: **human labels only**, hand-labelled by PJ
- Context is a *manipulated variable*: the same comments are labelled twice,
  once without context (L0), once with the parent visible (L1)
- The label changes between passes are the finding

**Study 2 — the engineering question.** Can a model reproduce those labels,
and does context help it?
- Evaluated against Study 1's human labels
- Mica-v0.5-4B (`sky7350/Mica-v0.5-4B`) plus simpler baselines

## Variables

| | Variable | Where |
|---|---|---|
| IV-1 | Code-Mixing Index (CMI), 0–100 | `src/lid.py` |
| IV-2 | Switch points | `src/lid.py` |
| IV-3 | Swear frequency + valence | `src/profanity.py` |
| DV | Body-shaming label 3/2/1/0/X | hand-labelled |

Labels: **3** body-shaming · **2** other aggression · **1** profane not
aggressive · **0** neither · **X** cannot judge without context.

## Rules that must not be broken

1. **Never use model-generated labels as the DV for a hypothesis test.**
   Mica may be weaker on code-mixed text. If it is, an IV→DV relationship
   drawn from its labels measures model error, not language. Run
   `mica_classify.cmi_degradation_check()` before reporting anything.

2. **Never use random train/test splits.** Pilot 1 scored 0.988 macro-F1 with
   random folds and 0.655 with template-grouped folds. A third of the
   performance was memorisation. Group by thread or template, always.

3. **Never show IV values during labelling.** `make_sheets.py` strips them.
   Do not add them back, do not print them, do not sort by them.

4. **L0 must be locked before L1 is generated.** The tooling enforces this.
   Do not bypass it.

5. **Never store YouTube usernames or channel IDs.** They are salted-hashed
   at collection. Do not add raw identifiers.

6. **Roman script is an orthography, not a language.**
   `bhai ye to bilkul kharab lag rahi hai` is monolingual Hindi written in
   Roman letters. The script filter removes Devanagari, not Hindi.

7. **Profanity ≠ aggression.** `kya mast h ye bhenchod` is praise. This is the
   project's central finding and the reason for the separate valence feature.

## Current state

- Measurement pipeline **validated**: token-level language tagging 97.2% on a
  hand-tagged gold sample; CMI construct validation r = 0.976.
- Pilot 1 classifier: 0.655 macro-F1 (thread-grouped), with the leakage
  finding documented in `RESULTS.md`.
- **No comments collected yet.** That is the only blocker.

## Where to start

`RUNBOOK.md` has the full sequence. Step 1 is:

```bash
python src/collect_youtube.py --search --per-query 3 --max-threads 25
```

Small run first, skim `data/videos_found.csv`, drop anything off-topic, then
scale up.

## Known weaknesses — do not paper over these

- Gold language-tag sample is 215 tokens, single annotator. Needs re-tagging
  by three people and a Krippendorff's α.
- Single primary annotator for the labels. Mitigated by the blinding protocol
  and test–retest, not eliminated.
- Videos found by keyword search, so the sample is not representative.
- Mica's model card does not mention Hindi or Hinglish.

## Style

PJ is an AI engineer — be direct and technical, skip the hand-holding. But he
is doing this for a linguistics professor, so when a choice has a
methodological consequence, say so plainly rather than just implementing it.
