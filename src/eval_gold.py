"""Evaluate the token-level tagger against the hand-tagged gold sample."""
import json
import sys
from pathlib import Path
from collections import Counter
import joblib
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lid import LanguageTagger  # noqa: E402
from char_model import as_tagger_fn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT, DATA = ROOT / 'out', ROOT / 'data'


def load_gold():
    sents, cur, sid = [], [], None
    for line in (DATA / 'gold_lid.tsv').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        s, tok, tag = line.split('\t')
        if s != sid:
            if cur:
                sents.append((sid, cur))
            sid, cur = s, []
        cur.append((tok, tag))
    if cur:
        sents.append((sid, cur))
    return sents


def build_tagger():
    en = json.loads((OUT / 'lex_en.json').read_text())
    hi = json.loads((OUT / 'lex_hi.json').read_text())
    char = joblib.load(OUT / 'char_lid.joblib')
    return LanguageTagger(en, hi, char_model=as_tagger_fn(char))


def main():
    gold = load_gold()
    tagger = build_tagger()
    y_true, y_pred, errors = [], [], []

    for sid, toks in gold:
        text = ' '.join(t for t, _ in toks)
        pred = tagger.tag_tokens(text)
        pred_map = [p for _, p in pred]
        if len(pred_map) != len(toks):
            # re-align by tagging each gold token individually
            pred_map = [tagger.tag_token(t) for t, _ in toks]
            pred_map = ['hi' if p == 'amb' else p for p in pred_map]
        for (tok, g), p in zip(toks, pred_map):
            y_true.append(g)
            y_pred.append(p)
            if g != p:
                errors.append((sid, tok, g, p))

    print(f'Gold sentences: {len(gold)}   gold tokens: {len(y_true)}')
    print(f'Token accuracy: {sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true):.3f}\n')
    labels = sorted(set(y_true) | set(y_pred))
    print(classification_report(y_true, y_pred, labels=labels,
                                zero_division=0, digits=3))
    print('Confusion (rows=gold, cols=pred):', labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))

    print(f'\nErrors ({len(errors)}):')
    for sid, tok, g, p in errors:
        print(f'  {sid}  {tok:<18} gold={g:<5} pred={p}')
    print('\nMost-confused tokens:',
          Counter(t for _, t, _, _ in errors).most_common(10))


if __name__ == '__main__':
    main()
