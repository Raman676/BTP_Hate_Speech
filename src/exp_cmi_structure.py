"""
Experiment A - construct validation of the CMI measure.

HateCheckHIn's multilingual block is built from designed code-mixing
functionalities. Two of them are pure Roman script:

  MF1 : fully romanized Hindi            -> predicted LOW  CMI, matrix = Hindi
  MF6 : English frame + romanized Hindi  -> predicted HIGH CMI, matrix = mixed

The ground truth here comes from the test suite's design, not from our tagger,
so this is an independent check that the pipeline measures what it claims to.
Pre-registered prediction: CMI(MF6) > CMI(MF1).
"""
import sys
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lid import LanguageTagger, describe  # noqa: E402
from char_model import as_tagger_fn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT, DATA = ROOT / 'out', ROOT / 'data'
DEV = re.compile(r'[ऀ-ॿ]')


def mannwhitney_u(a, b):
    """Two-sided Mann-Whitney U with normal approximation + rank-biserial r."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2
    tie = sum(c ** 3 - c for c in cnt)
    n = n1 + n2
    sd = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    z = (u - mu) / sd
    import math
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / np.sqrt(2)))) if sd > 0 else 1.0
    rb = 2 * u1 / (n1 * n2) - 1   # rank-biserial; positive => b > a
    return u, z, p, rb


def main():
    en = json.loads((OUT / 'lex_en.json').read_text())
    hi = json.loads((OUT / 'lex_hi.json').read_text())
    char = joblib.load(OUT / 'char_lid.joblib')
    tagger = LanguageTagger(en, hi, char_model=as_tagger_fn(char))

    m = pd.read_csv(DATA / 'multilingual_functionalities.csv')
    rom = m[~m.test_case.astype(str).str.contains(DEV)].copy()

    rows = [describe(s, tagger) for s in rom.test_case]
    feat = pd.DataFrame(rows, index=rom.index)
    df = pd.concat([rom.reset_index(drop=True), feat.reset_index(drop=True)],
                   axis=1)
    df.to_csv(OUT / 'cmi_multilingual.csv', index=False)

    print('=' * 68)
    print('EXPERIMENT A - does CMI recover the designed code-mixing structure?')
    print('=' * 68)
    print(f'\nRoman-script rows analysed: {len(df)}')

    summ = df.groupby('functionality').agg(
        n=('cmi', 'size'),
        cmi_mean=('cmi', 'mean'),
        cmi_sd=('cmi', 'std'),
        cmi_median=('cmi', 'median'),
        switch_mean=('switch_points', 'mean'),
        tokens_mean=('n_lang_tokens', 'mean'),
    ).round(2)
    print('\n', summ.to_string())

    print('\nMatrix language distribution (row %):')
    mx = pd.crosstab(df.functionality, df.matrix_lang, normalize='index')
    print((mx * 100).round(1).to_string())

    a = df.loc[df.functionality == 'MF1', 'cmi'].values
    b = df.loc[df.functionality == 'MF6', 'cmi'].values
    u, z, p, r = mannwhitney_u(a, b)
    print(f'\nPrediction: CMI(MF6) > CMI(MF1)')
    print(f'  MF1 (romanized Hindi only) mean CMI = {a.mean():.2f}  (n={len(a)})')
    print(f'  MF6 (English + Hindi)      mean CMI = {b.mean():.2f}  (n={len(b)})')
    print(f'  Mann-Whitney U = {u:.0f}, z = {z:.2f}, p = {p:.3g}')
    print(f'  rank-biserial effect size r = {r:.3f}')
    print(f'  RESULT: {"CONFIRMED" if b.mean() > a.mean() and p < .05 else "NOT CONFIRMED"}')

    # How cleanly does CMI alone separate the two designed conditions?
    best_acc, best_t = 0, None
    for t in np.arange(0, 60, 0.5):
        acc = ((a < t).sum() + (b >= t).sum()) / (len(a) + len(b))
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f'\n  Single-threshold separation of MF1 vs MF6 using CMI alone:')
    print(f'    best threshold CMI = {best_t}, accuracy = {best_acc:.3f}')


if __name__ == '__main__':
    main()
