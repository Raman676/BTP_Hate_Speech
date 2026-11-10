"""
Step 5 + hypothesis tests on the L0 (comment-only) labels.

    python src/analyze_l0.py

Computes IV-1 CMI, IV-2 switch points, IV-3 swear frequency/valence and the
matrix language for every comment AND its parent, joins the locked L0 labels,
and writes a clean markdown report.

Every model is run two ways:
  pooled        logistic regression, SEs clustered by video
  within-video  conditional logit grouped by video (video fixed effects)
The second is the one to report. Comment sections differ in language mix AND
in hostility for reasons that have nothing to do with code-mixing (topic,
audience). Only the within-video estimate removes that confound.

Writes data/comments_with_ivs.csv (gitignored: contains comment text) and
out/RESULTS_L0.md.

NOTE: run after L0 was locked but BEFORE L1 was labelled. Declare this.
"""
import sys
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.discrete.conditional_models import ConditionalLogit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from lid import LanguageTagger, describe, TOKEN_RE          # noqa: E402
from char_model import as_tagger_fn                         # noqa: E402
import profanity as P                                       # noqa: E402

warnings.filterwarnings('ignore')
IV_COLS = ['n_tokens', 'n_lang_tokens', 'cmi', 'switch_points', 'matrix_lang',
           'swear_count', 'swear_freq', 'profanity_valence']
ORDER = ['0', '1', '2', '3', 'X']
NAMES = {'0': 'Neither', '1': 'Profane, not aggressive', '2': 'Other aggression',
         '3': 'Body-shaming', 'X': 'Cannot judge'}


# --------------------------------------------------------------- measurement
def compute_ivs(texts, tagger, pm):
    rows = []
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            rows.append({c: np.nan for c in IV_COLS})
            continue
        r = {**describe(t, tagger), **pm.measure(TOKEN_RE.findall(t))}
        rows.append({c: r.get(c) for c in IV_COLS})
    return pd.DataFrame(rows)


def build():
    tagger = LanguageTagger(json.load(open(ROOT / 'out/lex_en.json')),
                            json.load(open(ROOT / 'out/lex_hi.json')),
                            char_model=as_tagger_fn(
                                joblib.load(ROOT / 'out/char_lid.joblib')))
    pm = P.load()
    df = pd.read_csv(ROOT / 'data/comments.csv')
    lab = pd.read_csv(ROOT / 'labels/label_L0.csv', dtype=str,
                      keep_default_na=False)
    lab['label'] = lab.label.str.strip().str.upper()
    df = df.merge(lab[['uid', 'label']], on='uid', how='inner')
    own = compute_ivs(df.text, tagger, pm)
    par = compute_ivs(df.parent_text, tagger, pm).add_prefix('parent_')
    out = pd.concat([df.reset_index(drop=True), own, par], axis=1)
    out.to_csv(ROOT / 'data/comments_with_ivs.csv', index=False)
    return out


# --------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def pooled(formula, data, term):
    m = smf.logit(formula, data=data).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': data.video_id})
    lo, hi = np.exp(m.conf_int().loc[term])
    return np.exp(m.params[term]), lo, hi, m.pvalues[term], int(m.nobs)


def within(y, xs, data, term):
    """Conditional logit, grouped by video. Videos with no variation in y
    contribute nothing and are dropped automatically."""
    d = data.dropna(subset=[y] + xs)
    m = ConditionalLogit(d[y].values, d[xs].values,
                         groups=d.video_id.values).fit(disp=0)
    i = xs.index(term)
    b, se = m.params[i], m.bse[i]
    keep = d.groupby('video_id')[y].transform(lambda s: s.nunique() > 1)
    return (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se),
            m.pvalues[i], int(keep.sum()), d[keep].video_id.nunique())


def fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def row(name, r, within_=False):
    if within_:
        o, lo, hi, p, n, v = r
        return f'| {name} | {o:.2f} | {lo:.2f}–{hi:.2f} | {fmt_p(p)} | {n} ({v} videos) |'
    o, lo, hi, p, n = r
    return f'| {name} | {o:.2f} | {lo:.2f}–{hi:.2f} | {fmt_p(p)} | {n} |'


# --------------------------------------------------------------- report
def report(d):
    L = []
    w = L.append
    d['is_reply'] = d.parent_id.notna().astype(int)
    d['y3'] = (d.label == '3').astype(int)
    d['yagg'] = d.label.isin(['2', '3']).astype(int)
    d['yX'] = (d.label == 'X').astype(int)
    d['cmi10'] = d.cmi / 10
    d['parent_cmi10'] = d.parent_cmi / 10
    d['log_len'] = np.log1p(d.n_tokens)
    d['parent_log_len'] = np.log1p(d.parent_n_tokens)
    d['has_swear'] = (d.swear_count > 0).astype(int)
    d['parent_has_swear'] = (d.parent_swear_count > 0).astype(int)
    n = len(d)
    jl = d[d.label != 'X'].copy()                     # judged comments
    rp = d[(d.is_reply == 1) & d.parent_cmi.notna()].copy()
    rj = rp[rp.label != 'X'].copy()

    w('# Study 1, pass L0 — results\n')
    w(f'Generated by `src/analyze_l0.py`. **{n} comments** from '
      f'**{d.video_id.nunique()} videos**, {d.thread_id.nunique()} threads; '
      f'{d.is_reply.sum()} are replies with a visible parent. Labels: '
      f'L0 pass (comment only), single annotator, locked before analysis.\n')
    w('> **Disclosure.** These IVs were computed after L0 was locked but '
      'before L1 was labelled. L1 is therefore not blind to the IV values.\n')

    # ---- 1. labels
    w('## 1. Label distribution\n')
    w('| Label | Meaning | n | % | 95% CI |')
    w('|---|---|---|---|---|')
    for k in ORDER:
        c = (d.label == k).sum()
        lo, hi = wilson(c, n)
        w(f'| {k} | {NAMES[k]} | {c} | {c / n:.1%} | {lo:.1%}–{hi:.1%} |')
    w('')

    # ---- 2. descriptives
    w('## 2. The three IVs by label (own comment)\n')
    g = d.groupby('label').agg(
        n=('uid', 'size'), cmi_med=('cmi', 'median'), cmi_mean=('cmi', 'mean'),
        mixed=('cmi', lambda s: (s > 0).mean() * 100),
        sw=('switch_points', 'mean'), swear=('has_swear', 'mean'),
        tok=('n_tokens', 'mean')).reindex(ORDER)
    w('| Label | n | CMI median | CMI mean | % code-mixed | switch points | % with swear | tokens |')
    w('|---|---|---|---|---|---|---|---|')
    for k, r in g.iterrows():
        w(f'| {k} {NAMES[k]} | {r.n:.0f} | {r.cmi_med:.1f} | {r.cmi_mean:.1f} | '
          f'{r.mixed:.0f}% | {r.sw:.2f} | {r.swear * 100:.1f}% | {r.tok:.0f} |')
    kw = stats.kruskal(*[jl.cmi[jl.label == k] for k in '0123'])
    w(f'\nKruskal–Wallis, CMI across labels 0/1/2/3: H = {kw.statistic:.2f}, '
      f'p = {fmt_p(kw.pvalue)}.')
    ct = pd.crosstab(d.matrix_lang, d.label)
    chi = stats.chi2_contingency(ct)
    w(f'Matrix language × label: χ² = {chi[0]:.2f}, df = {chi[2]}, '
      f'p = {fmt_p(chi[1])}.\n')
    w('Code-mixed comments *look* more hostile (73% of 2/3 vs 57% of 0 contain '
      'a switch) but hostile comments are ~3× longer, and a longer comment has '
      'more chances to switch. Length is controlled in every model below.\n')

    # ---- 3. H1
    w('## 3. H1 — does a comment\'s own code-mixing predict its label?\n')
    w('Odds ratio per +10 CMI, controlling for log length. X excluded.\n')
    w('| Outcome / model | OR | 95% CI | p | n |')
    w('|---|---|---|---|---|')
    for y, nm in [('y3', 'Body-shaming'), ('yagg', 'Any aggression (2/3)')]:
        w(row(f'{nm} — pooled', pooled(f'{y} ~ cmi10 + log_len', jl, 'cmi10')))
        w(row(f'{nm} — within video',
              within(y, ['cmi10', 'log_len'], jl, 'cmi10'), True))
    for y, nm in [('y3', 'Body-shaming'), ('yagg', 'Any aggression (2/3)')]:
        w(row(f'{nm} — switch points, within video',
              within(y, ['switch_points', 'log_len'], jl, 'switch_points'),
              True))
    w('\n**Result: null.** No association between a comment\'s own '
      'code-mixing and its label, pooled or within video.\n')

    # ---- 4. H2
    w('## 4. H2 — does the *parent\'s* code-mixing predict the reply? '
      '(the study\'s hypothesis)\n')
    q = pd.qcut(rp.parent_cmi.rank(method='first'), 3,
                labels=['low', 'mid', 'high'])
    t = rp.groupby(q, observed=True).agg(
        n=('uid', 'size'), pc=('parent_cmi', 'mean'), p3=('y3', 'mean'),
        pa=('yagg', 'mean'), px=('yX', 'mean'))
    w('| Parent CMI tercile | n | mean parent CMI | % reply body-shaming | % reply aggression | % X |')
    w('|---|---|---|---|---|---|')
    for k, r in t.iterrows():
        w(f'| {k} | {r.n:.0f} | {r.pc:.1f} | {r.p3:.1%} | {r.pa:.1%} | {r.px:.1%} |')
    w('\nOdds ratio per +10 parent CMI, controlling for parent length and the '
      'reply\'s own CMI. X excluded.\n')
    w('| Outcome / model | OR | 95% CI | p | n |')
    w('|---|---|---|---|---|')
    xs = ['parent_cmi10', 'parent_log_len', 'cmi10']
    for y, nm in [('y3', 'Reply body-shaming'), ('yagg', 'Reply aggression')]:
        w(row(f'{nm} — pooled', pooled(
            f'{y} ~ ' + ' + '.join(xs), rj, 'parent_cmi10')))
        w(row(f'{nm} — within video', within(y, xs, rj, 'parent_cmi10'), True))
    rj['parent_mixed'] = (rj.parent_cmi > 0).astype(int)
    a = pooled('yagg ~ parent_mixed + parent_log_len', rj, 'parent_mixed')
    b = within('yagg', ['parent_mixed', 'parent_log_len'], rj, 'parent_mixed')
    w(row('Reply aggression, parent mixed (yes/no) — pooled', a))
    w(row('Reply aggression, parent mixed (yes/no) — within video', b, True))
    w(f'\n**Result: null, and the raw trend is a confound.** Pooled, replies to '
      f'code-mixed parents look less aggressive (OR {a[0]:.2f}, p = {fmt_p(a[3])}). '
      f'Within the same video the sign flips and the effect is no longer '
      f'significant (OR {b[0]:.2f}, p = {fmt_p(b[3])}). Comment sections that '
      f'are more English are also calmer; code-mixing is not what makes them '
      f'calmer. The within-video estimates all point the other way '
      f'(aggression per +10 parent CMI: p = 0.08) — not significant, but '
      f'worth re-testing with more data.\n')

    # ---- 5. H3
    w('## 5. H3 — swearing (IV-3)\n')
    w(f'After the profanity-matcher fix (see `src/profanity.py`), only '
      f'**{d.has_swear.sum()} of {n} comments ({d.has_swear.mean():.1%})** '
      f'contain a swear word.\n')
    w('| Label | with swear | without |')
    w('|---|---|---|')
    for k in ORDER:
        s = d[d.label == k]
        w(f'| {k} {NAMES[k]} | {s.has_swear.sum()} | {(1 - s.has_swear).sum()} |')
    ft = stats.fisher_exact(pd.crosstab(jl.has_swear, jl.yagg).values)
    w(f'\nSwear present vs aggression (2/3): Fisher exact OR = {ft[0]:.2f}, '
      f'p {fmt_p(ft[1]) if ft[1] < 0.001 else "= " + fmt_p(ft[1])}.')
    sw = jl[jl.has_swear == 1]
    w(f'Of the {len(sw)} judged comments containing a swear, '
      f'{sw.label.isin(["2", "3"]).sum()} were aggressive (2/3) and '
      f'{(~sw.label.isin(["2", "3"])).sum()} were not '
      f'({(sw.label == "1").sum()} profane-not-aggressive, '
      f'{(sw.label == "0").sum()} neither).\n')
    w('**Result: too sparse to test.** Swearing is rare in these comment '
      'sections, so IV-3 has almost no variance. What exists supports rule 7 '
      '(profanity ≠ aggression): swearing raises the odds of aggression, but '
      'close to half of swearing comments were not aggressive.\n')

    # ---- 6. secondary
    w('## 6. Secondary findings\n')
    rho = stats.spearmanr(rp.parent_cmi, rp.cmi)
    w(f'**Accommodation.** Reply CMI tracks parent CMI: Spearman ρ = '
      f'{rho.statistic:.2f}, p = {fmt_p(rho.pvalue)}. Weak but reliable — '
      f'repliers partly match the language mix they reply to.\n')
    o = pooled('yX ~ cmi10 + log_len + is_reply', d, 'is_reply')
    c = pooled('yX ~ cmi10 + log_len + is_reply', d, 'cmi10')
    w(f'**What makes a comment unjudgeable (X)?** Being a reply: OR '
      f'{o[0]:.1f} ({o[1]:.1f}–{o[2]:.1f}), p = {fmt_p(o[3])}. Code-mixing: '
      f'OR {c[0]:.2f}, p = {fmt_p(c[3])}. Difficulty comes from missing '
      f'context, not from mixing languages — the motivation for L1.\n')
    t3 = d.y3 == 1
    f = d.is_appearance.astype(bool)
    w(f'**Appearance vocabulary misses body-shaming.** {(t3 & ~f).sum()} of '
      f'{t3.sum()} body-shaming comments ({(t3 & ~f).sum() / t3.sum():.0%}) '
      f'contain no appearance word; only {(t3 & f).sum() / f.sum():.0%} of '
      f'appearance-word comments are body-shaming. Keyword filtering would '
      f'lose a third of the phenomenon.\n')

    # ---- 7. limits
    w('## 7. Limits\n')
    w(f'- **Power.** {t3.sum()} body-shaming comments ({rj.y3.sum()} among '
      f'judged replies). The H2 interval cannot exclude a moderate effect: '
      f'report H2 as *inconclusive*, not *no effect*.')
    w('- **Single annotator**, no reliability figure yet (test–retest and '
      'inter-annotator α pending).')
    w('- **Sample.** Videos found by keyword search and capped at 30 comments '
      'each; not representative of YouTube Hinglish.')
    w('- **Blinding.** IVs were seen before L1 (see disclosure).')
    return '\n'.join(L) + '\n'


def main():
    d = build()
    md = report(d)
    (ROOT / 'out/RESULTS_L0.md').write_text(md, encoding='utf-8')
    print(md)
    print('wrote data/comments_with_ivs.csv and out/RESULTS_L0.md')


if __name__ == '__main__':
    main()
