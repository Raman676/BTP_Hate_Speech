"""
Experiment B - baseline hate classifier, and does adding the CMI features help?

Data: HateCheckHIn monolingual block (4,754 Devanagari sentences, hateful /
nonhateful), transliterated to Roman script so it matches the project's
Roman-script-only constraint.

!! READ THIS BEFORE QUOTING THE NUMBER !!
Transliterated text has ONE canonical spelling per word. Real Roman-script
Hinglish has many. So these F1 scores are an UPPER BOUND on what the same
model would score on scraped YouTube comments, not an estimate of it. The
comparison between conditions is the meaningful part, not the absolute value.

Conditions:
  A  text only                (char n-gram TF-IDF)
  B  CMI features only        (cmi, switch_points, matrix_lang, length)
  C  text + CMI features
"""
import sys
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from indic_transliteration.sanscript import transliterate
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold,
                                     cross_val_predict)
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lid import LanguageTagger, describe  # noqa: E402
from char_model import as_tagger_fn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT, DATA = ROOT / 'out', ROOT / 'data'
SEED = 42


def romanize_sentence(s: str) -> str:
    out = []
    for tok in re.findall(r'[ऀ-ॿ]+|[A-Za-z]+|[^\w\s]|\d+', str(s)):
        if re.search(r'[ऀ-ॿ]', tok):
            r = transliterate(tok, 'devanagari', 'itrans').lower()
            r = re.sub(r'[^a-z]', '', r)
            out.append(r or tok)
        else:
            out.append(tok)
    return ' '.join(out)


def build_frame(tagger):
    mono = pd.read_csv(DATA / 'monolingual_functionalities.csv')
    mono['roman'] = mono.test_case.map(romanize_sentence)
    feats = pd.DataFrame([describe(s, tagger) for s in mono.roman])
    df = pd.concat([mono.reset_index(drop=True), feats], axis=1)
    df['y'] = (df.label == 'hateful').astype(int)
    df.to_csv(OUT / 'mono_romanized_features.csv', index=False)
    return df


def add_template_ids(df):
    """Collapse each test case to its generating template.

    HateCheck builds many sentences from one frame, varying only the target
    group. If those land in different CV folds the model memorises the frame
    rather than learning the phenomenon, so the template is the grouping unit.

    The target slot cannot be matched from the `target` column directly (it is
    English, e.g. "Muslim", while the sentence is romanized Devanagari,
    e.g. "musalamaanon"). Instead we find slot tokens empirically: within a
    functionality, a token position that varies across otherwise-identical
    sentences IS the slot. We blank any token that is rare within its
    functionality, which leaves the invariant frame.
    """
    from collections import Counter
    tmpl = []
    for func, grp in df.groupby('functionality', sort=False):
        counts = Counter()
        for s in grp.roman:
            counts.update(set(str(s).lower().split()))
        n = len(grp)
        # frame tokens appear in most sentences of the functionality;
        # slot tokens appear in few.
        frame = {w for w, c in counts.items() if c >= 0.5 * n}
        for idx, s in zip(grp.index, grp.roman):
            skel = ' '.join(w if w in frame else '<T>'
                            for w in str(s).lower().split())
            skel = re.sub(r'(<T> )+', '<T> ', skel)
            tmpl.append((idx, f'{func}||{skel}'))
    out = pd.Series(dict(tmpl))
    return out.reindex(df.index).values


def evaluate(name, pipe, X, y, cv, groups=None):
    pred = cross_val_predict(pipe, X, y, cv=cv, groups=groups, n_jobs=-1)
    macro = f1_score(y, pred, average='macro')
    print(f'\n--- {name} ---')
    print(classification_report(y, pred, target_names=['nonhateful', 'hateful'],
                                digits=3, zero_division=0))
    return macro, pred


def main():
    en = json.loads((OUT / 'lex_en.json').read_text())
    hi = json.loads((OUT / 'lex_hi.json').read_text())
    char = joblib.load(OUT / 'char_lid.joblib')
    tagger = LanguageTagger(en, hi, char_model=as_tagger_fn(char))

    df = build_frame(tagger)
    print('=' * 68)
    print('EXPERIMENT B - hate classification on romanized Hinglish')
    print('=' * 68)
    print(f'\nrows={len(df)}  hateful={df.y.sum()}  nonhateful={(1-df.y).sum()}')
    print(f'majority-class baseline macro-F1 = '
          f'{f1_score(df.y, np.ones(len(df)), average="macro"):.3f}')

    y = df.y.values
    df['template'] = add_template_ids(df)
    groups = df.template.values
    print(f'distinct templates: {df.template.nunique()} '
          f'(mean {len(df)/df.template.nunique():.1f} sentences per template)')
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    naive_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    num = ['cmi', 'switch_points', 'n_lang_tokens', 'n_hi', 'n_en']
    cat = ['matrix_lang']
    X = df[['roman'] + num + cat]

    text_only = Pipeline([
        ('vec', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                sublinear_tf=True, min_df=2, max_features=60000)),
        ('clf', LogisticRegression(max_iter=3000, C=5.0,
                                   class_weight='balanced', random_state=SEED)),
    ])
    cmi_only = Pipeline([
        ('prep', ColumnTransformer([
            ('n', StandardScaler(), num),
            ('c', OneHotEncoder(handle_unknown='ignore'), cat)])),
        ('clf', LogisticRegression(max_iter=3000, class_weight='balanced',
                                   random_state=SEED)),
    ])
    combined = Pipeline([
        ('prep', ColumnTransformer([
            ('t', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                  sublinear_tf=True, min_df=2,
                                  max_features=60000), 'roman'),
            ('n', StandardScaler(), num),
            ('c', OneHotEncoder(handle_unknown='ignore'), cat)])),
        ('clf', LogisticRegression(max_iter=3000, C=5.0,
                                   class_weight='balanced', random_state=SEED)),
    ])

    fa_naive, _ = evaluate('A-naive: text only, RANDOM folds (LEAKY)',
                           text_only, df.roman, y, naive_cv)
    fa, _ = evaluate('A: text only, template-grouped folds', text_only,
                     df.roman, y, cv, groups)
    fb, _ = evaluate('B: CMI features only', cmi_only, X, y, cv, groups)
    fc, _ = evaluate('C: text + CMI features', combined, X, y, cv, groups)

    print('\n' + '=' * 68)
    print('SUMMARY (5-fold CV, macro-F1)')
    print('=' * 68)
    print(f'  majority baseline              {f1_score(y, np.ones(len(y)), average="macro"):.3f}')
    print(f'  A-naive text only, random CV   {fa_naive:.3f}   <-- inflated by template leakage')
    print(f'  A  text only, grouped CV       {fa:.3f}')
    print(f'  B  CMI only, grouped CV        {fb:.3f}')
    print(f'  C  text + CMI, grouped CV      {fc:.3f}')
    print(f'\n  leakage inflation = {fa_naive - fa:+.3f} macro-F1')
    print(f'\n  delta C - A = {fc - fa:+.3f}')
    if fc - fa < 0.01:
        print('  => CMI adds no usable signal once character n-grams are present.')
    else:
        print('  => CMI adds signal beyond character n-grams.')

    # Does CMI differ between classes at all?
    h = df.loc[df.y == 1, 'cmi']
    n = df.loc[df.y == 0, 'cmi']
    print(f'\n  mean CMI  hateful={h.mean():.2f}  nonhateful={n.mean():.2f}')
    print(f'  mean switch points  hateful={df.loc[df.y==1,"switch_points"].mean():.2f}'
          f'  nonhateful={df.loc[df.y==0,"switch_points"].mean():.2f}')

    json.dump({'A_naive_leaky': fa_naive, 'majority': float(f1_score(y, np.ones(len(y)), average='macro')),
               'A_text': fa, 'B_cmi': fb, 'C_both': fc},
              open(OUT / 'classifier_scores.json', 'w'), indent=2)


if __name__ == '__main__':
    main()
