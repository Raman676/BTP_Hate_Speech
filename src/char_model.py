"""
Character n-gram backoff classifier for tokens in neither lexicon.

Romanized Hindi has distinctive orthographic signatures (aspirated clusters
bh/dh/gh/kh/th, retroflex-derived spellings, final -aa/-ii/-on/-ke) that
separate it from English well above chance. Trained on lexicon membership,
so it costs no extra annotation.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'out'


def train(seed: int = 42, verbose: bool = True):
    en = set(json.loads((OUT / 'lex_en.json').read_text()))
    hi = set(json.loads((OUT / 'lex_hi.json').read_text()))
    # Train only on unambiguous forms: a word in both lexicons teaches nothing.
    en_only, hi_only = sorted(en - hi), sorted(hi - en)

    # Balance classes — English list is ~90x larger and would swamp Hindi.
    rng = np.random.default_rng(seed)
    n = min(len(en_only), len(hi_only)) * 3
    en_s = list(rng.choice(en_only, size=min(n, len(en_only)), replace=False))
    hi_s = hi_only

    X = en_s + hi_s
    y = ['en'] * len(en_s) + ['hi'] * len(hi_s)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)

    pipe = Pipeline([
        ('vec', TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4),
                                sublinear_tf=True, min_df=2)),
        ('clf', LogisticRegression(max_iter=2000, C=5.0,
                                   class_weight='balanced')),
    ])
    pipe.fit(Xtr, ytr)

    if verbose:
        print(f'char-model training words: en={len(en_s):,} hi={len(hi_s):,}')
        print(classification_report(yte, pipe.predict(Xte), digits=3))
    return pipe


def as_tagger_fn(pipe):
    """Wrap the sklearn pipeline into the callable LanguageTagger expects."""
    def fn(token: str) -> str:
        return pipe.predict([token])[0]
    return fn


if __name__ == '__main__':
    import joblib
    m = train()
    joblib.dump(m, OUT / 'char_lid.joblib')
    print('saved ->', OUT / 'char_lid.joblib')
