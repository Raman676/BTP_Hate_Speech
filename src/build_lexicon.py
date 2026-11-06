"""
Build the romanized-Hindi lexicon and the character n-gram backoff classifier.

Hindi side: the HateCheckHIn Devanagari corpus is transliterated to Roman and
expanded with spelling variants, because real users do not write one canonical
romanization. English side: a general 234k-word English list, independent of
any corpus used here.

NOTE ON LEAKAGE: the Hindi lexicon is derived from the Devanagari half of the
same test suite whose Roman half we later measure. Vocabulary therefore
overlaps. This is fine for building a working pipeline but means in-suite
tagging accuracy is optimistic; the honest number is the held-out hand-tagged
gold evaluation (eval_gold.py).
"""
import re
import json
import pandas as pd
from pathlib import Path
from indic_transliteration.sanscript import transliterate
from english_words import get_english_words_set

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / 'data', ROOT / 'out'
OUT.mkdir(exist_ok=True)

DEV_WORD = re.compile(r'[ऀ-ॿ]+')


def romanize(dev_word: str) -> str:
    """Devanagari -> plain ASCII roman, ITRANS then stripped of markup."""
    r = transliterate(dev_word, 'devanagari', 'itrans').lower()
    r = re.sub(r'[^a-z]', '', r)
    return r


def spelling_variants(r: str) -> set:
    """Expand one canonical romanization into forms users actually type."""
    v = {r}
    rules = [
        (r'aa', 'a'), (r'ee', 'i'), (r'ii', 'i'), (r'oo', 'u'), (r'uu', 'u'),
        (r'ph', 'f'), (r'v', 'w'), (r'w', 'v'), (r'z', 'j'),
        (r'am$', 'an'), (r'm$', 'n'), (r'ca', 'cha'), (r'sh', 's'),
    ]
    for _ in range(2):
        for a, b in rules:
            for w in list(v):
                nw = re.sub(a, b, w)
                if nw and nw != w:
                    v.add(nw)
    # drop trailing schwa: "nafarata" -> "nafarat" -> "nafrat"
    for w in list(v):
        if len(w) > 4 and w.endswith('a'):
            v.add(w[:-1])
    # collapse internal schwa in CaCaC patterns
    for w in list(v):
        nw = re.sub(r'([bcdfghjklmnpqrstvwxyz])a([bcdfghjklmnpqrstvwxyz])a'
                    r'([bcdfghjklmnpqrstvwxyz])', r'\1a\2\3', w)
        if nw != w:
            v.add(nw)
    return {w for w in v if len(w) >= 2}


def main():
    en = {w.lower() for w in get_english_words_set(['web2'], lower=True)
          if w.isalpha() and len(w) >= 2}
    print(f'English lexicon: {len(en):,}')

    mono = pd.read_csv(DATA / 'monolingual_functionalities.csv')
    dev_vocab = set()
    for s in mono.test_case.astype(str):
        dev_vocab.update(DEV_WORD.findall(s))
    print(f'Devanagari vocabulary: {len(dev_vocab):,}')

    hi_canon, hi_full = set(), set()
    for w in dev_vocab:
        r = romanize(w)
        if len(r) >= 2:
            hi_canon.add(r)
            hi_full |= spelling_variants(r)
    print(f'Romanized Hindi (canonical): {len(hi_canon):,}')
    print(f'Romanized Hindi (+variants): {len(hi_full):,}')

    # Hinglish profanity list is an independent resource (Mathur et al. 2018)
    prof = pd.read_csv(DATA / 'Hinglish_Profanity_List.csv', header=None,
                       names=['word', 'gloss', 'level'],
                       encoding='latin-1')
    prof_words = {str(w).strip().lower() for w in prof.word if str(w).strip()}
    hi_full |= prof_words
    print(f'Hinglish profanity list: {len(prof_words):,}')

    (OUT / 'lex_en.json').write_text(json.dumps(sorted(en)))
    (OUT / 'lex_hi.json').write_text(json.dumps(sorted(hi_full)))
    (OUT / 'lex_hi_canonical.json').write_text(json.dumps(sorted(hi_canon)))
    (OUT / 'profanity.json').write_text(json.dumps(sorted(prof_words)))
    print('\nwrote lexicons to', OUT)

    overlap = en & hi_full
    print(f'en/hi overlap (ambiguous surface forms): {len(overlap):,}')
    print('  examples:', sorted(list(overlap))[:15])


if __name__ == '__main__':
    main()
