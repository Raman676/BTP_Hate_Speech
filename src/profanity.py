"""
IV-3: profanity measurement for Roman-script Hinglish.

Two deliberately separate constructs. Conflating them is the single most
common error in this literature (Kumar et al. 2021: profanity alone does not
make a comment offensive; Davidson et al. 2017 built a three-way scheme around
exactly this distinction).

  PRESENCE  - is a swear word there, and how severe?        (lexical fact)
  VALENCE   - is it aimed AT someone, or used as an
              intensifier / solidarity marker?              (pragmatic fact)

"kya mast h ye bhenchod" scores high on presence and NEGATIVE on direction:
the profanity is an intensifier expressing enthusiasm, not an attack. Any
model that sees only presence will call this hate speech.

Romanization variance is handled by normalising both the lexicon and the
input to a consistent skeleton, because no fixed list covers the spellings
users actually type (bhenchod / bhanchod / behenchod / bhencho / bc).
"""
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'out'


def normalize(w: str) -> str:
    """Fold a romanized token to a spelling-invariant form.

    Romanized Hindi has no standard orthography, so the same word appears
    many ways. These rules fold the variation users actually produce:
    repeated characters, f/ph, v/w, z/j, long vowels, and the aspiration
    marker <h>, whose placement is the most variable part of all
    (bhenchod / bahenchod / behenchod all differ only in <h> and schwas).

    NOTE: vowels are KEPT. An earlier version stripped them to a consonant
    skeleton, which collapsed "ganda" (dirty) onto "gandu" (an insult) and
    "kitna" (how much) onto a profanity. Vowel-stripping is too lossy for
    this lexicon; near-miss pairs are instead handled by NEVER_PROFANE.
    """
    w = w.lower().strip()
    w = re.sub(r'[^a-z]', '', w)
    w = re.sub(r'(.)\1+', r'\1', w)        # bhaaaii -> bhai
    w = w.replace('ph', 'f').replace('v', 'w').replace('z', 'j')
    w = w.replace('h', '')                 # aspiration: most variable marker
    w = w.replace('aa', 'a').replace('ee', 'i').replace('oo', 'u')
    w = w.replace('y', 'i')
    return w


def edit_distance(a: str, b: str, cap: int = 2) -> int:
    """Levenshtein distance, early-exit once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


# Common, non-profane Hindi words that sit within edit distance 1 of a
# profanity and would otherwise produce false positives. Checked FIRST.
# This list grows as the error analysis finds more; every entry is a real
# collision observed in testing, not a guess.
NEVER_PROFANE = {
    'ganda', 'gandi', 'gande', 'gandagi',      # dirty  vs  gandu
    'kitna', 'kitni', 'kitne',                 # how much
    'land', 'lagta', 'lagti', 'laga',          # vs lund
    'chat', 'chata', 'chata',                  # vs chut
    'bahan', 'bahen', 'behen', 'behn',         # sister (neutral kin term)
    'bhai', 'bhaiya',                          # brother
    'chota', 'choti', 'chote',                 # small
    'mast', 'manda',
    'kutta', 'kutte',   # literally "dog"; insult only when directed -> valence
}


# Short forms and initialisms that the skeleton rule cannot reach, because
# they are abbreviations rather than spelling variants.
ALIASES = {
    'bc': 'bhenchod', 'bkl': 'bhosadike', 'mc': 'madarchod',
    'bsdk': 'bhosadike', 'mkc': 'madarchod', 'lnd': 'lund',
    'chu': 'chutiya', 'ch': None,          # too short / ambiguous: ignore
}

# Second person / addressee markers. Profanity co-occurring with one of these
# is far more likely to be DIRECTED than used as an intensifier.
ADDRESSEE = {
    'tu', 'tum', 'tera', 'teri', 'tere', 'tumhara', 'tumhari', 'tumhare',
    'tujhe', 'tujhko', 'tuje', 'tumhe', 'tumhein', 'aap', 'aapka', 'aapki',
    'you', 'your', 'youre', 'u', 'ur', 'tujhse', 'tumse',
}

# Positive evaluative context. Profanity next to one of these is far more
# likely to be an INTENSIFIER (the "kya mast h ye bhenchod" pattern).
POSITIVE = {
    'mast', 'badhiya', 'badiya', 'sahi', 'zabardast', 'jhakaas', 'jhakas',
    'kadak', 'solid', 'awesome', 'amazing', 'great', 'love', 'loved',
    'best', 'perfect', 'beautiful', 'cute', 'nice', 'super', 'op', 'fire',
    'goat', 'legend', 'king', 'queen', 'wah', 'waah',
}
# 'kya' was removed: it is a bare question word, not an evaluative one, and
# including it tagged "bhenchod ye kya hai" as an intensifier on no evidence.

# Imperative / threat verbs. Profanity near one of these is directed.
THREAT = {
    'mar', 'maro', 'marna', 'marunga', 'marenge', 'jaa', 'jao', 'nikal',
    'nikal', 'bhag', 'chup', 'hatt', 'die', 'kill', 'hang', 'maar',
}


class ProfanityMeter:
    def __init__(self, lexicon: dict):
        """lexicon: {word: severity 1-10} from the Hinglish Profanity List."""
        self.lex = {w.lower(): s for w, s in lexicon.items()}
        # normalized form -> max severity among words sharing it
        self.norm = {}
        for w, s in self.lex.items():
            k = normalize(w)
            if len(k) >= 2:
                self.norm[k] = max(self.norm.get(k, 0), s)
        # bucket by length so fuzzy lookup only compares plausible candidates
        self.by_len = {}
        for k, s in self.norm.items():
            self.by_len.setdefault(len(k), []).append((k, s))

    def lookup(self, token: str):
        """Return severity 1-10, or 0 if not profane.

        Three tiers: exact -> alias -> edit distance 1. Fuzzy matching is
        restricted to normalized forms of length >= 5, because short forms
        collide far too readily (every 3-letter string is within distance 1
        of several others).
        """
        t = re.sub(r'[^a-z]', '', token.lower().strip())
        if not t:
            return 0
        if t in NEVER_PROFANE:
            return 0
        if t in self.lex:
            return self.lex[t]
        if t in ALIASES and ALIASES[t]:
            tgt = ALIASES[t].lower()
            if tgt in self.lex:
                return self.lex[tgt]
            if normalize(tgt) in self.norm:
                return self.norm[normalize(tgt)]
        k = normalize(t)
        if len(k) < 3 or k in {normalize(x) for x in NEVER_PROFANE}:
            return 0
        if k in self.norm:
            return self.norm[k]
        if len(k) >= 5:
            for L in (len(k) - 1, len(k), len(k) + 1):
                for cand, sev in self.by_len.get(L, ()):
                    if edit_distance(k, cand, 1) <= 1:
                        return sev
        return 0

    # ------------------------------------------------------------ the IV
    def measure(self, tokens) -> dict:
        """Compute presence AND valence features for one comment.

        direction_score > 0  => profanity looks DIRECTED at an addressee
        direction_score < 0  => profanity looks like an INTENSIFIER
        direction_score == 0 => profane but ambiguous, or no profanity
        """
        toks = [str(t).lower() for t in tokens]
        sev = [self.lookup(t) for t in toks]
        hits = [(t, s) for t, s in zip(toks, sev) if s > 0]
        n = max(len(toks), 1)

        ctx = set(toks)
        has_addr = bool(ctx & ADDRESSEE)
        has_pos = bool(ctx & POSITIVE)
        has_threat = bool(ctx & THREAT)

        direction = 0
        if hits:
            direction = (2 * has_addr) + (2 * has_threat) - (2 * has_pos)

        return {
            'swear_count': len(hits),
            'swear_freq': round(len(hits) / n, 4),   # IV-3 as specified
            'swear_max_severity': max([s for _, s in hits], default=0),
            'swear_mean_severity': round(
                sum(s for _, s in hits) / len(hits), 2) if hits else 0.0,
            'has_addressee': int(has_addr),
            'has_positive_ctx': int(has_pos),
            'has_threat_verb': int(has_threat),
            'direction_score': direction,
            'profanity_valence': ('directed' if direction > 0 else
                                  'intensifier' if direction < 0 else
                                  'ambiguous' if hits else 'none'),
            'swear_tokens': [t for t, _ in hits],
        }


def load(path: Path = None) -> ProfanityMeter:
    import pandas as pd
    p = path or (ROOT / 'data' / 'Hinglish_Profanity_List.csv')
    d = pd.read_csv(p, header=None, names=['w', 'gloss', 'lvl'],
                    encoding='latin-1')
    lex = {str(w).strip(): int(l) for w, l in zip(d.w, d.lvl)
           if str(w).strip()}
    return ProfanityMeter(lex)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lid import TOKEN_RE
    pm = load()
    tests = [
        "bhenchod ye kya hai",
        "behenchod",
        "bhanchod",
        "ye bohot ganda lag raha hai",
        "kitna time lagega",
        'kya mast h ye bhenchod',
        'bhenchod tu mar jaa',
        'tera chehra kitna ganda hai bc',
        'ye video bohot badhiya hai',
        'madarchod teri body dekh',
        'bhai kya mast video hai',
    ]
    for s in tests:
        r = pm.measure(TOKEN_RE.findall(s))
        print(f'{s!r}')
        print(f"   hits={r['swear_tokens']}  freq={r['swear_freq']}  "
              f"sev={r['swear_max_severity']}  -> {r['profanity_valence'].upper()}"
              f" (dir={r['direction_score']})")
