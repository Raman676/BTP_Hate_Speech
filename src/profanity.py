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
    # Observed collisions in the L0 corpus (Nov 2026 audit):
    'jyada', 'zyada', 'jiyada', 'jada',        # "more"    vs jihadi (fuzzy)
    'aaram', 'aram', 'araam', 'garam',         # "rest/hot" vs haraam
    'shaadi', 'shadi', 'sadi', 'saadi',        # "wedding" vs sadi
    'khali', 'kaali', 'kali', 'kala', 'kaala', # "empty"/"black" vs kali
    'chhod', 'chod', 'chodo', 'chhodo', 'chhoda', 'choda', 'chhodna',
    'chodna', 'chhodi', 'chodi', 'chhoddo',            # "leave" vs chod
    'paate', 'pate',                                     # vs pathe
}

# Entries in the Hinglish Profanity List (Mathur et al. 2018) that are
# ordinary vocabulary, not taboo words. The list was built for a classifier,
# so it includes anything that CAN be used insultingly. IV-3 measures swear
# frequency: a count of taboo lexis. Whether an ordinary word was used to
# insult is a pragmatic fact that the human label captures, not a lexical one.
#
# Colour terms (kali/kala) are excluded on the same principle. They are the
# core vocabulary of skin-tone body-shaming, so counting them as "swearing"
# would make IV-3 partly a restatement of the DV.  kaalu (a slur-only
# nickname) is kept.
NOT_SWEAR = {
    # severity 1-2: animals, food, objects, everyday nouns
    'jaanvar', 'khota', 'jaat', 'bhains', 'chatani', 'chipkali', 'chipkili',
    'pasine', 'bhoot', 'bhajiye', 'doob', 'khatmal', 'lassan', 'makkhi',
    'haathi', 'gadha', 'unday', 'chaarpai', 'chup', 'danda', 'keera', 'keeda',
    'safaid', 'poot', 'bimaar', 'dum', 'jungli', 'backar', 'kutta', 'kutte',
    'kute', 'kutton',
    # severity 3+: ordinary words with an offensive secondary sense
    'joon', 'kaat', 'gandi', 'chatri', 'bambu', 'maarey', 'jamai', 'sadi',
    'kali', 'laltern', 'behendi', 'padma', 'beej', 'meetha', 'ghussa',
    'toota', 'banda', 'maka', 'gadde', 'bum', 'neech', 'nirodh',
}

# English profanity. The Mathur list is Hindi-only, so without this "fuck"
# and "bitch" scored zero. Severities set by analogy with the list's scale.
EN_PROFANE = {
    'fuck': 8, 'fucking': 8, 'fucked': 8, 'fucker': 8, 'fuckin': 8,
    'motherfucker': 10, 'wtf': 5, 'stfu': 5,
    'shit': 5, 'bullshit': 4, 'bitch': 7, 'bitches': 7, 'bastard': 6,
    'asshole': 7, 'dick': 7, 'pussy': 8, 'slut': 8, 'whore': 8, 'cunt': 9,
}


# Short forms and initialisms that the skeleton rule cannot reach, because
# they are abbreviations rather than spelling variants.
ALIASES = {
    'bc': 'bhenchod', 'bkl': 'bhosda', 'mc': 'madarchod',
    'bsdk': 'bhosda', 'mkc': 'madarchod', 'lnd': 'lund',
    'chu': 'chutiya', 'ch': None,          # too short / ambiguous: ignore
    # vowel-dropped spellings that normalisation cannot reach
    'chtiya': 'chutiya', 'chtiye': 'chutiya', 'ctiya': 'chutiya',
    'chutya': 'chutiya', 'chutiye': 'chutiya', 'chutiyo': 'chutiya',
    'mf': 'motherfucker',
    'bhenchod': 'bahenchod', 'behenchod': 'bahenchod', 'benchod': 'bahenchod',
    'bhencho': 'bahenchod', 'bhenchood': 'bahenchod', 'bhenchodd': 'bahenchod',
    'behnchod': 'bahenchod', 'bahinchod': 'bahenchod', 'bhanchod': 'bahenchod',
    'bhanchodd': 'bahenchod', 'bnchod': 'bahenchod',
    'maderchod': 'madarchod', 'madharchod': 'madarchod',
    'madarchod': 'madarchod', 'madarjat': 'madarchod', 'mdrchod': 'madarchod',
    'bhosdike': 'bhosda', 'bhosdi': 'bhosda', 'bhosdiwale': 'bhosda',
    'bhosadi': 'bhosda', 'bhosri': 'bhosda', 'bhosriwala': 'bhosda',
    'gaandu': 'gaandu', 'gandu': 'gandu',
    'randi': 'randi', 'raand': 'raand', 'harami': 'haraami',
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
    def __init__(self, lexicon: dict, english: set = frozenset()):
        """lexicon: {word: severity 1-10} from the Hinglish Profanity List.
        english: English word list. English words are never spelling-folded
        or fuzzy-matched: "cut" -> chut, "count" -> cunt, "both" -> bhoot
        were the main false positives before this guard."""
        self.english = english
        self.lex = {w.lower(): s for w, s in lexicon.items()
                    if w.lower() not in NOT_SWEAR}
        self.lex.update(EN_PROFANE)
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

        Tiers: exact -> alias -> normalized spelling. English words are only
        ever matched exactly, never folded.
        """
        t = re.sub(r'[^a-z]', '', token.lower().strip())
        if not t:
            return 0
        if t in NEVER_PROFANE or t in NOT_SWEAR:
            return 0
        if t in self.lex:
            return self.lex[t]
        if t in self.english:
            return 0
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
        # Edit-distance-1 matching was removed after the Nov 2026 audit: on the
        # L0 corpus it produced 7 false hits (dharam->haraam, patle->pathe,
        # chopra->chodra, ...) for 1 true one (chtiya), which is now an alias.
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
    en = set()
    if (OUT / 'lex_en.json').exists():
        en = {w.lower() for w in json.load(open(OUT / 'lex_en.json'))}
    return ProfanityMeter(lex, en)


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
