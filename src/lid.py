"""
Token-level language identification + code-mixing metrics for Roman-script
Hindi-English text.

IV operationalisation (see project IV sheet):
  IV-1  CMI            continuous 0-100   (Gamback & Das 2016)
  IV-2  switch_points  integer            language changes between adjacent tokens
  IV-3  matrix_lang    {English, Hindi, Balanced}

Token tagset: hi | en | univ | ne
  univ and ne are EXCLUDED from all three IVs.
"""
import re
import unicodedata
from functools import lru_cache

DEVANAGARI = re.compile(r'[ऀ-ॿ]')
TOKEN_RE = re.compile(r"[A-Za-zऀ-ॿ]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", re.UNICODE)

# ---------------------------------------------------------------- universal
# Excluded from CMI: they belong to neither language.
UNIV_WORDS = {
    'lol', 'lmao', 'rofl', 'haha', 'hahaha', 'hehe', 'hmm', 'hmmm', 'ok', 'okay',
    'oh', 'ohh', 'ah', 'aah', 'arre', 'arey', 'uff', 'ufff', 'omg', 'wtf', 'bro',
    'plz', 'pls', 'thx', 'ty', 'xd', 'hm', 'yaar',  # yaar: discourse marker, debatable
}
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "\U0000FE00-\U0000FE0F" "]+"
)

# ------------------------------------------- Hindi closed-class (high freq)
# Closed-class items are a finite, well-defined set: pronouns, postpositions,
# auxiliaries, determiners, conjunctions. These carry most of the LID signal in
# Hinglish and are spelling-stable, so they are listed explicitly rather than
# learned. Open-class Hindi vocabulary is handled by the lexicon + char model.
HI_FUNCTION = {
    # pronouns / demonstratives
    'main', 'mai', 'mein', 'me', 'hum', 'ham', 'tu', 'tum', 'tumhe', 'tumhein',
    'aap', 'aapko', 'ye', 'yeh', 'yh', 'wo', 'woh', 'vo', 'unhe', 'unhein',
    'uska', 'uski', 'uske', 'iska', 'iski', 'iske', 'unka', 'unki', 'unke',
    'mera', 'meri', 'mere', 'tera', 'teri', 'tere', 'tumhara', 'tumhari',
    'tumhare', 'hamara', 'hamari', 'hamare', 'apna', 'apni', 'apne', 'khud',
    'mujhe', 'mujhko', 'muje', 'humein', 'hume', 'inko', 'unko', 'usko', 'isko',
    # postpositions / case markers
    'ka', 'ki', 'ke', 'ko', 'se', 'par', 'pe', 'tak', 'liye', 'lie', 'saath',
    'sath', 'bina', 'wala', 'wali', 'wale', 'vala', 'vali', 'vale',
    # auxiliaries / copula
    'hai', 'hain', 'he', 'hi', 'ho', 'hoon', 'hun', 'hu', 'tha', 'thi', 'the',
    'thee', 'honge', 'hogi', 'hoga', 'hota', 'hoti', 'hote', 'raha', 'rahi',
    'rahe', 'rha', 'rhi', 'rhe', 'gaya', 'gayi', 'gaye', 'diya', 'diye', 'di',
    'liya', 'liye', 'kiya', 'kiye', 'karo', 'kar', 'karna', 'karne', 'karta',
    'karti', 'karte', 'kare', 'karenge', 'karunga', 'hue', 'hua', 'hui',
    # negation / question
    'nahi', 'nahin', 'nai', 'na', 'mat', 'kya', 'kyu', 'kyun', 'kyon', 'kaise',
    'kaisa', 'kaisi', 'kab', 'kahan', 'kaha', 'kaun', 'kitna', 'kitni', 'kitne',
    'kisko', 'kisne', 'kis',
    # conjunctions / adverbs / quantifiers
    'aur', 'ya', 'lekin', 'par', 'magar', 'kyunki', 'agar', 'to', 'toh', 'bhi',
    'hi', 'sirf', 'bas', 'abhi', 'ab', 'phir', 'fir', 'jab', 'tab', 'yahan',
    'wahan', 'idhar', 'udhar', 'bahut', 'bohot', 'bht', 'thoda', 'zyada',
    'jyada', 'sab', 'sabhi', 'kuch', 'koi', 'har', 'ek', 'do', 'teen', 'itna',
    'itni', 'itne', 'aisa', 'aisi', 'aise', 'waise', 'jaise', 'jaisa', 'jaisi',
    'jo', 'jis', 'jin', 'ke', 'baad', 'pehle', 'pahle', 'andar', 'bahar',
    'upar', 'niche', 'samne', 'peeche', 'piche',
}

# Words that exist in BOTH languages with different meanings. Tagging these by
# dictionary lookup alone is the single biggest error source in Hinglish LID,
# so they are resolved by context (see tag_tokens).
AMBIGUOUS = {
    'to', 'he', 'is', 'me', 'main', 'do', 'bas', 'par', 'the', 'hi', 'so',
    'no', 'in', 'us', 'are', 'ho', 'jo', 'ki', 'ka', 'key', 'man', 'sab',
    'age', 'din', 'bad', 'mere', 'use', 'such', 'hum', 'tu', 'pet', 'kar',
    # Hindi slang whose English sense is ALSO common, so it fails the
    # HI_SLANG inclusion rule below and must be resolved by context instead.
    'solid', 'sun', 'lag', 'beta', 'dal', 'milo', 'sahab', 'chal',
}

# ---------------------------------------------------- Hinglish slang override
# High-frequency colloquial Hindi whose romanization collides with a RARE
# English word present in a large dictionary ("mast" = ship's mast, "mar" = to
# spoil, "sari" = a Japanese measure). The English sense is vanishingly rare in
# social media; the Hindi sense is everywhere. Without this list the tagger
# reports code-mixing in monolingual Hindi comments, inflating IV-1 and IV-2.
#
# Inclusion rule, applied consistently: a form belongs here if its Hindi sense
# is common in informal speech AND its English sense is rare enough that a
# social-media token is almost never the English word. Forms that are genuinely
# frequent in both languages (to, he, is, me, do) stay in AMBIGUOUS above and
# are resolved by context instead.
HI_SLANG = {
    # evaluative adjectives / intensifiers
    'mast', 'jhakaas', 'jhakas', 'bakwaas', 'bakwas', 'badhiya', 'badiya',
    'sahi', 'ganda', 'gandi', 'gande', 'bekar', 'bekaar', 'faltu', 'faltoo',
    'jhand', 'ghatiya', 'zabardast', 'kadak',
    # verbs commonly colliding with English
    'mar', 'mara', 'maro', 'marna', 'pata', 'chal', 'chalo', 'chala', 'bhag',
    'bol', 'bola', 'bolo', 'dekh', 'dekha', 'dekho', 'suno', 'rakh',
    'dala', 'pel', 'ghus', 'laga', 'lage', 'mila',
    # address terms / discourse
    'bhai', 'behen', 'behn', 'didi', 'bhaiya', 'bete', 'bacha',
    'pagal', 'paglu', 'sahab', 'ji', 're', 'abe', 'oye', 'arre',
    # frequent nouns
    'baat', 'baatein', 'kaam', 'cheez', 'cheeze', 'log', 'logo', 'logon',
    'aadmi', 'aurat', 'ladka', 'ladki', 'ladke', 'ladkiyan', 'bacche',
    'paisa', 'paise', 'desh', 'duniya', 'zindagi', 'jindagi', 'shakal',
    'chehra', 'sharir', 'body',  # 'body' stays EN: see note below
    # misc high-frequency
    'turant', 'sah', 'jaa', 'jao', 'aao', 'aaja', 'hatt', 'chup', 'theek',
    'thik', 'accha', 'acha', 'achha', 'bura', 'nalayak', 'kamina', 'kamine',
}
# 'body' is a genuine English borrowing into Hindi speech. Per the annotation
# guideline it is tagged EN (established borrowing keeps its source language),
# so it is removed here rather than silently included above.
HI_SLANG.discard('body')


def _strip(tok: str) -> str:
    return unicodedata.normalize('NFKC', tok).lower().strip()


class LanguageTagger:
    """Lexicon-first tagger with a character n-gram backoff for OOV tokens."""

    def __init__(self, en_words, hi_words, ne_words=None, char_model=None):
        self.en = set(en_words)
        self.hi = set(hi_words) | HI_FUNCTION
        self.ne = set(ne_words or ())
        self.char_model = char_model
        # Any surface form present in BOTH lexicons is ambiguous by definition
        # and must go to context resolution. Deriving this set from the data
        # rather than hand-listing it was the single biggest accuracy win:
        # frequent Hindi postpositions ("se", "ka", "ki") are also entries in
        # a large English wordlist, and letting them fall through to the
        # character model tagged them English every time.
        self.ambiguous = (self.en & self.hi) | AMBIGUOUS
        self.en_only = self.en - self.ambiguous
        self.hi_only = self.hi - self.ambiguous
        # Closed-class English function words always win: they are short,
        # extremely frequent, and the character model has no chance on them.
        self.en_force = {
            'i', 'a', 'an', 'the', 'is', 'am', 'are', 'was', 'were', 'be',
            'been', 'we', 'you', 'he', 'she', 'it', 'they', 'them', 'his',
            'her', 'our', 'your', 'their', 'my', 'me', 'us', 'him', 'this',
            'that', 'these', 'those', 'and', 'or', 'but', 'if', 'of', 'to',
            'in', 'on', 'at', 'for', 'with', 'from', 'by', 'as', 'not',
            'all', 'should', 'would', 'could', 'will', 'can', 'do', 'does',
            'did', 'have', 'has', 'had', 'want', 'think', 'know', 'like',
            'totally', 'always', 'never', 'very', 'really', 'just', 'so',
            'what', 'when', 'where', 'who', 'why', 'how', 'people', 'about',
        }

    # ---------------------------------------------------------------- tagging
    def tag_token(self, tok: str) -> str:
        t = _strip(tok)
        if not t:
            return 'univ'
        if EMOJI_RE.search(tok) or not re.search(r'[A-Za-zऀ-ॿ]', tok):
            return 'univ'          # punctuation, digits, emoji
        if t in UNIV_WORDS:
            return 'univ'
        if DEVANAGARI.search(tok):
            return 'hi'            # script is decisive when present
        if t in self.ne:
            return 'ne'
        if t in self.en_force:
            return 'en'            # closed-class English beats everything
        if t in HI_FUNCTION or t in HI_SLANG:
            return 'hi'            # closed-class + slang Hindi beat the lexicon
        if t in self.hi_only:
            return 'hi'
        if t in self.en_only:
            return 'en'
        if t in self.ambiguous:
            return 'amb'           # resolved later using context
        if self.char_model is not None:
            return self.char_model(t)
        return 'en' if t in self.en else 'hi'

    def tag_tokens(self, text: str):
        toks = TOKEN_RE.findall(str(text))
        tags = [self.tag_token(t) for t in toks]
        # Context resolution for ambiguous tokens: adopt the language of the
        # nearest unambiguous content neighbour (Hindi wins ties, because an
        # ambiguous form inside a Hindi clause is far more often Hindi).
        for i, tg in enumerate(tags):
            if tg != 'amb':
                continue
            left = next((tags[j] for j in range(i - 1, -1, -1)
                         if tags[j] in ('hi', 'en')), None)
            right = next((tags[j] for j in range(i + 1, len(tags))
                          if tags[j] in ('hi', 'en')), None)
            if left == right and left is not None:
                tags[i] = left
            elif left is not None and right is not None:
                tags[i] = 'hi' if 'hi' in (left, right) else 'en'
            else:
                tags[i] = left or right or 'hi'
        return list(zip(toks, tags))


# ----------------------------------------------------------- the three IVs
def cmi(tags) -> float:
    """Code-Mixing Index, Gamback & Das (2016), utterance level, 0-100.

    CMI = 100 * (1 - max(w_Li)/n)  where n = language-tagged tokens only.
    univ / ne tokens are excluded from n, per the IV sheet exclusion rule.
    Returns 0.0 for an utterance with no language-tagged tokens.
    """
    lang = [t for t in tags if t in ('hi', 'en')]
    n = len(lang)
    if n == 0:
        return 0.0
    top = max(lang.count('hi'), lang.count('en'))
    return 100.0 * (1 - top / n)


def switch_points(tags) -> int:
    """Number of language changes between adjacent language-tagged tokens."""
    lang = [t for t in tags if t in ('hi', 'en')]
    return sum(1 for a, b in zip(lang, lang[1:]) if a != b)


def matrix_language(tags, threshold: float = 0.60) -> str:
    """Dominant language. 'Balanced' when neither side exceeds `threshold`."""
    lang = [t for t in tags if t in ('hi', 'en')]
    n = len(lang)
    if n == 0:
        return 'None'
    hi = lang.count('hi') / n
    if hi >= threshold:
        return 'Hindi'
    if (1 - hi) >= threshold:
        return 'English'
    return 'Balanced'


def describe(text: str, tagger: LanguageTagger) -> dict:
    pairs = tagger.tag_tokens(text)
    tags = [t for _, t in pairs]
    lang = [t for t in tags if t in ('hi', 'en')]
    return {
        'n_tokens': len(tags),
        'n_lang_tokens': len(lang),
        'n_hi': lang.count('hi'),
        'n_en': lang.count('en'),
        'cmi': round(cmi(tags), 2),
        'switch_points': switch_points(tags),
        'matrix_lang': matrix_language(tags),
    }
