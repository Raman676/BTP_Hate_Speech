"""
Study 2 classifier: Mica-v0.5-4B (sky7350/Mica-v0.5-4B) for 3-way aggression.

Mica is a Qwen3.5-4B fine-tune that answers a constrained question about an
input and returns a probability for every allowed answer. That suits this
study better than a generic LLM: you get calibrated probabilities over YOUR
label set rather than free text to parse.

!! NOT TESTED IN THIS SANDBOX !!
huggingface.co is unreachable from the container this was written in, and
there is no GPU here. Everything below is written against the model card's
documented API and must be run on Kaggle/Colab. Treat the first run as a
smoke test, not a result.

!! THE CAVEAT THAT MATTERS MOST !!
The model card lists "English, Korean, Chinese + more". It does NOT mention
Hindi, Hinglish, or code-mixed text. That is exactly the circularity risk in
this study's design: if Mica degrades as code-mixing increases, and
code-mixing is your IV, you will measure model failure and report it as a
linguistic finding. Run cmi_degradation_check() BEFORE reporting any
IV -> DV relationship. It is not optional.

--------------------------------------------------------------------------
SETUP (Kaggle notebook, GPU on)

  !pip install -q huggingface_hub requests
  !wget -q https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-server
  # or build llama.cpp per its README; then:
  !huggingface-cli download sky7350/Mica-v0.5-4B --local-dir /kaggle/working/mica

  import subprocess, time
  p = subprocess.Popen(['./llama-server','-m','/kaggle/working/mica/<model>.gguf',
                        '--port','8010','-ngl','99'])
  time.sleep(60)   # wait for load

Check the model card's "Use this model" section for the exact server command
and GGUF filename -- they change between releases.
--------------------------------------------------------------------------
"""
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import requests
except ImportError:
    sys.exit('pip install requests')

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'out'
DEFAULT_URL = 'http://localhost:8010/v1/systemone'

# Label definitions lifted VERBATIM from ANNOTATION_GUIDELINE.md §2.
# The model and the human annotator must judge the same construct, or the
# comparison between them measures nothing.
CRITERIA = {
    'body_shaming':
        'Hostility aimed at a person\'s APPEARANCE: weight, shape, size, '
        'face, skin colour, hair, teeth, ageing, or perceived cosmetic '
        'surgery, filters or editing. The target may be the person addressed '
        'or a third party such as a celebrity or the video\'s subject. '
        'Includes mockery dressed up as concern, and sarcasm where the '
        'literal wording praises but the intended reading mocks how someone '
        'looks. Also includes implicit appearance jabs that use no appearance '
        'vocabulary at all, such as "pehle better thi" (she was better '
        'before) on a video about someone\'s changed looks.',
    'other_aggression':
        'Hostility aimed at a person or group, but NOT about appearance: '
        'their character, intelligence, morals, family, caste, religion, or '
        'gender. Includes insults, threats and demeaning comparisons that do '
        'not concern how someone looks.',
    'profane_not_aggressive':
        'Contains profanity, crude language, or strong negative words, but '
        'is NOT directed hostilely at anyone. Covers profanity used as an '
        'intensifier or in-group solidarity marker between peers, venting '
        'about a situation, and self-directed remarks including self-'
        'deprecating comments about the speaker\'s own body. A severe swear '
        'word used to express enthusiasm belongs here.',
    'neither':
        'An ordinary comment. May be critical, negative, or blunt about a '
        'thing (a video, an idea) without hostility toward a person. '
        'Genuine, non-mocking concern about someone\'s health or wellbeing '
        'belongs here.',
}

INSTRUCTIONS = (
    'This is a comment from an Indian YouTube thread about appearance, body '
    'image or someone\'s changed looks, written in Hindi-English code-mixed '
    'text in Roman script (Hinglish). Judge its PRAGMATIC FORCE: what it '
    'does to its target, not which words it contains. '
    'Two things matter most. First, profanity alone does not make a comment '
    'aggressive -- Hindi swear words such as bhenchod are frequently used as '
    'intensifiers or friendly address between peers, and a comment can be '
    'cruel with no profanity at all. Second, decide whether the hostility is '
    'about how someone LOOKS (body_shaming) or about something else '
    '(other_aggression); when a comment attacks both, choose body_shaming. '
    'Ask what a competent Hinglish speaker would feel this comment was '
    'aimed at doing.'
)

# Maps Mica's option ids onto the guideline's numeric codes (§2).
TO_CODE = {'body_shaming': '3', 'other_aggression': '2',
           'profane_not_aggressive': '1', 'neither': '0'}


class MicaClient:
    def __init__(self, url=DEFAULT_URL, timeout=60, abstain_threshold=0.50,
                 retries=3):
        self.url = url
        self.timeout = timeout
        self.abstain = abstain_threshold
        self.retries = retries

    def health(self) -> bool:
        try:
            r = requests.post(self.url, timeout=10, json={
                'state': {'comment': 'test'},
                'questions': {'aggression': {
                    'type': 'choice', 'criteria': CRITERIA,
                    'instructions': INSTRUCTIONS}}})
            return r.status_code == 200
        except Exception as e:
            print(f'Mica server not reachable at {self.url}: {e}')
            return False

    def decide(self, state: dict) -> dict:
        q = {'aggression': {
            'type': 'choice',
            'criteria': CRITERIA,
            'instructions': INSTRUCTIONS,
            'abstain_threshold': self.abstain,
        }}
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(self.url, timeout=self.timeout,
                                  json={'state': state, 'questions': q})
                r.raise_for_status()
                return r.json()['answers']['aggression']
            except Exception as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f'Mica call failed after {self.retries}: {last}')

    # --------------------------------------------- the two context conditions
    def classify(self, text: str, parent_text: str = None) -> dict:
        """L0 = comment alone. L1 = comment with its parent.

        The context condition is expressed in `state`, which Mica accepts as
        arbitrary JSON. This is the model-side mirror of the human L0/L1
        annotation passes, so the two are directly comparable.
        """
        if parent_text:
            state = {'parent_comment': parent_text, 'comment': text}
        else:
            state = {'comment': text}
        a = self.decide(state)
        probs = a.get('probabilities', {})
        choice = a.get('choice')
        return {
            'mica_choice': choice,
            'mica_code': 'X' if a.get('abstain') else TO_CODE.get(choice, 'X'),
            'mica_confidence': a.get('confidence', float('nan')),
            'mica_abstain': bool(a.get('abstain', False)),
            'p_body_shaming': probs.get('body_shaming', float('nan')),
            'p_other_agg': probs.get('other_aggression', float('nan')),
            'p_profane': probs.get('profane_not_aggressive', float('nan')),
            'p_neither': probs.get('neither', float('nan')),
        }


def run(infile, outfile=None, condition='L0', url=DEFAULT_URL, limit=None):
    """Classify a comments csv. condition: 'L0' (no context) or 'L1' (parent)."""
    df = pd.read_csv(infile)
    if limit:
        df = df.head(limit)
    if condition == 'L1' and 'parent_text' not in df.columns:
        sys.exit("condition L1 needs a 'parent_text' column")

    client = MicaClient(url=url)
    if not client.health():
        sys.exit('Start the llama.cpp server first (see module docstring).')

    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        parent = getattr(r, 'parent_text', None) if condition == 'L1' else None
        if isinstance(parent, float) and np.isnan(parent):
            parent = None
        try:
            res = client.classify(r.text, parent)
        except Exception as e:
            print(f'  row {i} failed: {e}')
            res = {k: None for k in
                   ('mica_choice', 'mica_code', 'mica_confidence',
                    'mica_abstain', 'p_body_shaming', 'p_other_agg',
                    'p_profane', 'p_neither')}
        res['uid'] = r.uid
        rows.append(res)
        if i % 50 == 0:
            print(f'  {i}/{len(df)}')

    res = pd.DataFrame(rows)
    res['condition'] = condition
    out = outfile or (OUT / f'mica_{condition}.csv')
    res.to_csv(out, index=False)
    print(f'wrote {out}  n={len(res)}')
    print(res.mica_code.value_counts().to_string())
    return res


# ===========================================================================
#  THE DIAGNOSTIC THAT GUARDS THE WHOLE STUDY
# ===========================================================================
def cmi_degradation_check(merged: pd.DataFrame, label_col='label',
                          pred_col='mica_code', cmi_col='cmi', bins=5):
    """Does model accuracy fall as code-mixing rises?

    If it does, the model is weaker on code-mixed text, and ANY apparent
    relationship between your code-mixing IV and the model-predicted DV is
    confounded by model error. You would be measuring the instrument, not
    the phenomenon -- the same failure as the template leakage in Pilot 1,
    one level up.

    Pass a frame joining human labels (ground truth) to Mica predictions
    and the CMI features.
    """
    d = merged.dropna(subset=[label_col, pred_col, cmi_col]).copy()
    if d.empty:
        print('nothing to check -- need human labels joined to predictions')
        return None
    d['correct'] = (d[label_col].astype(str).str.upper()
                    == d[pred_col].astype(str).str.upper()).astype(int)
    # equal-width CMI bins; CMI 0 (monolingual) gets its own bin
    d['cmi_bin'] = pd.cut(d[cmi_col], bins=[-0.01, 0.01, 15, 30, 45, 100],
                          labels=['0 (mono)', '0-15', '15-30', '30-45', '45+'])
    g = d.groupby('cmi_bin', observed=True).agg(
        n=('correct', 'size'),
        accuracy=('correct', 'mean'),
        abstain_rate=(pred_col, lambda s: (s.astype(str) == 'X').mean()),
    ).round(3)

    print('=' * 64)
    print('CMI DEGRADATION CHECK  (guards against the circularity confound)')
    print('=' * 64)
    print(g.to_string())

    valid = g[g.n >= 20]
    if len(valid) >= 2:
        # Spearman correlation between bin rank and accuracy
        x = np.arange(len(valid))
        y = valid.accuracy.values
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        rho = np.corrcoef(rx, ry)[0, 1]
        drop = valid.accuracy.iloc[0] - valid.accuracy.iloc[-1]
        print(f'\nSpearman(rank of CMI bin, accuracy) = {rho:+.3f}')
        print(f'accuracy drop, lowest -> highest CMI bin = {drop:+.3f}')
        if rho < -0.5 and drop > 0.05:
            print('\n  *** DEGRADATION DETECTED ***')
            print('  Mica is measurably worse on code-mixed text.')
            print('  Do NOT report an IV -> DV relationship from model labels.')
            print('  Use human labels as the DV, and report this degradation')
            print('  as a finding in its own right -- it is a real result')
            print('  about off-the-shelf models on Hinglish.')
        else:
            print('\n  No strong degradation. Model-labelled DV is defensible')
            print('  as a secondary analysis; human labels remain primary.')
    else:
        print('\n  too few items per bin to judge -- label more data')
    return g


def compare_conditions(l0_path, l1_path, labels_path=None):
    """How much does context change the model's mind? (Study 1 mirror)"""
    a = pd.read_csv(l0_path).set_index('uid')
    b = pd.read_csv(l1_path).set_index('uid')
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    changed = (a.mica_code.astype(str) != b.mica_code.astype(str))
    print(f'items classified in both conditions : {len(common)}')
    print(f'label changed when context added    : {changed.sum()} '
          f'({changed.mean():.1%})')
    print(f'abstain rate L0 (no context)        : '
          f'{(a.mica_code.astype(str)=="X").mean():.1%}')
    print(f'abstain rate L1 (with parent)       : '
          f'{(b.mica_code.astype(str)=="X").mean():.1%}')
    x = pd.crosstab(a.mica_code.astype(str), b.mica_code.astype(str))
    print('\nL0 (rows) -> L1 (cols):')
    print(x.to_string())
    return x


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'health', 'degrade', 'compare'])
    ap.add_argument('--in', dest='infile')
    ap.add_argument('--out', dest='outfile')
    ap.add_argument('--condition', default='L0', choices=['L0', 'L1'])
    ap.add_argument('--url', default=DEFAULT_URL)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--l0'); ap.add_argument('--l1')
    a = ap.parse_args()

    if a.cmd == 'health':
        print('reachable' if MicaClient(url=a.url).health() else 'NOT reachable')
    elif a.cmd == 'run':
        run(a.infile, a.outfile, a.condition, a.url, a.limit)
    elif a.cmd == 'degrade':
        cmi_degradation_check(pd.read_csv(a.infile))
    elif a.cmd == 'compare':
        compare_conditions(a.l0, a.l1)
