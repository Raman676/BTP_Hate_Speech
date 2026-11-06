"""
Collect threaded YouTube comments for the code-mixing study.

Why YouTube and not X/Twitter: commentThreads.list returns a top-level comment
WITH its replies nested. That parent-child link is the independent variable's
whole basis. X charges $200-5000/month and does not hand you clean threads.

Outputs comments.csv with the columns make_sheets.py expects:
    uid, video_id, thread_id, comment_id, parent_id, reply_position,
    text, parent_text, author_hash, published_at, like_count

PRIVACY: author display names and channel IDs are never stored. Each author
becomes a salted hash, so you can still tell "same person replied twice"
without holding anyone's identity. Do not remove this.

QUOTA: default YouTube API allowance is 10,000 units/day. commentThreads.list
and comments.list cost 1 unit per call, 100 items per call. Collecting 5,000
comments costs well under 100 units, so quota is not a real constraint here.
"""
import os
import re
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
import pandas as pd

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    sys.exit('pip install google-api-python-client')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

# ---------------------------------------------------------------------------
# Key resolution order: --key flag -> secrets.py -> YOUTUBE_API_KEY env var.
# secrets.py is gitignored, so the key never travels with the shared project.
def _load_key():
    try:
        sys.path.insert(0, str(ROOT))
        from config_local import YOUTUBE_API_KEY           # noqa
        if YOUTUBE_API_KEY and 'PASTE' not in YOUTUBE_API_KEY:
            return YOUTUBE_API_KEY
    except Exception:
        pass
    return os.environ.get('YOUTUBE_API_KEY', '')


# Search queries used to FIND body-shaming-prone videos automatically, so you
# do not have to hand-collect video IDs. These target Hindi/Hinglish content
# whose COMMENT SECTIONS carry appearance talk -- that is where the phenomenon
# lives, not necessarily in the video itself.
#
# Mixing Devanagari and Roman queries is deliberate: they surface different
# creators. The comments we keep are Roman-script only, but the SEARCH need
# not be.
BODY_SHAMING_QUERIES = [
    # transformation / weight, where appearance commentary is the topic
    'weight loss transformation india vlog',
    'motapa kam kaise kare vlog',
    'वजन कम करने के बाद transformation',
    'fat to fit journey india',
    # celebrity appearance gossip, reliably draws appearance comments
    'bollywood actress weight gain news',
    'actress looking different trolled',
    'celebrity body transformation bollywood',
    'actress trolled for looks',
    # body image / plus size creators, draws both support and abuse
    'plus size fashion india haul',
    'body positivity india hindi',
    'dark skin beauty india hindi',
    'skinny shaming india',
    # reality TV / influencer drama, high-volume aggressive comment sections
    'bigg boss contestant looks trolled',
    'influencer trolled appearance india',
]

# Or hand-pick video IDs (the part after v= in the URL) and skip search.
VIDEO_IDS = []
# ---------------------------------------------------------------------------

SALT = 'cmi-study-2026'                      # change this and keep it private
DEVANAGARI = re.compile(r'[ऀ-ॿ]')
LATIN = re.compile(r'[A-Za-z]')

# Appearance vocabulary, Hindi (romanized) and English. Used to FLAG comments
# as appearance-relevant, not to filter them by default.
#
# Why a flag and not a hard filter: the base rate of body-shaming in a comment
# section is itself a finding, and hard-filtering would destroy it. It would
# also bias the sample toward explicit vocabulary and miss the implicit cases
# ("pehle better thi") that are the most interesting ones. Use
# --appearance-only when you want to concentrate labelling effort, and say so
# in the paper if you do.
APPEARANCE_TERMS = {
    # Hindi — body / weight / shape
    'mota', 'moti', 'motu', 'motapa', 'mote', 'patla', 'patli', 'patle',
    'dubla', 'dubli', 'sukhi', 'sukha', 'bhains', 'haathi', 'tank',
    'sehat', 'wajan', 'vajan', 'body', 'figure', 'sharir', 'dhancha',
    # Hindi — face / skin / colour
    'shakal', 'chehra', 'chehre', 'muh', 'rang', 'kala', 'kali', 'kaali',
    'gora', 'gori', 'saanwla', 'sanwli', 'daag', 'pimples', 'keel',
    'badsurat', 'bhadda', 'bhaddi', 'ganda', 'gandi',
    # Hindi — hair / age / general appearance
    'ganja', 'baal', 'budha', 'budhi', 'buddha', 'jawan', 'aunty', 'uncle',
    'surgery', 'botox', 'makeup', 'filter', 'edit',
    # English
    'fat', 'obese', 'chubby', 'skinny', 'thin', 'slim', 'bony', 'anorexic',
    'weight', 'weigh', 'size', 'shape', 'belly', 'tummy', 'stomach',
    'face', 'skin', 'dark', 'fair', 'ugly', 'pretty', 'beautiful', 'looks',
    'looking', 'look', 'appearance', 'bald', 'old', 'aged', 'plastic',
    'nose', 'teeth', 'lips', 'thighs', 'arms', 'cheeks', 'jawline',
}


def appearance_hits(text: str):
    """Which appearance terms appear in this comment."""
    toks = set(re.findall(r"[a-z]+", str(text).lower()))
    return sorted(toks & APPEARANCE_TERMS)


def author_hash(channel_id: str) -> str:
    return hashlib.sha256((SALT + str(channel_id)).encode()).hexdigest()[:16]


def uid_for(text: str) -> str:
    """Must match make_sheets.uid_for so sheets and features join correctly."""
    return hashlib.sha1(('cmi2026' + str(text)).encode()).hexdigest()[:12]


def clean(t: str) -> str:
    t = re.sub(r'<br\s*/?>', ' ', str(t))
    t = re.sub(r'<[^>]+>', ' ', t)
    t = (t.replace('&quot;', '"').replace('&#39;', "'")
          .replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>'))
    return re.sub(r'\s+', ' ', t).strip()


def is_roman_script(t: str) -> bool:
    """Keep Roman-script only (the project's stated constraint).

    NOTE: Roman script is an ORTHOGRAPHY, not a language. "bhai ye to bilkul
    hi kharab lag rahi hai" is monolingual Hindi written in Roman letters and
    is kept. This filter removes Devanagari, not Hindi.
    """
    return bool(LATIN.search(t)) and not DEVANAGARI.search(t)


def fetch_video(yt, video_id, max_threads=200, min_replies=1, sleep=0.1):
    """Fetch comment threads plus their replies for one video."""
    rows, page, n_threads = [], None, 0
    while True:
        try:
            resp = yt.commentThreads().list(
                part='snippet,replies', videoId=video_id, maxResults=100,
                pageToken=page, textFormat='plainText', order='relevance',
            ).execute()
        except HttpError as e:
            reason = ''
            try:
                reason = json.loads(e.content)['error']['errors'][0]['reason']
            except Exception:
                reason = str(e)[:120]
            if reason in ('commentsDisabled', 'videoNotFound'):
                print(f'  [{video_id}] skipped: {reason}')
            elif reason == 'quotaExceeded':
                print('  QUOTA EXCEEDED — stopping. Resume tomorrow.')
                raise SystemExit(1)
            else:
                print(f'  [{video_id}] error: {reason}')
            return rows

        for item in resp.get('items', []):
            top = item['snippet']['topLevelComment']
            ts = top['snippet']
            thread_id = item['id']
            total_replies = item['snippet'].get('totalReplyCount', 0)
            if total_replies < min_replies:
                continue        # no replies -> no parent/child pair -> useless

            parent_text = clean(ts.get('textOriginal', ''))
            rows.append(dict(
                video_id=video_id, thread_id=thread_id,
                comment_id=top['id'], parent_id='', reply_position=0,
                text=parent_text, parent_text='',
                author_hash=author_hash(
                    ts.get('authorChannelId', {}).get('value', '')),
                published_at=ts.get('publishedAt', ''),
                like_count=ts.get('likeCount', 0),
                total_replies=total_replies,
            ))

            # replies: the inline block holds at most 5, so fetch the rest
            replies = item.get('replies', {}).get('comments', [])
            if total_replies > len(replies):
                replies = fetch_all_replies(yt, top['id']) or replies
            for pos, r in enumerate(replies, start=1):
                rs = r['snippet']
                rows.append(dict(
                    video_id=video_id, thread_id=thread_id,
                    comment_id=r['id'], parent_id=top['id'],
                    reply_position=pos,
                    text=clean(rs.get('textOriginal', '')),
                    parent_text=parent_text,
                    author_hash=author_hash(
                        rs.get('authorChannelId', {}).get('value', '')),
                    published_at=rs.get('publishedAt', ''),
                    like_count=rs.get('likeCount', 0),
                    total_replies=0,
                ))
            n_threads += 1
            if n_threads >= max_threads:
                return rows

        page = resp.get('nextPageToken')
        if not page:
            return rows
        time.sleep(sleep)


def search_videos(yt, queries, per_query=8, region='IN', lang='hi'):
    """Find candidate videos by query.

    QUOTA: search.list costs 100 units per call against a 10,000/day default.
    14 queries = 1,400 units. commentThreads.list costs 1 unit per 100
    comments, so collection itself is nearly free. Budget accordingly: you
    can afford roughly 80 searches a day, not 800.
    """
    found, seen = [], set()
    for q in queries:
        try:
            resp = yt.search().list(
                part='snippet', q=q, type='video', maxResults=per_query,
                regionCode=region, relevanceLanguage=lang,
                order='relevance', videoDuration='medium',
            ).execute()
        except HttpError as e:
            reason = str(e)[:100]
            if 'quota' in reason.lower():
                print('  QUOTA EXCEEDED during search — stopping.')
                break
            print(f'  search failed for {q!r}: {reason}')
            continue
        for it in resp.get('items', []):
            vid = it['id']['videoId']
            if vid in seen:
                continue
            seen.add(vid)
            found.append({'video_id': vid,
                          'title': it['snippet']['title'],
                          'channel': it['snippet']['channelTitle'],
                          'query': q})
        print(f'  {len(resp.get("items", []))} results  <- {q!r}')
        time.sleep(0.2)
    return found


def fetch_all_replies(yt, parent_id, sleep=0.1):
    out, page = [], None
    while True:
        try:
            resp = yt.comments().list(
                part='snippet', parentId=parent_id, maxResults=100,
                pageToken=page, textFormat='plainText').execute()
        except HttpError:
            return out
        out.extend(resp.get('items', []))
        page = resp.get('nextPageToken')
        if not page:
            return out
        time.sleep(sleep)


def main():
    ap = argparse.ArgumentParser(
        description='Collect threaded Hinglish YouTube comments on '
                    'body-shaming content.')
    ap.add_argument('--key', default=None)
    ap.add_argument('--videos', nargs='*', default=None,
                    help='video IDs; omit to search automatically')
    ap.add_argument('--search', action='store_true',
                    help='find videos with BODY_SHAMING_QUERIES')
    ap.add_argument('--queries', nargs='*', default=None,
                    help='override the built-in search queries')
    ap.add_argument('--per-query', type=int, default=8)
    ap.add_argument('--max-threads', type=int, default=200, help='per video')
    ap.add_argument('--min-replies', type=int, default=1,
                    help='skip threads with fewer replies (no parent pair)')
    ap.add_argument('--appearance-only', action='store_true',
                    help='keep only comments (or replies to comments) that '
                         'mention appearance; concentrates labelling effort '
                         'but destroys the base rate — say so in the paper')
    ap.add_argument('--out', default=str(DATA / 'comments.csv'))
    a = ap.parse_args()

    key = a.key or _load_key()
    if not key or 'PASTE' in key:
        sys.exit('No API key. Put it in config_local.py, set YOUTUBE_API_KEY, '
                 'or pass --key.')

    yt = build('youtube', 'v3', developerKey=key, cache_discovery=False)

    vids = a.videos or VIDEO_IDS
    if not vids or a.search:
        qs = a.queries or BODY_SHAMING_QUERIES
        print(f'searching {len(qs)} queries for body-shaming-prone videos '
              f'(~{100*len(qs)} quota units)...')
        found = search_videos(yt, qs, a.per_query)
        if not found:
            sys.exit('search returned nothing — check the key and quota.')
        pd.DataFrame(found).to_csv(DATA / 'videos_found.csv', index=False)
        vids = [f['video_id'] for f in found]
        print(f'\n{len(vids)} unique videos -> data/videos_found.csv')
        print('Skim that file and drop anything off-topic before a big run.\n')

    allrows = []
    for i, v in enumerate(vids, 1):
        print(f'[{i}/{len(vids)}] {v}')
        r = fetch_video(yt, v, a.max_threads, a.min_replies)
        print(f'  collected {len(r)} comments')
        allrows += r

    if not allrows:
        sys.exit('Nothing collected.')

    df = pd.DataFrame(allrows)
    n_raw = len(df)
    df['text'] = df.text.map(clean)
    df['parent_text'] = df.parent_text.map(clean)
    df = df[df.text.str.len() >= 3]
    n_short = n_raw - len(df)
    df = df[df.text.map(is_roman_script)]
    n_dev = n_raw - n_short - len(df)
    df = df.drop_duplicates('text').reset_index(drop=True)
    df.insert(0, 'uid', df.text.map(uid_for))

    # appearance relevance, on the comment and on its parent
    df['appearance_terms'] = df.text.map(lambda t: ','.join(appearance_hits(t)))
    df['is_appearance'] = (df.appearance_terms != '').astype(int)
    df['parent_is_appearance'] = df.parent_text.map(
        lambda t: int(bool(appearance_hits(t))))
    n_app = int(df.is_appearance.sum())

    if a.appearance_only:
        df = df[(df.is_appearance == 1) | (df.parent_is_appearance == 1)]
        df = df.reset_index(drop=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f'\n{"="*60}')
    print(f'raw comments fetched       : {n_raw}')
    print(f'dropped, too short         : {n_short}')
    print(f'dropped, Devanagari        : {n_dev}')
    print(f'Roman, deduped             : {n_raw - n_short - n_dev}')
    print(f'  mentioning appearance    : {n_app} '
          f'({n_app / max(n_raw - n_short - n_dev, 1):.1%} base rate)')
    if a.appearance_only:
        print(f'  --appearance-only kept   : {len(df)}')
    print(f'FINAL written              : {len(df)}')
    print(f'  threads                  : {df.thread_id.nunique()}')
    print(f'  replies (have a parent)  : {(df.parent_id != "").sum()}')
    print(f'  distinct authors         : {df.author_hash.nunique()}')
    print(f'\nwrote {a.out}')
    if n_app < 200 and not a.appearance_only:
        print('\n  WARNING: few appearance-related comments. These videos may '
              'not be drawing body talk — try different ones before labelling.')
    print(f'\nNext:  python src/make_sheets.py l0 {a.out}')


if __name__ == '__main__':
    main()
