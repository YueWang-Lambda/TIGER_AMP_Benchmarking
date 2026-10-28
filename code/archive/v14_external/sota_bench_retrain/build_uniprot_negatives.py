#!/usr/bin/env python3
"""Build the UniProt non-AMP negative set shared by the three new benchmark methods.

Protocol follows Pandi et al. (Nat Commun 14:7197, 2023), Methods
"Pretraining and training datasets / Non-AMP data":
  - UniProtKB proteins labelled as NOT antimicrobial / antibiotic / antiviral /
    antifungal
  - unique sequences shorter than 49 amino acids
  - entries with ambiguous amino acids removed

Deviation (disclosed): UniProt is queried at its current release (the paper used
the July 2021 snapshot, yielding 10,612 negatives); to keep the class balance of
the paper, the pool is randomly subsampled to 10,612 sequences (seed 42).

Output: baseline/sota_bench/data_restricted/uniprot_negatives.csv  (column:
sequence)
"""
import argparse
import os
import time
import urllib.parse

import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'data_restricted', 'uniprot_negatives.csv')
EXCLUDE_KW = ('antimicrobial', 'antibiotic', 'antifungal', 'antiviral')
CANON = set('ACDEFGHIKLMNPQRSTVWY')

QUERY = 'reviewed:true AND length:[5 TO 48]'
FIELDS = 'accession,length,keyword,sequence'


def fetch_all(proxy: str | None):
    sess = requests.Session()
    if proxy:
        sess.proxies = {'https': proxy, 'http': proxy}
    url = ('https://rest.uniprot.org/uniprotkb/search?query='
           + urllib.parse.quote(QUERY)
           + f'&format=tsv&fields={FIELDS}&size=500')
    rows, seen_acc, n_pages = [], set(), 0
    while url:
        for attempt in range(6):
            try:
                r = sess.get(url, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:  # noqa: BLE001
                print(f'  retry {attempt + 1}: {e}', flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError('UniProt kept failing')
        lines = r.text.strip().split('\n')
        if n_pages == 0:
            header = lines[0].split('\t')
            print(f'total results: {r.headers.get("X-Total-Results")}', flush=True)
        for ln in lines[1:]:
            parts = ln.split('\t')
            if len(parts) != 4:
                continue
            acc, ln_len, kw, seq = parts
            if acc in seen_acc:
                continue
            seen_acc.add(acc)
            rows.append((acc, int(ln_len), kw, seq))
        n_pages += 1
        if n_pages % 10 == 0:
            print(f'  pages={n_pages} entries={len(rows)}', flush=True)
        url = r.links.get('next', {}).get('url')
    return pd.DataFrame(rows, columns=['accession', 'length', 'keyword', 'sequence'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proxy', default='http://127.0.0.1:17892')
    ap.add_argument('--target-n', type=int, default=10612,
                    help='subsample size matching the Pandi et al. paper')
    args = ap.parse_args()

    df = fetch_all(args.proxy or None)
    print(f'fetched {len(df)} reviewed entries 5-48 aa', flush=True)

    kwl = df.keyword.str.lower()
    keep = ~kwl.apply(lambda s: any(k in s for k in EXCLUDE_KW))
    df = df[keep]
    print(f'after NOT antimicrobial/antibiotic/antifungal/antiviral: {len(df)}', flush=True)

    df = df[df.sequence.apply(lambda s: set(s) <= CANON)]
    df['sequence'] = df.sequence.str.upper()
    df = df.drop_duplicates('sequence')
    print(f'after canonical AAs + dedup: {len(df)}', flush=True)

    if len(df) > args.target_n:
        df = df.sample(n=args.target_n, random_state=42).reset_index(drop=True)
        print(f'subsampled to {len(df)} (Pandi et al. class balance)', flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df[['sequence', 'length']].to_csv(OUT, index=False)
    print(f'wrote {OUT}: {len(df)} negatives, len '
          f'{df.length.min()}-{df.length.max()}', flush=True)


if __name__ == '__main__':
    main()
