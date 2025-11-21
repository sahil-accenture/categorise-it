"""
classify_reels.py (refactored read_dataset)

Usage examples:
  python classify_reels.py --input data/dataset_instagram-reel.csv --topics data/Topics.txt
  python classify_reels.py -i data/dataset_instagram-reel.csv -t data/Topics.txt --encoding latin-1 --skip-bad-lines --show-sample

Requirements:
  pip install pandas requests beautifulsoup4 scikit-learn numpy
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Config
REQUEST_TIMEOUT = 10.0
MAX_WORKERS = 6
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)
MAX_EXTRACT_CHARS = 40_000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classify_reels")


def try_open_text_file(path: str, enc: Optional[str]) -> List[str]:
    enc_list = [enc] if enc else ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    for e in enc_list:
        try:
            with open(path, "r", encoding=e, errors="strict") as f:
                lines = [ln.rstrip("\n\r") for ln in f]
            logger.info("Opened %s with encoding=%s", path, e)
            return lines
        except Exception as exc:
            logger.debug("Failed reading %s with encoding=%s: %s", path, e, exc)
    logger.warning("All strict encodings failed for %s. Using 'utf-8' with errors='replace'.", path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n\r") for ln in f]
    return lines


def load_topics(path: str, encoding: Optional[str] = None) -> List[str]:
    lines = try_open_text_file(path, encoding)
    topics = [ln.strip() for ln in lines if ln.strip()]
    if not topics:
        raise ValueError(f"No topics found in {path}. Please place one topic per line.")
    logger.info("Loaded %d topics from %s", len(topics), path)
    if len(topics) != 24:
        logger.warning("Topics count is %d (expected 24). Proceeding with loaded topics.", len(topics))
    return topics


def _sniff_delimiter(sample_text: str) -> str:
    # Try csv.Sniffer first; fallback to heuristic counts
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text, delimiters=[",", ";", "\t", "|", ":"])
        logger.info("csv.Sniffer detected delimiter=%r", dialect.delimiter)
        return dialect.delimiter
    except Exception:
        candidates = [",", "\t", "|", ";", ":"]
        counts = {d: sample_text.count(d) for d in candidates}
        best = max(counts, key=counts.get)
        logger.info("Sniffer failed; heuristic delimiter=%r (counts=%s)", best, counts)
        return best

def read_dataset(path: str, encoding: Optional[str] = None, skip_bad_lines: bool = False) -> pd.DataFrame:
    """
    Robust loader that:
     - Detects if the file is CSV, XLSX, ZIP (containing CSV), or gzipped CSV
     - If CSV-like: parse robustly (csv.reader fallback)
     - If XLSX: read with pandas.read_excel
     - If ZIP: try to find the first CSV inside and read it
     - Normalize columns to: caption, ownerFullName, ownerUsername, url, timestamp
    """
    import csv
    import zipfile
    import gzip
    from pathlib import Path
    p = Path(path)
    raw = p.read_bytes()
    # quick magic detection
    head = raw[:4]
    is_zip = head.startswith(b"PK\x03\x04")
    is_gzip = head.startswith(b"\x1f\x8b")
    # try candidate encodings for CSV decode
    candidate_encs = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
    def parse_csv_from_text(text: str) -> pd.DataFrame:
        # robust CSV parsing fallback (use pandas then csv.reader fallback)
        try:
            df_try = pd.read_csv(StringIO(text), sep=None, engine="python", dtype=str, keep_default_na=False, low_memory=False,
                                 quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip' if skip_bad_lines else 'warn')
            if len(df_try.columns) >= 5 and len(df_try) > 0:
                logger.info("pandas successfully parsed CSV-like content with %d rows and %d columns", len(df_try), len(df_try.columns))
                return df_try
        except Exception as e:
            logger.debug("pandas.read_csv auto-detect failed: %s", e)
        # fallback: try sniff delimiter + csv.reader with quote-repair
        sample = "\n".join(text.splitlines()[:500])
        candidates = [",", "\t", "|", ";", ":"]
        counts = {d: sample.count(d) for d in candidates}
        delim = max(counts, key=counts.get)
        logger.info("Fallback CSV parse using delimiter=%r (counts=%s)", delim, counts)
        phys_lines = text.splitlines()
        # repair by quote balancing
        logical_lines = []
        buf = []
        for ln in phys_lines:
            buf.append(ln)
            joined = "\n".join(buf)
            if joined.count('"') % 2 == 0:
                logical_lines.append(joined)
                buf = []
        if buf:
            logical_lines.append("\n".join(buf))
            logger.warning("Trailing unbalanced buffer flushed as last logical record.")
        reader = csv.reader(logical_lines, delimiter=delim, quotechar='"', doublequote=True)
        rows = []
        for r in reader:
            rows.append(r)
        # normalize to 5 cols
        expected = 5
        norm = []
        for r in rows:
            if len(r) < expected:
                r = r + [""] * (expected - len(r))
            elif len(r) > expected:
                merged = delim.join(r[expected - 1 :])
                r = r[: expected - 1] + [merged]
            r = [str(cell).replace("\r", " ").replace("\n", " ").replace('""', '"').strip() for cell in r[:expected]]
            norm.append(r)
        cols = ["caption", "ownerFullName", "ownerUsername", "url", "timestamp"]
        df = pd.DataFrame(norm, columns=cols)
        return df

    # 1) If ZIP (possibly .xlsx or zipped CSV)
    if is_zip:
        logger.info("File %s seems to be a ZIP archive (PK signature). Trying to inspect inside...", path)
        try:
            with zipfile.ZipFile(p) as z:
                # list files: prefer .csv, then .xlsx
                names = z.namelist()
                logger.info("ZIP contents: %s", names[:10])
                # priority: csv files
                csv_candidates = [n for n in names if n.lower().endswith(".csv")]
                xlsx_candidates = [n for n in names if n.lower().endswith((".xlsx", ".xlsm", ".xls"))]
                if csv_candidates:
                    logger.info("Found CSV inside ZIP: %s. Reading first match.", csv_candidates[0])
                    with z.open(csv_candidates[0]) as fh:
                        raw_bytes = fh.read()
                        # try decode and parse
                        for e in candidate_encs:
                            try:
                                text = raw_bytes.decode(e)
                                df = parse_csv_from_text(text)
                                logger.info("Parsed CSV from ZIP with encoding=%s", e)
                                return _normalize_columns(df)
                            except Exception:
                                continue
                        # fallback decode replace
                        text = raw_bytes.decode("utf-8", errors="replace")
                        df = parse_csv_from_text(text)
                        return _normalize_columns(df)
                elif xlsx_candidates:
                    # if XLSX inside ZIP (likely this file actually is XLSX itself) -> read that file with pandas
                    logger.info("Found XLSX inside ZIP: %s. Reading first match with pandas.read_excel.", xlsx_candidates[0])
                    with z.open(xlsx_candidates[0]) as fh:
                        # pandas can read file-like object via BytesIO
                        from io import BytesIO
                        bio = BytesIO(fh.read())
                        df = pd.read_excel(bio, dtype=str, engine="openpyxl" if "openpyxl" in sys.modules else None)
                        return _normalize_columns(df)
                else:
                    logger.warning("No CSV/XLSX files inside ZIP. Will try to parse raw bytes as CSV fallback.")
        except Exception as e:
            logger.debug("ZIP handling failed: %s", e)
        # if zip handling fails, fallthrough to checking xlsx via pandas

    # 2) If gzip compressed
    if is_gzip:
        logger.info("File is gzip-compressed. Decompressing and attempting CSV parse.")
        try:
            with gzip.open(p, "rt", encoding=encoding or "utf-8", errors="replace") as fh:
                text = fh.read()
            df = parse_csv_from_text(text)
            return _normalize_columns(df)
        except Exception as e:
            logger.debug("gzip handling failed: %s", e)

    # 3) Try reading as XLSX (Excel)
    # many Excel files start with PK as well (xlsx). Try pandas.read_excel
    try:
        logger.info("Attempting to read file with pandas.read_excel (Excel).")
        try:
            df_excel = pd.read_excel(p, dtype=str, engine="openpyxl")
        except Exception:
            df_excel = pd.read_excel(p, dtype=str)
        logger.info("pandas.read_excel returned %d rows and %d cols", len(df_excel), len(df_excel.columns))
        return _normalize_columns(df_excel)
    except Exception as e:
        logger.debug("read_excel failed: %s", e)

    # 4) Otherwise, try decode as CSV text using candidate encodings
    for e in candidate_encs:
        try:
            text = raw.decode(e)
            df = parse_csv_from_text(text)
            logger.info("Parsed file as CSV with encoding=%s", e)
            return _normalize_columns(df)
        except Exception:
            continue
    # final fallback: decode utf-8 replace
    text = raw.decode("utf-8", errors="replace")
    df = parse_csv_from_text(text)
    return _normalize_columns(df)

# helper to normalize column names / types
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["caption", "ownerFullName", "ownerUsername", "url", "timestamp"]
    # map likely header names to required names
    mapping = {}
    for c in list(df.columns):
        lc = str(c).lower().replace(" ", "").replace("_", "")
        if "caption" in lc:
            mapping[c] = "caption"
        elif "ownerfullname" in lc or "ownerfull" in lc or "ownername" in lc:
            mapping[c] = "ownerFullName"
        elif "ownerusername" in lc or "username" in lc:
            mapping[c] = "ownerUsername"
        elif lc in ("url", "link"):
            mapping[c] = "url"
        elif "time" in lc or "timestamp" in lc or "date" in lc:
            mapping[c] = "timestamp"
    if mapping:
        df = df.rename(columns=mapping)
    # if still missing, take first five columns by position
    cols_now = list(df.columns)
    if not set(required).issubset(set(cols_now)):
        if len(cols_now) >= 5:
            df = df.rename(columns={cols_now[0]: "caption", cols_now[1]: "ownerFullName", cols_now[2]: "ownerUsername", cols_now[3]: "url", cols_now[4]: "timestamp"})
        else:
            # pad with empty columns
            for i, name in enumerate(required):
                if name not in df.columns:
                    df[name] = ""
            df = df[required]
    # ensure types
    for c in required:
        df[c] = df[c].fillna("").astype(str)
    return df[required]

def fetch_url_text(url: str) -> str:
    if not url or url.strip() == "":
        return ""
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        logger.debug("Request error for %s: %s", url, e)
        return ""
    if resp.status_code != 200 or not resp.content:
        logger.debug("Non-200 or empty for %s: %s", url, resp.status_code)
        return ""
    try:
        soup = BeautifulSoup(resp.content, "html.parser")
    except Exception:
        return ""
    parts = []
    # meta
    og = soup.find("meta", property="og:description")
    if og and og.get("content"):
        parts.append(og.get("content").strip())
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        parts.append(md.get("content").strip())
    tw = soup.find("meta", attrs={"name": "twitter:description"})
    if tw and tw.get("content"):
        parts.append(tw.get("content").strip())
    if soup.title and soup.title.string:
        parts.append(soup.title.string.strip())
    # ld+json
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string
        if not raw:
            continue
        try:
            j = json.loads(raw)
        except Exception:
            try:
                cleaned = re.sub(r",\s*}", "}", raw)
                j = json.loads(cleaned)
            except Exception:
                continue
        def extract_from_obj(o):
            if isinstance(o, dict):
                for k in ("caption", "description", "transcript", "text"):
                    v = o.get(k)
                    if isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                for v in o.values():
                    extract_from_obj(v)
            elif isinstance(o, list):
                for it in o:
                    extract_from_obj(it)
        extract_from_obj(j)
    # scripts with embedded JSON
    for s in soup.find_all("script"):
        txt = s.string or ""
        if not txt:
            continue
        if "window._sharedData" in txt:
            m = re.search(r"window\._sharedData\s*=\s*({.*?});", txt, flags=re.S)
            if m:
                try:
                    jd = json.loads(m.group(1))
                    def deep_find(o):
                        if isinstance(o, dict):
                            for k, v in o.items():
                                if k in ("text", "caption", "description") and isinstance(v, str):
                                    parts.append(v)
                                else:
                                    deep_find(v)
                        elif isinstance(o, list):
                            for it in o:
                                deep_find(it)
                    deep_find(jd)
                except Exception:
                    pass
    visible = " ".join(soup.stripped_strings)
    if visible:
        parts.append(visible[:MAX_EXTRACT_CHARS])
    combined = " \n ".join(dict.fromkeys([p for p in parts if p]))
    combined = re.sub(r"\s+", " ", combined).strip()
    if len(combined) > MAX_EXTRACT_CHARS:
        combined = combined[:MAX_EXTRACT_CHARS]
    return combined


def build_combined_text(row: pd.Series, fetched_text: str) -> str:
    parts = []
    caption = (row.get("caption") or "").strip()
    if caption:
        parts.append(caption)
    if row.get("ownerUsername"):
        parts.append(str(row["ownerUsername"]))
    if row.get("ownerFullName"):
        parts.append(str(row["ownerFullName"]))
    if row.get("timestamp"):
        parts.append(str(row["timestamp"]))
    if fetched_text:
        parts.append(fetched_text)
    combined = " \n ".join([p for p in parts if p])
    combined = re.sub(r"\s+", " ", combined).strip()
    return combined


def predict_topics(texts: List[str], topics: List[str]) -> List[str]:
    corpus = texts + topics
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=40000)
    X = vectorizer.fit_transform(corpus)
    X_texts = X[: len(texts)]
    X_topics = X[len(texts) :]
    sim = cosine_similarity(X_texts, X_topics)
    best_idx = np.argmax(sim, axis=1)
    preds = [topics[i] for i in best_idx]
    return preds


def main(args):
    topics = load_topics(args.topics, encoding=args.encoding)
    df = read_dataset(args.input, encoding=args.encoding, skip_bad_lines=args.skip_bad_lines)
    logger.info("Input rows: %d", len(df))

    urls = df["url"].tolist()
    fetched_texts = [""] * len(df)
    logger.info("Fetching textual content from %d URLs (max_workers=%d)...", len(urls), MAX_WORKERS)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(fetch_url_text, url): i for i, url in enumerate(urls)}
        for future in as_completed(future_map):
            i = future_map[future]
            try:
                fetched_texts[i] = future.result()
            except Exception as exc:
                logger.debug("fetch error index=%d: %s", i, exc)
            if args.delay_between_requests and args.delay_between_requests > 0:
                time.sleep(args.delay_between_requests)

    combined_texts = []
    for i, row in df.iterrows():
        combined = build_combined_text(row, fetched_texts[i])
        if not combined:
            combined = str(row.get("caption") or row.get("ownerUsername") or "")
        combined_texts.append(combined)

    predictions = predict_topics(combined_texts, topics)
    df["predicted_niche"] = predictions

    out_path = args.output or args.input.replace(".csv", "_with_predictions.csv")
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows with predictions to %s", len(df), out_path)
    if args.show_sample:
        print(df[["caption", "url", "predicted_niche"]].head(20).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify Instagram reels into provided topics.")
    parser.add_argument("--input", "-i", required=True, help="Path to dataset_instagram-reel.csv")
    parser.add_argument("--topics", "-t", required=True, help="Path to Topics.txt (one topic per line)")
    parser.add_argument("--output", "-o", required=False, help="Optional output CSV path")
    parser.add_argument("--encoding", "-e", required=False, help="Optional encoding to try (e.g. latin-1, cp1252)")
    parser.add_argument("--skip-bad-lines", action="store_true", help="Skip malformed CSV lines that can't be tokenized")
    parser.add_argument("--delay-between-requests", type=float, default=0.0, help="Delay between URL fetches (seconds)")
    parser.add_argument("--show-sample", action="store_true", help="Print a sample of predictions")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
