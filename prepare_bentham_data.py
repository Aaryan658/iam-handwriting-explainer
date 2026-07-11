"""Fetch N Bentham Papers (ImageCLEF 2016 / Zenodo 52994) line images + their
transcriptions via remotezip (no full multi-GB zip download needed), and
write train/val CSVs in the image_path,text format finetune_trocr.py expects.

Train split lines come from line_images_train.zip / transcript_train.txt.
Val split lines come from line_images_devel.zip / transcript_devel.txt (a
different set of documents, for a genuinely held-out eval per the fine-tuning
recipe's "ideally cross-document" note).

This machine's network runs Sophos TLS inspection whose intercepting CA cert
has a malformed Basic Constraints extension; Python's default strict X509
verification (VERIFY_X509_STRICT) rejects it even though the cert chain is
otherwise trusted (same underlying issue as ocr_engines.py's EasyOCR
workaround). requests/urllib3 build their own SSLContext rather than going
through ssl._create_default_https_context, so the fix here is a dedicated
requests.Session with a custom HTTPAdapter carrying a relaxed SSLContext,
rather than any global ssl module patching.
"""
import argparse
import csv
import os
import random
import ssl

import certifi
import requests
from requests.adapters import HTTPAdapter
from remotezip import RemoteZip

ZENODO_BASE = "https://zenodo.org/records/52994/files"

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SOPHOS_CA = os.path.join(_PROJECT_ROOT, "sophos_ca.pem")
_COMBINED_CA = os.path.join(_PROJECT_ROOT, "sophos_ca_combined.pem")


def _build_combined_ca_bundle():
    with open(_COMBINED_CA, "w", encoding="utf-8") as out, \
         open(certifi.where(), encoding="utf-8") as certifi_bundle, \
         open(_SOPHOS_CA, encoding="utf-8") as sophos_cert:
        out.write(certifi_bundle.read())
        out.write("\n")
        out.write(sophos_cert.read())


class RelaxedSSLAdapter(HTTPAdapter):
    """Trusts the combined CA bundle but relaxes VERIFY_X509_STRICT, since
    the Sophos intercepting cert has a malformed (non-critical) Basic
    Constraints extension that OpenSSL's strict mode otherwise rejects."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context(cafile=_COMBINED_CA)
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def make_session():
    if os.path.exists(_SOPHOS_CA):
        _build_combined_ca_bundle()
        session = requests.Session()
        session.mount("https://", RelaxedSSLAdapter())
        return session
    return requests.Session()


def load_transcripts(session, url):
    import zipfile
    import io

    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        inner_name = z.namelist()[0]
        text = z.read(inner_name).decode("utf-8")

    transcripts = {}
    for line in text.splitlines():
        lid, _, txt = line.partition(" ")
        if txt and "[" not in txt and "]" not in txt:
            transcripts[lid] = txt
    return transcripts


def _download_full_zip(session, split):
    """Download the whole line_images_<split>.zip once (cached under
    data/_zip_cache/), rather than issuing one HTTP range request per line --
    Zenodo rate-limits (429) after a couple hundred of those in a row."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "_zip_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"line_images_{split}.zip")
    if os.path.exists(cache_path):
        return cache_path

    url = f"{ZENODO_BASE}/line_images_{split}.zip"
    print(f"Downloading {url} (one-time, cached under {cache_dir})...")
    with session.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(cache_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    return cache_path


def fetch_split(session, split, n, out_dir):
    import zipfile

    transcripts = load_transcripts(session, f"{ZENODO_BASE}/transcript_{split}.txt.zip")
    ids = list(transcripts.keys())
    random.Random(42).shuffle(ids)
    chosen = ids[:n]

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    zip_path = _download_full_zip(session, split)
    with zipfile.ZipFile(zip_path) as z:
        for lid in chosen:
            filename = f"{int(lid):06d}.png"
            data = z.read(f"line_images_{split}/{filename}")
            out_path = os.path.join(out_dir, filename)
            with open(out_path, "wb") as f:
                f.write(data)
            rows.append((filename, transcripts[lid]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=10)
    parser.add_argument("--out-dir", default="data/bentham")
    args = parser.parse_args()

    session = make_session()

    print(f"Fetching {args.n_train} train lines...")
    train_rows = fetch_split(session, "train", args.n_train, os.path.join(args.out_dir, "train"))
    print(f"Fetching {args.n_eval} eval lines...")
    eval_rows = fetch_split(session, "devel", args.n_eval, os.path.join(args.out_dir, "val"))

    with open(os.path.join(args.out_dir, "train.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text"])
        writer.writerows((f"train/{fn}", txt) for fn, txt in train_rows)

    with open(os.path.join(args.out_dir, "val.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text"])
        writer.writerows((f"val/{fn}", txt) for fn, txt in eval_rows)

    print(f"Wrote {len(train_rows)} train rows, {len(eval_rows)} eval rows to {args.out_dir}")


if __name__ == "__main__":
    main()
