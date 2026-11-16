"""Download and unpack CIFAR-10-C (Zenodo record 2535967, CC BY 4.0, ~2.9 GB) into data_cache/.

Usage:  python scripts/download_cifar10c.py [--dest data_cache]
Afterwards the .npy files are in <dest>/CIFAR-10-C/ (this is `data.root` in the configs).
The md5 below is copied from the Zenodo record page; the script refuses to unpack on a mismatch.
If Zenodo is unreachable, the fallback is generating corruptions with the `imagecorruptions` package
(not implemented in Phase 3; tell the mentor).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1"
MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data_cache")
    ap.add_argument("--keep-tar", action="store_true")
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "CIFAR-10-C.tar"
    if not tar_path.exists():
        print(f"downloading {URL}")

        def hook(blocks: int, bsize: int, total: int) -> None:
            if total > 0 and blocks % 200 == 0:
                print(f"\r{min(100.0, 100.0 * blocks * bsize / total):5.1f}%", end="", flush=True)

        urllib.request.urlretrieve(URL, tar_path, hook)
        print()
    got = md5_of(tar_path)
    if got != MD5:
        sys.exit(f"md5 mismatch for {tar_path}: got {got}, expected {MD5}. Delete the file and retry.")
    print("md5 ok; unpacking .npy files")
    with tarfile.open(tar_path) as tar:
        for member in tar:
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts:       # never trust paths inside an archive
                sys.exit(f"unsafe path in archive: {member.name}")
            if member.isfile() and member.name.endswith(".npy"):
                tar.extract(member, dest)
    root = dest / "CIFAR-10-C"
    print(f"done: {len(list(root.glob('*.npy')))} .npy files in {root}")
    if not args.keep_tar:
        tar_path.unlink()


if __name__ == "__main__":
    main()
