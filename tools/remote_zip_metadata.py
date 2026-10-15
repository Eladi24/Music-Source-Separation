"""
Pulls only the data.json metadata files out of a large remote zip via HTTP
range requests, without downloading the whole archive. Built for MoisesDB
(moisesdb.zip, ~82.7GB on Cloudflare R2) to answer the lead/rhythm-pair
gating question (see CLAUDE.md, "The step change") from a few MB of
transfer instead of the full download.

Works for any S3-compatible host that returns `Accept-Ranges: bytes` and
honors a signed GetObject URL's Range header (confirmed true for R2 here).

Usage:
    python tools/remote_zip_metadata.py "<presigned URL>" --suffix data.json --out /tmp/moisesdb_meta
"""
import argparse
import io
import os
import zipfile

import requests


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable file-like object backed by HTTP range requests.
    Enough for zipfile.ZipFile to read the central directory and individual
    members without ever fetching the whole object."""

    def __init__(self, url, session=None, chunk_log=None):
        self.url = url
        self.session = session or requests.Session()
        self.pos = 0
        self.chunk_log = chunk_log if chunk_log is not None else []
        head = self.session.get(url, headers={"Range": "bytes=0-0"})
        head.raise_for_status()
        content_range = head.headers["Content-Range"]  # "bytes 0-0/<size>"
        self.size = int(content_range.split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        r.raise_for_status()
        data = r.content
        self.chunk_log.append(len(data))
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--suffix", default="data.json", help="only extract files ending with this")
    ap.add_argument("--out", default=None, help="directory to save extracted files (mirrors zip paths)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N matching files (0 = all)")
    args = ap.parse_args()

    chunk_log = []
    f = HTTPRangeFile(args.url, chunk_log=chunk_log)
    print(f"Remote object size: {f.size:,} bytes ({f.size/1e9:.1f} GB)")

    zf = zipfile.ZipFile(f)
    names = zf.namelist()
    print(f"Zip contains {len(names):,} entries")

    matches = [n for n in names if n.endswith(args.suffix)]
    print(f"Entries matching '*{args.suffix}': {len(matches):,}")
    if args.limit:
        matches = matches[: args.limit]

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    bytes_before_members = sum(chunk_log)
    for i, name in enumerate(matches):
        data = zf.read(name)
        if args.out:
            dest = os.path.join(args.out, name.replace("/", "__"))
            with open(dest, "wb") as out:
                out.write(data)
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(matches)} extracted")

    total_bytes = sum(chunk_log)
    print(f"\nDone. Total HTTP bytes transferred: {total_bytes:,} "
          f"({total_bytes/1e6:.2f} MB) out of {f.size/1e9:.1f} GB "
          f"({100*total_bytes/f.size:.6f}% of the archive)")
    print(f"  (central-directory + housekeeping reads: {bytes_before_members:,} bytes "
          f"before the first member was extracted -- rough split, not exact)")


if __name__ == "__main__":
    main()
