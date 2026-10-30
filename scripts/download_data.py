"""Download only LongKou MAT entries from the official WHU archive.

Uses HTTP ranges to avoid downloading the other two scenes. No third-party mirror,
registration submission, scientific dependencies, or training is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://rsidea.whu.edu.cn/code_data/WHU-Hi/Matlab_data_format.zip"


class RemoteZip(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
        self.position = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("Negative archive offset")
        return self.position

    def request_range(self, start, end):
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(request, timeout=120)
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
            response.close()
            raise OSError("Official server did not honor HTTP ranges; download the ZIP manually.")
        return response

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        with self.request_range(self.position, self.position + size - 1) as response:
            data = response.read()
        if len(data) != size:
            raise OSError("Truncated archive range")
        self.position += len(data)
        return data


def download(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    remote = RemoteZip(URL)
    manifest = {"source_page": "https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm",
                "form": "https://rsidea-whu-hi.mikecrm.com/MqhjRPN", "archive_url": URL,
                "archive_bytes": remote.size, "etag": remote.etag, "files": []}
    with zipfile.ZipFile(remote) as archive:
        entries = [x for x in archive.infolist() if "longkou" in x.filename.lower()
                   and x.filename.lower().endswith(".mat")]
        # Exclude the supplied fixed train/test subsets: this project makes its own splits.
        entries = [x for x in entries if not re.search(r"(?:train|test)\d", x.filename, re.I)]
        print("LongKou entries:", [(x.filename, x.file_size) for x in entries], flush=True)
        if not entries:
            raise RuntimeError("No LongKou MAT files found in official archive")
        for entry in entries:
            filename = entry.filename.replace("\\", "/").split("/")[-1]
            target = destination / filename
            if target.exists():
                crc, written, digest = 0, 0, hashlib.sha256()
                with target.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        written += len(chunk)
                        crc = zlib.crc32(chunk, crc)
                        digest.update(chunk)
                if written != entry.file_size or crc & 0xffffffff != entry.CRC:
                    raise FileExistsError(f"Refusing to overwrite mismatched file {target}; inspect or move it first.")
                manifest["files"].append({"path": filename, "archive_entry": entry.filename,
                                          "bytes": written, "sha256": digest.hexdigest(), "crc32": entry.CRC})
                (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                print(f"Verified existing {target}; skipping transfer", flush=True)
                continue
            remote.seek(entry.header_offset)
            header = remote.read(30)
            if header[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            name_len, extra_len = struct.unpack_from("<HH", header, 26)
            start = entry.header_offset + 30 + name_len + extra_len
            if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError(f"Unsupported compression: {entry.compress_type}")
            decoder = zlib.decompressobj(-15) if entry.compress_type == zipfile.ZIP_DEFLATED else None
            crc, written, digest = 0, 0, hashlib.sha256()
            temporary = target.with_suffix(target.suffix + ".part")
            print(f"Downloading {filename} ({entry.compress_size / 1e6:.1f} MB transferred)", flush=True)
            with remote.request_range(start, start + entry.compress_size - 1) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    data = decoder.decompress(chunk) if decoder else chunk
                    output.write(data)
                    written += len(data)
                    crc = zlib.crc32(data, crc)
                    digest.update(data)
                if decoder:
                    data = decoder.flush()
                    output.write(data)
                    written += len(data)
                    crc = zlib.crc32(data, crc)
                    digest.update(data)
            if written != entry.file_size or crc & 0xffffffff != entry.CRC:
                raise OSError(f"ZIP size/CRC verification failed: {temporary}")
            temporary.replace(target)
            manifest["files"].append({"path": filename, "archive_entry": entry.filename,
                                      "bytes": written, "sha256": digest.hexdigest(), "crc32": entry.CRC})
            (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Verified {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data" / "raw")
    download(parser.parse_args().destination)
