#!/usr/bin/env python3
"""
Download newspaper-issue PDFs from Det Kgl. Bibliotek Mediestream.

Mediestream exposes one PDF per newspaper *issue* (edition/day), not a single
merged PDF per year. This script discovers all issues for each year via the
same JSON APIs the website uses, then saves PDFs under:

  <output_dir>/<year>/<date>_<edition_id>.pdf

Example (Dansk Vestindisk Regierings Avis):
  python download_mediastream_pdfs.py \\
    --paper-id 'doms_newspaperAuthority:uuid:46272a0e-0320-4b64-9d37-8fe6be52b988'

Use --years 1816,1817 to limit scope, or --dry-run to list without downloading.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = "https://www2.statsbiblioteket.dk/mediestream/"
DEFAULT_PAPER_ID = (
    "doms_newspaperAuthority:uuid:46272a0e-0320-4b64-9d37-8fe6be52b988"
)
USER_AGENT = "history_nlp-mediastream-downloader/1.0 (+research use; respectful rate limit)"


class MediestreamClient:
    def __init__(self, base_url: str = BASE_URL, delay: float = 0.5) -> None:
        self.base_url = base_url if base_url.endswith("/") else base_url + "/"
        self.delay = delay
        self._opener = urllib.request.build_opener()
        self._opener.addheaders = [("User-Agent", USER_AGENT), ("Accept", "application/json")]

    def _get_json(self, path: str, params: dict[str, str] | None = None) -> Any:
        url = urllib.parse.urljoin(self.base_url, path.lstrip("/"))
        if params:
            url += "?" + urllib.parse.urlencode(params)
        time.sleep(self.delay)
        req = urllib.request.Request(url)
        with self._opener.open(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _download(self, url: str, dest: Path) -> None:
        time.sleep(self.delay)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with self._opener.open(req, timeout=300) as resp:
            data = resp.read()
        dest.write_bytes(data)

    def list_years(self, paper_id: str) -> list[dict[str, Any]]:
        data = self._get_json(
            f"services/newspapers/calendar/title/{paper_id}",
            {"locale": "en"},
        )
        return [row for row in data if row.get("hasData")]

    def title_uuid_for_year(self, paper_id: str, year: int) -> str:
        data = self._get_json(
            f"services/newspapers/calendar/title/{paper_id}/{year}",
            {"locale": "en"},
        )
        title_uuid = data.get("titleUUID")
        if not title_uuid:
            raise ValueError(f"No titleUUID returned for year {year}")
        return title_uuid

    def editions_for_year(self, title_uuid: str, year: int, perpage: int = 50) -> list[str]:
        query = (
            f'iso_dateTime:[{year}-01-01T00:00:00Z TO {year}-12-31T23:59:59Z] '
            f'titleUUID:"{title_uuid}"'
        )
        editions: list[str] = []
        seen: set[str] = set()
        start = 0
        while True:
            params = {
                "query": query,
                "site": "avis",
                "allowedonly": "false",
                "sort": "date",
                "start": str(start),
                "perpage": str(perpage),
            }
            data = self._get_json("services/search/", params)
            groups = data.get("groups") or []
            for group in groups:
                edition_id = group.get("groupValue")
                if edition_id and edition_id not in seen:
                    seen.add(edition_id)
                    editions.append(edition_id)
            if len(groups) < perpage:
                break
            start += perpage
        return editions

    def record_pdf(self, edition_id: str) -> tuple[str | None, dict[str, Any]]:
        params = {
            "id": edition_id,
            "type": "avis",
        }
        data = self._get_json("services/record/", params)
        if isinstance(data, list):
            if not data:
                return None, {}
            record = data[0]
        else:
            record = data
        content = record.get("contentInfo") or {}
        download = content.get("Download") or {}
        if not download.get("mataccess"):
            return None, record
        urls = download.get("url") or []
        if not urls:
            return None, record
        return urls[0], record


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:180] if name else "unknown"


def pdf_dest_path(output_dir: Path, record: dict[str, Any], edition_id: str, pdf_url: str) -> Path:
    year = (record.get("date") or "unknown")[:4]
    date = record.get("date") or "unknown-date"
    parsed = urllib.parse.urlparse(pdf_url)
    qs = urllib.parse.parse_qs(parsed.query)
    filename = None
    if "filename" in qs and qs["filename"]:
        filename = safe_filename(qs["filename"][0])
    if not filename:
        short_id = edition_id.rsplit(":", 1)[-1][:12]
        filename = safe_filename(f"{date}_{short_id}.pdf")
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    return output_dir / year / filename


def parse_years(spec: str | None, available: list[int]) -> list[int]:
    if not spec:
        return available
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.update(range(int(a), int(b) + 1))
        else:
            wanted.add(int(part))
    return [y for y in available if y in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Mediestream newspaper issue PDFs by year."
    )
    parser.add_argument(
        "--paper-id",
        default=DEFAULT_PAPER_ID,
        help="Newspaper authority ID from the Mediestream URL (default: Curacao paper).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mediastream_pdfs"),
        help="Directory for downloaded PDFs (default: mediestream_pdfs).",
    )
    parser.add_argument(
        "--years",
        help="Comma-separated years and/or ranges, e.g. 1816,1820-1825. Default: all years with data.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between HTTP requests (default: 0.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List editions and PDF URLs without downloading.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: on).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-download even if the output file exists.",
    )
    args = parser.parse_args()

    client = MediestreamClient(delay=args.delay)
    years_data = client.list_years(args.paper_id)
    available_years = [int(row["year"]) for row in years_data]
    selected_years = parse_years(args.years, available_years)

    if not selected_years:
        print("No years selected (check --years).", file=sys.stderr)
        return 1

    print(f"Paper: {args.paper_id}")
    print(f"Years: {selected_years[0]}–{selected_years[-1]} ({len(selected_years)} years)")
    print(f"Output: {args.output_dir.resolve()}")

    total_downloaded = 0
    total_skipped = 0
    total_denied = 0
    total_failed = 0

    for year in selected_years:
        print(f"\n=== {year} ===")
        try:
            title_uuid = client.title_uuid_for_year(args.paper_id, year)
        except (urllib.error.URLError, ValueError) as exc:
            print(f"  Could not load calendar for {year}: {exc}", file=sys.stderr)
            total_failed += 1
            continue

        try:
            editions = client.editions_for_year(title_uuid, year)
        except urllib.error.URLError as exc:
            print(f"  Search failed for {year}: {exc}", file=sys.stderr)
            total_failed += 1
            continue

        print(f"  {len(editions)} issue(s)")
        for i, edition_id in enumerate(editions, 1):
            try:
                pdf_url, record = client.record_pdf(edition_id)
            except urllib.error.URLError as exc:
                print(f"  [{i}/{len(editions)}] record error: {edition_id}: {exc}")
                total_failed += 1
                continue

            if not pdf_url:
                print(f"  [{i}/{len(editions)}] no PDF access: {edition_id}")
                total_denied += 1
                continue

            dest = pdf_dest_path(args.output_dir, record, edition_id, pdf_url)
            if args.skip_existing and dest.exists() and dest.stat().st_size > 0:
                print(f"  [{i}/{len(editions)}] exists: {dest.name}")
                total_skipped += 1
                continue

            if args.dry_run:
                print(f"  [{i}/{len(editions)}] would download -> {dest}")
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                client._download(pdf_url, dest)
                print(f"  [{i}/{len(editions)}] saved: {dest}")
                total_downloaded += 1
            except urllib.error.URLError as exc:
                print(f"  [{i}/{len(editions)}] download failed: {dest.name}: {exc}")
                total_failed += 1

    print(
        "\nDone."
        f" downloaded={total_downloaded}"
        f" skipped={total_skipped}"
        f" no_access={total_denied}"
        f" failed={total_failed}"
    )
    return 0 if total_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
