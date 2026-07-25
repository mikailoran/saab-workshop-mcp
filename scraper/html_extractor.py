"""Extracts the embedded HTML from scraped JSON data files into standalone .html files.

Walks a scraped data folder (as produced by crawl.py) and writes each leaf
document's "html" field out as a formatted .html file under a sibling
`data_html` folder, mirroring the data folder's structure.

Usage:
    python html_extractor.py                # defaults to the data_last_run symlink
    python html_extractor.py data-2026-07-25_14-32-08
"""

import argparse
import json
from pathlib import Path

from bs4 import BeautifulSoup


def extract_html(json_path: Path, out_path: Path) -> None:
    """Extract the "html" field from a scraped JSON document into a formatted .html file.

    Args:
        json_path: path to a JSON file produced by save_leaf().
        out_path: path to write the formatted HTML to; parent dirs are created as needed.
    """
    record = json.loads(json_path.read_text(encoding="utf-8"))
    formatted = BeautifulSoup(record["html"], "html.parser").prettify()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(formatted, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=Path("data_last_run"),
        type=Path,
        help="root folder of scraped JSON data",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    # Output directory is a sibling of the data_dir, named with a "_html" suffix
    out_root = data_dir.with_name(data_dir.name + "_html")

    # Gather all JSON paths under the data_dir, excluding the manifest.json file
    json_paths = [p for p in data_dir.rglob("*.json") if p.name != "manifest.json"]

    # Extract HTML from each JSON file and write to the corresponding .html file
    for json_path in json_paths:
        out_path = out_root / json_path.relative_to(data_dir).with_suffix(".html")
        extract_html(json_path, out_path)
        print(f"{json_path} -> {out_path}")

    print(f"\nExtracted {len(json_paths)} HTML files to: {out_root}")


if __name__ == "__main__":
    main()
