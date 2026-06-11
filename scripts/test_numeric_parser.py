import sys
from pathlib import Path

# ensure repository root is on sys.path so text_utils can be imported
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from text_utils import parse_numeric_fields


def test_docs_txt():
    files = list(Path("docs").glob("*.txt"))
    if not files:
        print("No txt files under docs/")
        return

    for f in files:
        print(f"\n=== {f} ===")
        text = f.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs[:10]:
            fields, found = parse_numeric_fields(p)
            if found:
                print("PARAGRAPH:", p[:200])
                print("PARSED:", fields)


if __name__ == '__main__':
    test_docs_txt()
