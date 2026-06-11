import re
from typing import Tuple, Dict, Any


def parse_numeric_fields(text: str) -> Tuple[Dict[str, Any], bool]:
    """Extract common numeric fields from a text snippet.

    Returns a tuple (fields_dict, found_flag).
    """
    fields = {}
    found = False

    # currency, e.g. $8,925,000 or USD 8,925,000
    for m in re.finditer(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)", text, flags=re.I):
        val = m.group(1).replace(",", "")
        try:
            fields.setdefault("currency_values", []).append(float(val))
            found = True
        except Exception:
            pass

    # plain numbers with commas (likely units or revenue without $)
    for m in re.finditer(r"(?<!\S)([0-9]{1,3}(?:,[0-9]{3})+)(?!\S)", text):
        val = m.group(1).replace(",", "")
        try:
            fields.setdefault("int_values", []).append(int(val))
            found = True
        except Exception:
            pass

    # percentages
    for m in re.finditer(r"([+-]?\d+(?:\.\d+)?)%", text):
        try:
            fields.setdefault("percent_values", []).append(float(m.group(1)))
            found = True
        except Exception:
            pass

    # specific 'days' mentions (stockout: 3 days)
    for m in re.finditer(r"(\d+)\s+days?", text, flags=re.I):
        try:
            fields.setdefault("days", []).append(int(m.group(1)))
            found = True
        except Exception:
            pass

    # units sold pattern
    for m in re.finditer(r"units\s*sold[:\s]*([0-9][0-9,]*)", text, flags=re.I):
        val = m.group(1).replace(",", "")
        try:
            fields.setdefault("units_sold", []).append(int(val))
            found = True
        except Exception:
            pass

    return fields, found
