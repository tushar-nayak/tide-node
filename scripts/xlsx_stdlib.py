"""Minimal .xlsx reader using only the standard library (the project venv has no openpyxl).

read(path) -> {sheet name: [row, ...]}, each row a list of cell strings ('' when empty).
Numbers come back as their stored text; dates are not converted (MU-Glioma-Post stores
day counts, not dates).
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
       "p": "http://schemas.openxmlformats.org/package/2006/relationships"}
_MAX_UNCOMPRESSED = 200_000_000


def _column(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def _text(node) -> str:
    return "".join(t.text or "" for t in node.iter("{%s}t" % _NS["m"]))


def read(path: Path | str) -> dict[str, list[list[str]]]:
    z = zipfile.ZipFile(path)
    if sum(i.file_size for i in z.infolist()) > _MAX_UNCOMPRESSED:
        raise ValueError(f"{path}: archive too large")
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        shared = [_text(si) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS)]
    rels = {r.get("Id"): r.get("Target")
            for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).findall("p:Relationship", _NS)}
    out = {}
    for sheet in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", _NS).findall("m:sheet", _NS):
        target = rels[sheet.get("{%s}id" % _NS["r"])].lstrip("/")
        target = target if target.startswith("xl/") else "xl/" + target
        rows = []
        for row in ET.fromstring(z.read(target)).iter("{%s}row" % _NS["m"]):
            cells = {}
            for c in row.findall("m:c", _NS):
                kind, v = c.get("t"), c.find("m:v", _NS)
                if kind == "s" and v is not None:
                    value = shared[int(v.text)]
                elif kind == "inlineStr":
                    value = _text(c)
                else:
                    value = v.text if v is not None and v.text is not None else ""
                cells[_column(c.get("r"))] = value
            if cells:
                rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
        out[sheet.get("name")] = rows
    return out


def records(path: Path | str, sheet: str) -> list[dict[str, str]]:
    """Rows of one sheet as dicts keyed by the stripped header."""
    rows = read(path)[sheet]
    header = [h.strip() for h in rows[0]]
    return [dict(zip(header, r + [""] * (len(header) - len(r)))) for r in rows[1:]]
