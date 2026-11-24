"""Render final PDFs and record structural / layout checks for local review."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = {}
    for name in ("Seedance25", "StabelDiffusion"):
        output = ROOT / "build/review" / name
        output.mkdir(parents=True, exist_ok=True)
        for pattern in ("page-*.png", "contact-*.png"):
            for previous in output.glob(pattern):
                previous.unlink()
        pdf = fitz.open(ROOT / (name + ".pdf"))
        text = "\n".join(p.get_text() for p in pdf)
        log = (ROOT / "build" / name / (name + ".log")).read_text(encoding="utf-8", errors="replace")
        report = {
            "pages": len(pdf), "bytes": (ROOT / (name + ".pdf")).stat().st_size,
            "module_presence": {f"M{i:02d}": f"M{i:02d}" in text for i in range(28)},
            "errors": re.findall(r"^.*(?:LaTeX Error|Undefined control sequence).*$", log, re.M),
            "undefined_references": re.findall(r"^.*(?:LaTeX Warning:.*undefined|multiply defined).*$", log, re.M),
            "missing_glyphs": re.findall(r"^Missing character:.*$", log, re.M),
            "overfull_boxes": re.findall(r"^Overfull.*$", log, re.M),
            "broken_internal_links": [], "text_outside_page": [],
            "rendered_pages": len(pdf),
        }
        thumbs = []
        for i, page in enumerate(pdf):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            pixmap.save(output / f"page-{i+1:03d}.png")
            thumb = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            thumb.thumbnail((270, 382))
            tile = Image.new("RGB", (294, 416), "#e5e7eb")
            tile.paste(thumb, ((294 - thumb.width)//2, 12))
            ImageDraw.Draw(tile).text((12, 396), f"PDF page {i+1}", fill="black")
            thumbs.append(tile)
            for link in page.get_links():
                if link.get("kind") == fitz.LINK_GOTO and not 0 <= link.get("page", -1) < len(pdf):
                    report["broken_internal_links"].append({"page": i+1, "link": str(link)})
            for block in page.get_text("dict")["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    box = fitz.Rect(line["bbox"])
                    if box.x0 < -1 or box.y0 < -1 or box.x1 > page.rect.width+1 or box.y1 > page.rect.height+1:
                        report["text_outside_page"].append({"page": i+1, "bbox": list(box)})
        for first in range(0, len(thumbs), 20):
            sheet = Image.new("RGB", (5*294, 4*416), "white")
            for offset, thumb in enumerate(thumbs[first:first+20]):
                sheet.paste(thumb, ((offset%5)*294, (offset//5)*416))
            sheet.save(output / f"contact-{first+1:03d}-{min(first+20,len(thumbs)):03d}.png")
        report["contact_sheets"] = [p.relative_to(ROOT).as_posix() for p in sorted(output.glob("contact-*.png"))]
        summary[name] = report
        print(json.dumps({"book": name, **{k:v for k,v in report.items() if k not in ("contact_sheets", "module_presence")}}))
    (ROOT / "build/review/report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
