#!/usr/bin/env python3
"""Convert trc_draft.md into an Elsevier-style manuscript.docx using python-docx.

No pandoc. Markdown subset handled: H1/H2/H3, pipe tables, `>` boxes,
bullet/numbered lists, `---` rules, inline **bold** / *italic* / `code` (as
italic) / ^superscript^.

Run:  python md_to_docx.py
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC = Path(__file__).with_name("trc_draft.md")
OUT = Path(__file__).with_name("manuscript.docx")

FONT = "Times New Roman"
BODY_PT = 12
TABLE_PT = 9
TEXT_WIDTH_CM = 16.0

HEADING_PT = {1: 16, 2: 14, 3: 12}


# ---------------------------------------------------------------- run helpers
def set_run_font(run, size=BODY_PT, bold=False, italic=False, sup=False,
                 color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if sup:
        run.font.superscript = True
    if color is not None:
        run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.insert(0, rFonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rFonts.set(qn(attr), FONT)


def parse_inline(text, bold=False, italic=False, sup=False):
    """Return [(text, bold, italic, superscript)] spans.

    Code spans are rendered as plain italic text (backticks stripped) and are
    not re-parsed, so `^+` or `∫_a^b` inside them stay literal.
    """
    spans = []
    buf = []

    def flush():
        if buf:
            spans.append(("".join(buf), bold, italic, sup))
            del buf[:]

    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "`":
            j = text.find("`", i + 1)
            if j == -1:
                buf.append(ch)
                i += 1
                continue
            flush()
            spans.append((text[i + 1:j], bold, True, sup))
            i = j + 1
        elif text.startswith("**", i):
            j = text.find("**", i + 2)
            if j == -1:
                buf.append("**")
                i += 2
                continue
            flush()
            spans.extend(parse_inline(text[i + 2:j], True, italic, sup))
            i = j + 2
        elif ch == "*":
            j = text.find("*", i + 1)
            if j == -1:
                buf.append(ch)
                i += 1
                continue
            flush()
            spans.extend(parse_inline(text[i + 1:j], bold, True, sup))
            i = j + 1
        elif ch == "^" and not sup:
            j = text.find("^", i + 1)
            if j == -1:
                buf.append(ch)
                i += 1
                continue
            flush()
            spans.append((text[i + 1:j], bold, italic, True))
            i = j + 1
        else:
            buf.append(ch)
            i += 1
    flush()
    return spans


def clean_text(text):
    # HTML em-spaces -> ordinary spaces; keeps "(1) ... (6)" visible.
    return text.replace("&emsp;", " ")


def add_runs(paragraph, text, size=BODY_PT, force_bold=False, color=None):
    for chunk, bold, italic, sup in parse_inline(clean_text(text)):
        run = paragraph.add_run(chunk)
        set_run_font(run, size=size, bold=bold or force_bold,
                     italic=italic, sup=sup, color=color)


# ----------------------------------------------------------- paragraph styles
def style_body_paragraph(p, justify=True, spacing=1.5):
    pf = p.paragraph_format
    pf.line_spacing = spacing
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_body(doc, text, justify=True):
    p = doc.add_paragraph()
    style_body_paragraph(p, justify=justify)
    add_runs(p, text)
    return p


def add_heading(doc, level, text, page_break=False):
    p = doc.add_paragraph(style=f"Heading {level}")
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    pf.keep_with_next = True
    pf.space_before = Pt({1: 0, 2: 14, 3: 10}[level])
    pf.space_after = Pt({1: 12, 2: 6, 3: 4}[level])
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if page_break:
        pf.page_break_before = True
    for chunk, bold, italic, sup in parse_inline(clean_text(text)):
        run = p.add_run(chunk)
        set_run_font(run, size=HEADING_PT[level], bold=True,
                     italic=italic or level == 3, sup=sup,
                     color=RGBColor(0, 0, 0))
    return p


def add_box_paragraph(doc, text):
    """Theorem / definition / proposition box: indented paragraph."""
    p = doc.add_paragraph()
    style_body_paragraph(p)
    pf = p.paragraph_format
    pf.left_indent = Cm(0.75)
    pf.right_indent = Cm(0.5)
    pf.space_before = Pt(4)
    pf.space_after = Pt(4)
    add_runs(p, text)
    return p


def add_list_item(doc, marker, text, numbered):
    if numbered:
        p = doc.add_paragraph()
        style_body_paragraph(p)
        pf = p.paragraph_format
        pf.left_indent = Cm(0.9)
        pf.first_line_indent = Cm(-0.45)
        r = p.add_run(marker + "\t")
        set_run_font(r, size=BODY_PT)
    else:
        p = doc.add_paragraph(style="List Bullet")
        style_body_paragraph(p)
        pf = p.paragraph_format
        pf.left_indent = Cm(0.9)
        pf.first_line_indent = Cm(-0.45)
    add_runs(p, text)
    return p


def add_rule(doc):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "000000")
    pBdr.append(bottom)
    pPr.insert_element_before(
        pBdr, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku",
        "w:wordWrap", "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE",
        "w:autoSpaceDN", "w:bidi", "w:adjustRightInd", "w:snapToGrid",
        "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
        "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
        "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle",
        "w:rPr", "w:sectPr", "w:pPrChange")
    return p


# --------------------------------------------------------------------- tables
def set_table_borders(table):
    tblPr = table._tbl.tblPr
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("left", "nil"),
                      ("bottom", "single"), ("right", "nil"),
                      ("insideH", "single"), ("insideV", "nil")):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), val)
        e.set(qn("w:sz"), "4")
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), "000000")
        borders.append(e)
    tblPr.insert_element_before(
        borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
        "w:tblCaption", "w:tblDescription", "w:tblPrChange")


def set_cell_margins(table, left=0.08, right=0.08, top=0.03, bottom=0.03):
    tblPr = table._tbl.tblPr
    old = tblPr.find(qn("w:tblCellMar"))
    if old is not None:
        tblPr.remove(old)
    mar = OxmlElement("w:tblCellMar")
    for edge, cm in (("top", top), ("left", left),
                     ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:w"), str(int(Cm(cm).twips)))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.insert_element_before(
        mar, "w:tblLook", "w:tblCaption", "w:tblDescription", "w:tblPrChange")


def plain_text(text):
    return re.sub(r"[*`^]", "", text)


def em_width(ch):
    """Rough Times New Roman advance width in em units."""
    if ch in "iljI.,'|:;·-–— \u2212":
        return 0.30
    if ch in "mwMW":
        return 0.88
    if ch in "[](){}%":
        return 0.42
    if ch.isupper() or ch.isdigit():
        return 0.62
    return 0.52


def text_width(txt, pt=TABLE_PT, bold=False):
    em = sum(em_width(c) for c in txt)
    return em * pt * 0.03528 * (1.06 if bold else 1.0)


def column_widths(header, body, total_cm=TEXT_WIDTH_CM):
    """Allocate widths so no single word has to break mid-word if avoidable."""
    n = len(header)
    mins, wants = [], []
    for j in range(n):
        vals = [(header[j], True)] + [(row[j], False) for row in body]
        longest = 0.0
        widest = 0.0
        for v, is_header in vals:
            txt = plain_text(v)
            widest = max(widest, text_width(txt, bold=is_header))
            for word in txt.split():
                longest = max(longest, text_width(word, bold=is_header))
        mins.append(longest + 0.22)
        wants.append(max(widest + 0.30, longest + 0.22))

    if sum(mins) < total_cm:
        spare = total_cm - sum(mins)
        extra = [max(0.4, w - m) for w, m in zip(wants, mins)]
        s = sum(extra)
        widths = [m + spare * e / s for m, e in zip(mins, extra)]
    else:
        widths = [total_cm * m / sum(mins) for m in mins]
    scale = total_cm / sum(widths)
    return [w * scale for w in widths]


def add_table(doc, raw_rows):
    cell_rows = []
    for raw in raw_rows:
        s = raw.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        cell_rows.append([c.strip() for c in s.split("|")])

    header = cell_rows[0]
    body = [r for r in cell_rows[1:]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    ncols = len(header)
    body = [(r + [""] * ncols)[:ncols] for r in body]

    # keep the caption ("**Table N. ...**") with its table
    if doc.paragraphs:
        last = doc.paragraphs[-1]
        if re.match(r"^Table\s+\d+\.", last.text):
            last.paragraph_format.keep_with_next = True
            last.paragraph_format.space_after = Pt(4)

    table = doc.add_table(rows=1 + len(body), cols=ncols)
    table.style = "Normal Table"          # borderless base style ...
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)              # ... plus horizontal rules only
    set_cell_margins(table)

    widths = column_widths(header, body)
    for row in table.rows:
        for j, cell in enumerate(row.cells):
            cell.width = Cm(widths[j])

    def fill(cell, text, force_bold, keep_next=True):
        p = cell.paragraphs[0]
        pf = p.paragraph_format
        pf.line_spacing = 1.0
        pf.space_before = Pt(1)
        pf.space_after = Pt(1)
        pf.keep_with_next = keep_next
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_runs(p, text, size=TABLE_PT, force_bold=force_bold)

    for j, text in enumerate(header):
        fill(table.rows[0].cells[j], text, True)

    trPr = table.rows[0]._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))
    trPr.append(OxmlElement("w:tblHeader"))

    last_row = len(body)
    for r, rowvals in enumerate(body, start=1):
        for j, text in enumerate(rowvals):
            fill(table.rows[r].cells[j], text, False,
                 keep_next=(r != last_row))
        # never split a row across a page boundary
        rowTrPr = table.rows[r]._tr.get_or_add_trPr()
        rowTrPr.append(OxmlElement("w:cantSplit"))

    return table


# ------------------------------------------------------------------ converter
def build():
    lines = [ln.rstrip("\r") for ln in SRC.read_text(encoding="utf-8").split("\n")]

    doc = Document()

    # page setup: A4, 2.5 cm margins, single column
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

    # Normal: TNR 12 pt, 1.5 spacing, justified
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    npf = normal.paragraph_format
    npf.line_spacing = 1.5
    npf.space_before = Pt(0)
    npf.space_after = Pt(0)
    npf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for level in (1, 2, 3):
        st = doc.styles[f"Heading {level}"]
        st.font.name = FONT
        st.font.size = Pt(HEADING_PT[level])
        st.font.bold = True
        st.font.italic = (level == 3)
        st.font.color.rgb = RGBColor(0, 0, 0)
        pf = st.paragraph_format
        pf.line_spacing = 1.5
        pf.keep_with_next = True
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT

    pending: list[str] = []
    skipped: list[tuple[int, str]] = []

    def flush():
        if pending:
            add_body(doc, " ".join(x.strip() for x in pending))
            pending.clear()

    i = 0
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()

        if not s:
            flush()
            i += 1
            continue

        m = re.match(r"(#{1,6})\s+(.*)$", s)
        if m:
            flush()
            level = min(len(m.group(1)), 3)
            text = m.group(2).strip()
            page_break = text == "Abstract" or text.startswith("1. Introduction")
            add_heading(doc, level, text, page_break=page_break)
            i += 1
            continue

        if s.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            add_table(doc, block)
            continue

        if s.startswith(">"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                raw_q = lines[i].strip()
                content = raw_q[1:]
                if content.startswith(" "):
                    content = content[1:]
                block.append(content.rstrip())
                i += 1
            for para in split_box(block):
                add_box_paragraph(doc, para)
            continue

        if re.fullmatch(r"(---|\*\s\*\s\*|___+)", s):
            flush()
            add_rule(doc)
            i += 1
            continue

        ml = re.match(r"^(-|\d+\.)\s+(.*)$", s)
        if ml:
            flush()
            numbered = ml.group(1) != "-"
            marker = ml.group(1) if numbered else None
            add_list_item(doc, marker, ml.group(2), numbered)
            i += 1
            continue

        pending.append(raw)
        i += 1

    flush()

    doc.save(OUT)
    return doc, skipped, len(lines)


def split_box(block):
    """Split a blockquote into paragraphs.

    Breaks on blank `>` lines, on markdown list items, on `(P1)`-style
    hypothesis markers, and isolates display-equation lines (those carrying
    the `&emsp;&emsp;(n)` number).
    """
    paras, cur = [], []

    def close():
        if cur:
            paras.append(" ".join(cur))
            cur.clear()

    for line in block:
        if line == "":
            close()
            continue
        if cur and (re.match(r"^(-|\d+\.)\s+", line) or re.match(r"^\((P\d)\)", line)):
            close()
        eq_line = "&emsp;&emsp;(" in line
        if cur and eq_line:
            close()
        cur.append(line)
        if eq_line:
            close()
    close()
    return paras


# ------------------------------------------------------------------- verify
def verify(path=OUT):
    doc = Document(path)
    paras = doc.paragraphs
    ok = True

    print(f"(a) number of paragraphs: {len(paras)}")

    ntables = len(doc.tables)
    print(f"(b) number of tables: {ntables}")
    if ntables != 7:
        ok = False
        print("    FAIL: expected 7 tables")

    heads = [(p.style.name, p.text) for p in paras
             if p.style.name in ("Heading 1", "Heading 2", "Heading 3")]
    h1 = [t for s, t in heads if s == "Heading 1"]
    h2 = [t for s, t in heads if s == "Heading 2"]
    h3 = [t for s, t in heads if s == "Heading 3"]
    print(f"(c) headings: H1={len(h1)} H2={len(h2)} H3={len(h3)} "
          f"(total {len(heads)})")
    for s, t in heads:
        if s in ("Heading 1", "Heading 2"):
            print(f"    [{s}] {t}")

    body = "\n".join(p.text for p in paras if p.text.strip())
    print(f"(d) first 200 chars of body text:\n    {body[:200]!r}")

    bad_paras = [p.text[:120] for p in paras
                 if "**" in p.text or "`" in p.text]
    bad_cells = [c.text[:60] for t in doc.tables for r in t.rows
                 for c in r.cells if "**" in c.text or "`" in c.text]
    print(f"(e) paragraphs containing literal '**' or backtick: {len(bad_paras)}")
    print(f"    table cells containing literal '**' or backtick: {len(bad_cells)}")
    for x in bad_paras[:10]:
        print("      !!", x)
    for x in bad_cells[:10]:
        print("      !! cell:", x)
    if bad_paras or bad_cells:
        ok = False

    left_emsp = sum(p.text.count("&emsp;") for p in paras)
    if left_emsp:
        ok = False
        print(f"    FAIL: {left_emsp} leftover '&emsp;' entities")

    eqnums = [n for n in ("(1)", "(2)", "(3)", "(4)", "(5)", "(6)")
              if n not in body]
    if eqnums:
        print(f"    WARN: equation numbers missing from text: {eqnums}")

    pb = [p.text for p in paras if p.paragraph_format.page_break_before]
    print(f"(f) page breaks before: {pb}")
    if pb != ["Abstract", "1. Introduction"]:
        ok = False
        print("    FAIL: page breaks not exactly before Abstract and 1. Introduction")

    words = sum(len(p.text.split()) for p in paras)
    words += sum(len(c.text.split()) for t in doc.tables for r in t.rows
                 for c in r.cells)
    print(f"(g) approximate word count: {words}")

    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


def main():
    doc, skipped, nlines = build()
    print(f"wrote {OUT} from {nlines} source lines")
    ok = verify(OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
