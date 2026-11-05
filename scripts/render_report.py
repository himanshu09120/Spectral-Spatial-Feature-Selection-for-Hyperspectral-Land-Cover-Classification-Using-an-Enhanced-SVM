"""Render the checked-in Markdown report as a portable, paginated PDF.

Install requirements-report.txt in the project environment first.
"""
from pathlib import Path
import re
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, KeepTogether, PageBreak, HRFlowable)
import matplotlib
import datetime

ROOT = Path(__file__).resolve().parents[1]
TEAL_DARK = "#0B2F3A"
TEAL_MID = "#2E6F72"
GOLD = "#C9881E"
INK = "#1B1B1B"
MUTED = "#5C716C"
TINT = "#EEF3F1"


def render(source=ROOT / "report/final_report.md", target=ROOT / "output/pdf/final_report.pdf"):
    fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in [("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                           ("DejaVu-Oblique", "DejaVuSans-Oblique.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-Bold")
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "DejaVu"
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 13
    styles["Normal"].spaceAfter = 7
    for name in ["Title", "Heading1", "Heading2", "Heading3"]:
        styles[name].fontName = "DejaVu-Bold"
        styles[name].textColor = colors.HexColor(TEAL_DARK)
    styles["Title"].fontSize = 19
    styles["Title"].leading = 25
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 16
    styles.add(ParagraphStyle(name="Cell", parent=styles["Normal"], fontSize=7, leading=10, spaceAfter=0))
    styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8, leading=11, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=27, leading=33, alignment=TA_CENTER, textColor=colors.white))
    styles.add(ParagraphStyle(name="CoverEyebrow", parent=styles["Normal"], fontSize=12, leading=16, alignment=TA_CENTER,
                              textColor=colors.HexColor(GOLD), spaceAfter=14))
    styles.add(ParagraphStyle(name="CoverAuthor", parent=styles["Normal"], fontSize=12.5, leading=18, alignment=TA_CENTER, textColor=colors.white))
    styles.add(ParagraphStyle(name="CoverStatus", parent=styles["Normal"], fontSize=9.5, leading=14, alignment=TA_CENTER,
                              textColor=colors.HexColor("#CFE3DC"), spaceBefore=18))

    def inline(text):
        text = text.replace("—", "-").replace("–", "-").replace("‑", "-")
        text = escape(text)
        text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: ('<link href="' + m[2] + '" color="' + TEAL_MID + '">' + m[1] + '</link>') if m[2].startswith("http") else m[1], text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
        return text.replace("`", "")

    story, lines, i = [], source.read_text(encoding="utf-8").splitlines(), 0
    width = 6.9 * inch

    # ---- Cover page: title + bold front-matter lines up to the first "## " heading ----
    cover, i = [], 0
    title_line = lines[i].strip()
    assert title_line.startswith("# "), "report must start with a level-1 title"
    cover.append(Paragraph(inline(title_line.lstrip("# ")), styles["CoverTitle"]))
    i += 1
    front_matter = []
    while i < len(lines) and not lines[i].strip().startswith("## "):
        text = lines[i].strip()
        if text:
            front_matter.append(text)
        i += 1
    if front_matter:
        course_line = front_matter[0].strip("*")
        cover.insert(0, Paragraph(inline(course_line.upper()), styles["CoverEyebrow"]))
        cover[1] = Paragraph(inline(title_line.lstrip("# ")), styles["CoverTitle"])
        for text in front_matter[1:-1]:
            cover.append(Spacer(1, 4))
            cover.append(Paragraph(inline(text.strip("*")), styles["CoverAuthor"]))
        cover.append(Paragraph(inline(front_matter[-1]), styles["CoverStatus"]))
    story.append(Spacer(1, 3.7 * inch))
    story.extend(cover)
    story.append(PageBreak())

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip().strip("|").split("|")
                if not all(re.fullmatch(r"[ :\-]+", c) for c in row):
                    rows.append([Paragraph(inline(c.strip()), styles["Cell"]) for c in row])
                i += 1
            n = len(rows[0])
            table = Table(rows, colWidths=[width / n] * n, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(TEAL_DARK)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor(TINT)]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), .7, colors.HexColor(TEAL_MID))]))
            story.extend([table, Spacer(1, 10)])
            continue
        picture = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if picture:
            from PIL import Image as PILImage
            path = source.parent / picture[2]
            with PILImage.open(path) as im:
                w, h = im.size
            scale = min(width / w, 4.7 * inch / h)
            story.append(KeepTogether([Image(str(path), w * scale, h * scale), Paragraph(inline(picture[1]), styles["Caption"])]))
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            story.append(Paragraph(inline(line.lstrip("# ")), styles["Title" if level == 1 else "Heading2"]))
        elif line.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(escape(lines[i]))
                i += 1
            story.append(Paragraph("<br/>".join(block), styles["Cell"]))
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(inline(line), styles["Normal"]))
        i += 1
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(target), pagesize=(8.3 * inch, 11.7 * inch),
                            rightMargin=.7 * inch, leftMargin=.7 * inch,
                            topMargin=.65 * inch, bottomMargin=.65 * inch,
                            title="FRSS-SVM: Final Project Report", author="Himanshu Sinha; Ghule Yash Mahesh")

    _today = datetime.date.today()
    today = f"{_today.day} {_today.strftime('%B %Y')}"
    def cover_page(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(TEAL_DARK))
        canvas.rect(0, 0, document.pagesize[0], document.pagesize[1], fill=1, stroke=0)
        canvas.restoreState()

    def footer(canvas, document):
        canvas.setFont("DejaVu", 7)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(.7 * inch, .35 * inch, f"Pattern Recognition | WHU-Hi-LongKou | {today}")
        canvas.drawRightString(7.6 * inch, .35 * inch, str(document.page))

    doc.build(story, onFirstPage=cover_page, onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    render()
