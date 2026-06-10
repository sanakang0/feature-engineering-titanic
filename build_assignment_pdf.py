from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = Path(__file__).resolve().parent
MD_PATH = BASE_DIR / "과제_특성공학_보고서.md"
PDF_PATH = BASE_DIR / "과제_특성공학_보고서.pdf"
KOREAN_FONT = Path(r"C:\Windows\Fonts\malgun.ttf")
KOREAN_FONT_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def clean_inline_md(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = text.replace("`", "")
    return text


def parse_markdown(md_text: str):
    lines = md_text.splitlines()
    i = 0
    blocks = []

    while i < len(lines):
        line = lines[i].rstrip()

        if not line:
            blocks.append(("blank", ""))
            i += 1
            continue

        if line.startswith("# "):
            blocks.append(("h1", clean_inline_md(line[2:].strip())))
            i += 1
            continue

        if line.startswith("## "):
            blocks.append(("h2", clean_inline_md(line[3:].strip())))
            i += 1
            continue

        if line.startswith("### "):
            blocks.append(("h3", clean_inline_md(line[4:].strip())))
            i += 1
            continue

        if line.startswith("#### "):
            blocks.append(("h4", clean_inline_md(line[5:].strip())))
            i += 1
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            blocks.append(("image", image_match.group(2).strip()))
            i += 1
            continue

        if line.startswith("- "):
            blocks.append(("bullet", clean_inline_md(line[2:].strip())))
            i += 1
            continue

        if re.match(r"^\d+\.\s+", line):
            blocks.append(("number", clean_inline_md(line)))
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(("table", table_lines))
            continue

        blocks.append(("p", clean_inline_md(line)))
        i += 1

    return blocks


def markdown_table_to_data(table_lines):
    rows = []
    for idx, raw in enumerate(table_lines):
        parts = [c.strip() for c in raw.strip("|").split("|")]
        if idx == 1 and all(set(p) <= set(":- ") for p in parts):
            continue
        rows.append(parts)
    return rows


def build_pdf(md_path: Path, pdf_path: Path):
    if KOREAN_FONT.exists():
        pdfmetrics.registerFont(TTFont("MalgunGothic", str(KOREAN_FONT)))
    if KOREAN_FONT_BOLD.exists():
        pdfmetrics.registerFont(TTFont("MalgunGothic-Bold", str(KOREAN_FONT_BOLD)))

    md_text = md_path.read_text(encoding="utf-8")
    blocks = parse_markdown(md_text)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )

    styles = getSampleStyleSheet()
    body_font = "MalgunGothic" if KOREAN_FONT.exists() else "Helvetica"
    bold_font = "MalgunGothic-Bold" if KOREAN_FONT_BOLD.exists() else "Helvetica-Bold"
    style_h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontName=bold_font,
        fontSize=16,
        spaceAfter=8,
    )
    style_h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=13,
        spaceBefore=10,
        spaceAfter=6,
    )
    style_h3 = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontName=bold_font,
        fontSize=11,
        spaceBefore=8,
        spaceAfter=4,
    )
    style_h4 = ParagraphStyle(
        "H4",
        parent=styles["Heading4"],
        fontName=bold_font,
        fontSize=10,
        spaceBefore=6,
        spaceAfter=3,
    )
    style_p = ParagraphStyle(
        "P", parent=styles["BodyText"], fontName=body_font, fontSize=9.5, leading=13
    )
    style_b = ParagraphStyle(
        "B",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=9.5,
        leftIndent=12,
        bulletIndent=0,
        leading=13,
    )

    story = []

    for kind, value in blocks:
        if kind == "blank":
            story.append(Spacer(1, 0.15 * cm))
        elif kind == "h1":
            story.append(Paragraph(value, style_h1))
        elif kind == "h2":
            story.append(Paragraph(value, style_h2))
        elif kind == "h3":
            story.append(Paragraph(value, style_h3))
        elif kind == "h4":
            story.append(Paragraph(value, style_h4))
        elif kind == "p":
            story.append(Paragraph(value, style_p))
        elif kind == "bullet":
            story.append(Paragraph(f"• {value}", style_b))
        elif kind == "number":
            story.append(Paragraph(value, style_p))
        elif kind == "image":
            img_path = (BASE_DIR / value).resolve()
            if img_path.exists():
                img = Image(str(img_path))
                max_w = A4[0] - (doc.leftMargin + doc.rightMargin)
                scale = max_w / float(img.imageWidth)
                img.drawWidth = max_w
                img.drawHeight = float(img.imageHeight) * scale
                story.append(Spacer(1, 0.1 * cm))
                story.append(img)
                story.append(Spacer(1, 0.2 * cm))
            else:
                story.append(Paragraph(f"[이미지 누락] {value}", style_p))
        elif kind == "table":
            data = markdown_table_to_data(value)
            if data:
                table = Table(data, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("FONTNAME", (0, 0), (-1, 0), bold_font),
                            ("FONTNAME", (0, 1), (-1, -1), body_font),
                            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 0.2 * cm))

    doc.build(story)


if __name__ == "__main__":
    build_pdf(MD_PATH, PDF_PATH)
    print(f"PDF generated: {PDF_PATH}")
