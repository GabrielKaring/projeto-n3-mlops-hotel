"""Gera reports/relatorio_n3.pdf a partir de reports/relatorio_n3.md.

Uso:
    pip install reportlab                # só para o relatório, não entra no requirements do pipeline
    python scripts/gerar_relatorio.py

O texto fica no .md, que dá para editar e revisar no PR. O script só entende o que o relatório usa:
títulos (#, ##, ###), parágrafos, listas com "-", tabelas com "|", blocos de código com ``` e
imagens em linha própria (![legenda](caminho)). **negrito** e `código` funcionam dentro do texto.
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

RAIZ = Path(__file__).resolve().parents[1]
ENTRADA = RAIZ / "reports" / "relatorio_n3.md"
SAIDA = RAIZ / "reports" / "relatorio_n3.pdf"


def _fontes() -> tuple[str, str, str]:
    """DejaVu se existir (acentos e símbolos); senão as fontes padrão do PDF."""
    for pasta in (Path("/usr/share/fonts/truetype/dejavu"), Path("C:/Windows/Fonts")):
        normal, negrito, mono = pasta / "DejaVuSans.ttf", pasta / "DejaVuSans-Bold.ttf", pasta / "DejaVuSansMono.ttf"
        italico = pasta / "DejaVuSans-Oblique.ttf"
        if normal.exists() and negrito.exists() and mono.exists() and italico.exists():
            pdfmetrics.registerFont(TTFont("Corpo-Oblique", str(italico)))
            pdfmetrics.registerFont(TTFont("Corpo", str(normal)))
            pdfmetrics.registerFont(TTFont("Corpo-Bold", str(negrito)))
            pdfmetrics.registerFont(TTFont("Mono", str(mono)))
            pdfmetrics.registerFontFamily("Corpo", normal="Corpo", bold="Corpo-Bold", italic="Corpo-Oblique")
            return "Corpo", "Corpo-Bold", "Mono"
    return "Helvetica", "Helvetica-Bold", "Courier"


CORPO, NEGRITO, MONO = _fontes()
base = getSampleStyleSheet()
ESTILOS = {
    "titulo": ParagraphStyle("titulo", parent=base["Title"], fontName=NEGRITO, fontSize=18, leading=22),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=NEGRITO, fontSize=14, leading=18, spaceBefore=12),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=NEGRITO, fontSize=11.5, leading=15, spaceBefore=8),
    "corpo": ParagraphStyle("corpo", parent=base["BodyText"], fontName=CORPO, fontSize=10.5, leading=14.5),
    "celula": ParagraphStyle("celula", parent=base["BodyText"], fontName=CORPO, fontSize=9, leading=11.5),
    "legenda": ParagraphStyle("legenda", parent=base["BodyText"], fontName=CORPO, fontSize=9, leading=11,
                              alignment=TA_CENTER, textColor=colors.HexColor("#555555")),
    "codigo": ParagraphStyle("codigo", fontName=MONO, fontSize=8.5, leading=11),
}


def _inline(texto: str) -> str:
    texto = texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    texto = re.sub(r"`([^`]+)`", rf'<font face="{MONO}">\1</font>', texto)
    texto = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", texto)
    texto = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"<i>\1</i>", texto)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1a5fb4">\1</link>', texto)


def _tabela(linhas: list[str], largura: float) -> Table:
    celulas = [[c.strip() for c in l.strip().strip("|").split("|")] for l in linhas]
    celulas = [c for c in celulas if not all(re.fullmatch(r":?-+:?", x) for x in c)]
    dados = [[Paragraph(_inline(c), ESTILOS["celula"]) for c in linha] for linha in celulas]
    # Largura de cada coluna proporcional ao texto mais longo dela (com limites), para nome comprido não quebrar.
    pesos = [min(max(max(len(linha[k]) for linha in celulas), 6), 40) for k in range(len(celulas[0]))]
    t = Table(dados, colWidths=[largura * p / sum(pesos) for p in pesos], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b8c4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def montar(md: str, largura: float) -> list:
    story, linhas, i = [], md.splitlines(), 0
    while i < len(linhas):
        l = linhas[i]
        if not l.strip():
            i += 1
            continue
        if l.startswith("```"):
            j = i + 1
            while j < len(linhas) and not linhas[j].startswith("```"):
                j += 1
            story += [Preformatted("\n".join(linhas[i + 1:j]), ESTILOS["codigo"]), Spacer(1, 6)]
            i = j + 1
            continue
        m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", l.strip())
        if m:
            img = Image(str(RAIZ / "reports" / m.group(2)))
            escala = min(1.0, largura / img.drawWidth)
            img.drawWidth, img.drawHeight = img.drawWidth * escala, img.drawHeight * escala
            story += [img, Paragraph(_inline(m.group(1)), ESTILOS["legenda"]), Spacer(1, 8)]
            i += 1
            continue
        if l.startswith("# "):
            story.append(Paragraph(_inline(l[2:]), ESTILOS["titulo"]))
        elif l.startswith("## "):
            story.append(Paragraph(_inline(l[3:]), ESTILOS["h1"]))
        elif l.startswith("### "):
            story.append(Paragraph(_inline(l[4:]), ESTILOS["h2"]))
        elif l.startswith("|"):
            j = i
            while j < len(linhas) and linhas[j].startswith("|"):
                j += 1
            story += [_tabela(linhas[i:j], largura), Spacer(1, 8)]
            i = j
            continue
        elif l.startswith("- "):
            itens = []
            while i < len(linhas) and linhas[i].startswith("- "):
                itens.append(ListItem(Paragraph(_inline(linhas[i][2:]), ESTILOS["corpo"]), leftIndent=12))
                i += 1
            story += [ListFlowable(itens, bulletType="bullet", start="•", leftIndent=12), Spacer(1, 4)]
            continue
        else:
            j = i
            while j < len(linhas) and linhas[j].strip() and not re.match(r"(#|\||- |```|!\[)", linhas[j]):
                j += 1
            story += [Paragraph(_inline(" ".join(linhas[i:j])), ESTILOS["corpo"]), Spacer(1, 6)]
            i = j
            continue
        i += 1
    return story


def _rodape(canvas, doc) -> None:
    canvas.setFont(CORPO, 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Relatório N3 · página {doc.page}")


def main() -> int:
    doc = SimpleDocTemplate(str(SAIDA), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm, title="Relatório N3 · Pipeline MLOps")
    doc.build(montar(ENTRADA.read_text(encoding="utf-8"), doc.width), onFirstPage=_rodape, onLaterPages=_rodape)
    print("gerado:", SAIDA.relative_to(RAIZ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
