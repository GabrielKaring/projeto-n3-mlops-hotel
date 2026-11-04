"""Gera HISTORICO_DESENVOLVIMENTO.md a partir do `git log` (fonte única da verdade).

Uso:
    python scripts/gerar_historico.py            # reescreve HISTORICO_DESENVOLVIMENTO.md
    python scripts/gerar_historico.py --stdout   # imprime em vez de gravar

Respeita o arquivo .mailmap (mesma pessoa com vários e-mails vira um integrante só).
Não inventa nem edita nada: só reorganiza o que o git já registrou.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import OrderedDict
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "HISTORICO_DESENVOLVIMENTO.md"
SEP = "\x1f"
MARCA = "\x1e"


def git_log() -> list[dict]:
    formato = MARCA + SEP.join(["%h", "%aN", "%ad", "%s", "%(trailers:key=Co-Authored-By,valueonly,separator=%x2C )"])
    saida = subprocess.run(
        ["git", "log", "--reverse", "--no-merges", "--date=short", f"--pretty=format:{formato}", "--shortstat"],
        cwd=RAIZ, capture_output=True, text=True, check=True,
    ).stdout
    commits = []
    for bloco in saida.split(MARCA)[1:]:
        cabecalho, *resto = bloco.strip("\n").split("\n")
        hash_, autor, dia, assunto, coautores = (cabecalho.split(SEP) + [""] * 5)[:5]
        stat = " ".join(resto)
        nums = {k: int(v) for v, k in re.findall(r"(\d+) (file|insertion|deletion)", stat)}
        commits.append({
            "hash": hash_, "autor": autor, "data": dia, "assunto": assunto,
            "coautores": [c.split("<")[0].strip() for c in coautores.split(",") if c.strip()],
            "arquivos": nums.get("file", 0), "mais": nums.get("insertion", 0), "menos": nums.get("deletion", 0),
        })
    return commits


def renderizar(commits: list[dict]) -> str:
    por_autor: "OrderedDict[str, list[dict]]" = OrderedDict()
    for c in commits:
        por_autor.setdefault(c["autor"], []).append(c)

    L = [
        "# Histórico de desenvolvimento",
        "",
        f"_Gerado em {date.today():%d/%m/%Y} por `scripts/gerar_historico.py` a partir do `git log`. "
        "Não edite à mão: rode o script de novo._",
        "",
        "## Resumo por integrante",
        "",
        "| Integrante | Commits | Primeiro | Último | Linhas + | Linhas − |",
        "|---|---:|---|---|---:|---:|",
    ]
    for autor, cs in sorted(por_autor.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {autor} | {len(cs)} | {cs[0]['data']} | {cs[-1]['data']} | {sum(c['mais'] for c in cs):,} | {sum(c['menos'] for c in cs):,} |")
    L += ["", f"Total: {len(commits)} commits de {len(por_autor)} integrante(s).", ""]

    ia = [c for c in commits if any("claude" in x.lower() for x in c["coautores"])]
    if ia:
        L += [
            "## Assistência de IA declarada",
            "",
            f"{len(ia)} de {len(commits)} commits têm `Co-Authored-By: Claude` no corpo da mensagem: "
            "o trabalho foi feito com apoio de IA e supervisionado pelo autor do commit.",
            "",
        ]

    L += ["## Linha do tempo", "", "| Data | Commit | Integrante | Mensagem | Arquivos |", "|---|---|---|---|---:|"]
    for c in commits:
        marca = " 🤖" if any("claude" in x.lower() for x in c["coautores"]) else ""
        L.append(f"| {c['data']} | `{c['hash']}` | {c['autor']} | {c['assunto']}{marca} | {c['arquivos']} |")
    L += ["", "🤖 = commit com co-autoria de IA declarada.", ""]

    for autor, cs in por_autor.items():
        L += [f"## Contribuições de {autor}", ""]
        L += [f"- `{c['hash']}` {c['data']} · {c['assunto']}" for c in cs]
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    texto = renderizar(git_log())
    if args.stdout:
        print(texto)
    else:
        SAIDA.write_text(texto, encoding="utf-8")
        print(f"Gravado: {SAIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
