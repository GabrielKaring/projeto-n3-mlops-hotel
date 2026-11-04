"""Gera HISTORICO_DESENVOLVIMENTO.md a partir do `git log` (fonte única da verdade).

Uso:
    python scripts/gerar_historico.py            # reescreve HISTORICO_DESENVOLVIMENTO.md
    python scripts/gerar_historico.py --stdout   # imprime em vez de gravar

Respeita o arquivo .mailmap (mesma pessoa com vários e-mails vira um integrante só).
Não inventa nem edita nada: só reorganiza o que o git já registrou.

Visão por feature: commits cuja mensagem começa com `[F3]` (ou `[F3][F4]`, `[F3,F4]`) são atribuídos
àquela feature; o dono vem de config/equipe.yaml e precisa ter o mesmo nome do autor no git.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from collections import OrderedDict
from datetime import date
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "HISTORICO_DESENVOLVIMENTO.md"
EQUIPE = RAIZ / "config" / "equipe.yaml"
TAG = re.compile(r"^\s*((?:\[\s*F\d+(?:\s*[,/+]\s*F\d+)*\s*\]\s*)+)")
SEP = "\x1f"
MARCA = "\x1e"


def extrair_features(assunto: str) -> list[str]:
    """Ids de feature no começo da mensagem: ``[F3] x`` -> ['F3']; ``[F3][F4] x`` e ``[F3,F4] x`` -> ['F3', 'F4']."""
    m = TAG.match(assunto)
    return list(dict.fromkeys(re.findall(r"F\d+", m.group(1)))) if m else []


def carregar_equipe() -> dict:
    return yaml.safe_load(EQUIPE.read_text(encoding="utf-8"))


def git_log() -> list[dict]:
    formato = MARCA + SEP.join(["%h", "%aN", "%ad", "%s"])
    saida = subprocess.run(
        ["git", "log", "--reverse", "--no-merges", "--date=short", f"--pretty=format:{formato}", "--shortstat"],
        cwd=RAIZ, capture_output=True, text=True, check=True,
    ).stdout
    commits = []
    for bloco in saida.split(MARCA)[1:]:
        cabecalho, *resto = bloco.strip("\n").split("\n")
        hash_, autor, dia, assunto = (cabecalho.split(SEP) + [""] * 4)[:4]
        stat = " ".join(resto)
        nums = {k: int(v) for v, k in re.findall(r"(\d+) (file|insertion|deletion)", stat)}
        commits.append({
            "hash": hash_, "autor": autor, "data": dia, "assunto": assunto, "features": extrair_features(assunto),
            "arquivos": nums.get("file", 0), "mais": nums.get("insertion", 0), "menos": nums.get("deletion", 0),
        })
    return commits


def secao_por_feature(commits: list[dict], equipe: dict) -> list[str]:
    """Tabela feature -> dono -> commits do dono / de outros, mais o resumo por pessoa."""
    donos = {fid: f["dono"] for fid, f in equipe["features"].items()}
    L = [
        "## Visão por feature", "",
        "Commits com a tag `[Fn]` no início da mensagem. \"Do dono\" conta commits cujo autor é o dono da feature em "
        "`config/equipe.yaml`; \"de apoio\" são os de outras pessoas na mesma feature.", "",
        "| Feature | Título | Dono | Commits do dono | Commits de apoio | Situação |",
        "|---|---|---|---:|---:|---|",
    ]
    for fid in sorted(equipe["features"], key=lambda x: int(x[1:])):
        f = equipe["features"][fid]
        cs = [c for c in commits if fid in c["features"]]
        do_dono = sum(c["autor"] == f["dono"] for c in cs)
        apoio = len(cs) - do_dono
        situacao = "✅ com commits do dono" if do_dono else "⬜ sem commits do dono"
        L.append(f"| {fid} | {f['titulo']} | {f['dono']} | {do_dono} | {apoio} | {situacao} |")
    L.append("")

    L += ["### Por integrante", "", "| Integrante | Features | Commits nas próprias features |", "|---|---|---:|"]
    for i in equipe["integrantes"]:
        meus = sorted((fid for fid, d in donos.items() if d == i["nome"]), key=lambda x: int(x[1:]))
        n = sum(1 for c in commits if c["autor"] == i["nome"] and any(x in meus for x in c["features"]))
        L.append(f"| {i['nome']} | {', '.join(meus) or '—'} | {n} |")
    sem_tag = [c for c in commits if not c["features"]]
    L += ["", f"{len(sem_tag)} commit(s) sem tag de feature (docs, equipe, correções).", ""]

    for fid in sorted(equipe["features"], key=lambda x: int(x[1:])):
        cs = [c for c in commits if fid in c["features"]]
        if cs:
            L += [f"#### {fid} · {equipe['features'][fid]['titulo']}", ""]
            L += [f"- `{c['hash']}` {c['data']} · {c['autor']} · {c['assunto']}" for c in cs]
            L.append("")
    return L


def renderizar(commits: list[dict], equipe: dict | None = None) -> str:
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

    if equipe:
        L += secao_por_feature(commits, equipe)

    L += ["## Linha do tempo", "", "| Data | Commit | Integrante | Mensagem | Arquivos |", "|---|---|---|---|---:|"]
    for c in commits:
        L.append(f"| {c['data']} | `{c['hash']}` | {c['autor']} | {c['assunto']} | {c['arquivos']} |")
    L.append("")

    for autor, cs in por_autor.items():
        L += [f"## Contribuições de {autor}", ""]
        L += [f"- `{c['hash']}` {c['data']} · {c['assunto']}" for c in cs]
        L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    texto = renderizar(git_log(), carregar_equipe())
    if args.stdout:
        print(texto)
    else:
        SAIDA.write_text(texto, encoding="utf-8")
        print(f"Gravado: {SAIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
