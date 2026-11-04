"""Entrada e versionamento de dados (Etapa 2 do fluxo: "dados novos e versão").

Modelo de dados:
  * Cada lote é um arquivo ``lote_AAAA-MM.csv.gz`` com as reservas que chegam em um mês.
  * ``data/raw/manifest.json`` lista os lotes já incorporados, com sha256 e contagem de linhas.
  * ``data/incoming/`` guarda lotes "futuros" para simular a chegada de dados novos.

Garantias:
  * Idempotência: registrar duas vezes o mesmo lote (ou o mesmo conteúdo com outro nome) não duplica dados.
  * Integridade: se um lote registrado for alterado ou sumir, o pipeline falha.
  * ``data_version`` é determinístico: depende apenas do conteúdo dos lotes incorporados.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import RAIZ, carregar_config, resolver
from src.schema import COLUNAS_BRUTAS, construir_data_chegada

SUFIXOS = (".csv", ".csv.gz")
SCHEMA_MANIFESTO = 1


def _agora_iso(agora: datetime | None = None) -> str:
    return (agora or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lote_id(arquivo: Path) -> str:
    nome = arquivo.name
    for sufixo in SUFIXOS:
        if nome.endswith(sufixo):
            return nome[: -len(sufixo)]
    return arquivo.stem


def _listar_arquivos(diretorio: Path) -> list[Path]:
    if not diretorio.exists():
        return []
    return sorted(p for p in diretorio.iterdir() if p.is_file() and p.name.endswith(SUFIXOS))


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def carregar_manifesto(caminho: Path) -> dict:
    if not caminho.exists():
        return {"schema_version": SCHEMA_MANIFESTO, "lotes": []}
    return json.loads(caminho.read_text(encoding="utf-8"))


def salvar_manifesto(manifesto: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")


def _validar_lote(df: pd.DataFrame, arquivo: Path) -> None:
    faltando = [c for c in COLUNAS_BRUTAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Lote {arquivo.name} sem as colunas esperadas: {faltando}")
    if df.empty:
        raise ValueError(f"Lote {arquivo.name} está vazio")


def registrar_lotes_novos(dir_raw: Path, caminho_manifesto: Path, agora: datetime | None = None) -> dict:
    """Registra no manifesto os arquivos de ``dir_raw`` que ainda não foram incorporados.

    Retorna ``{"novos": [...], "duplicados_ignorados": [...]}``. Um arquivo cujo conteúdo (sha256)
    já esteja no manifesto é ignorado, mesmo que tenha outro nome, evitando dados duplicados.
    """
    manifesto = carregar_manifesto(caminho_manifesto)
    nomes = {l["arquivo"] for l in manifesto["lotes"]}
    shas = {l["sha256"] for l in manifesto["lotes"]}
    novos: list[dict] = []
    duplicados: list[str] = []

    for arquivo in _listar_arquivos(dir_raw):
        if arquivo.name in nomes:
            continue
        sha = sha256_arquivo(arquivo)
        if sha in shas:
            duplicados.append(arquivo.name)
            continue
        df = pd.read_csv(arquivo)
        _validar_lote(df, arquivo)
        chegada = construir_data_chegada(df)
        entrada = {
            "lote_id": _lote_id(arquivo),
            "arquivo": arquivo.name,
            "sha256": sha,
            "n_linhas": int(len(df)),
            "chegada_min": chegada.min().strftime("%Y-%m-%d"),
            "chegada_max": chegada.max().strftime("%Y-%m-%d"),
            "incorporado_em": _agora_iso(agora),
        }
        manifesto["lotes"].append(entrada)
        shas.add(sha)
        novos.append(entrada)

    if novos:
        manifesto["lotes"].sort(key=lambda l: l["arquivo"])
        salvar_manifesto(manifesto, caminho_manifesto)
    return {"novos": novos, "duplicados_ignorados": duplicados}


def liberar_proximo_lote(dir_incoming: Path, dir_raw: Path, quantidade: int = 1) -> list[str]:
    """Move os próximos lotes (em ordem de nome) de ``incoming`` para ``raw``.

    Simula a chegada de dados novos entre um ciclo e outro.
    """
    movidos: list[str] = []
    dir_raw.mkdir(parents=True, exist_ok=True)
    for arquivo in _listar_arquivos(dir_incoming)[:quantidade]:
        shutil.move(str(arquivo), str(dir_raw / arquivo.name))
        movidos.append(arquivo.name)
    return movidos


def verificar_integridade(dir_raw: Path, manifesto: dict) -> None:
    """Falha se algum lote registrado sumiu ou teve o conteúdo alterado."""
    for lote in manifesto["lotes"]:
        arquivo = dir_raw / lote["arquivo"]
        if not arquivo.exists():
            raise FileNotFoundError(f"Lote registrado ausente: {lote['arquivo']}")
        if sha256_arquivo(arquivo) != lote["sha256"]:
            raise ValueError(f"Lote {lote['arquivo']} foi alterado após o registro (sha256 diferente)")


def consolidar_lotes(dir_raw: Path, manifesto: dict) -> pd.DataFrame:
    """Lê todos os lotes registrados e devolve um único DataFrame bruto.

    Acrescenta ``lote_id`` e ``linha_origem`` para rastrear cada registro até o arquivo de origem.
    Não remove duplicatas: isso é decisão do ETL, que precisa diagnosticá-las e justificá-las.
    """
    if not manifesto["lotes"]:
        raise ValueError("Nenhum lote registrado. Coloque arquivos em data/raw e execute src.ingestao.")
    verificar_integridade(dir_raw, manifesto)
    partes = []
    for lote in manifesto["lotes"]:
        df = pd.read_csv(dir_raw / lote["arquivo"])
        df.insert(0, "linha_origem", range(len(df)))
        df.insert(0, "lote_id", lote["lote_id"])
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def calcular_data_version(manifesto: dict) -> str:
    """``dvNNN-<hash8>``: NNN = número de lotes; hash8 = resumo dos sha256 dos lotes."""
    shas = sorted(l["sha256"] for l in manifesto["lotes"])
    resumo = hashlib.sha256("\n".join(shas).encode()).hexdigest()[:8]
    return f"dv{len(shas):03d}-{resumo}"


def gerar_cycle_id(agora: datetime | None = None, ambiente: dict | None = None) -> str:
    """Identificador único da execução: ``AAAAMMDDTHHMMSSZ-<origem>``.

    No GitHub Actions a origem é ``gha<run_id>-<tentativa>``; localmente, ``local``.
    """
    env = os.environ if ambiente is None else ambiente
    carimbo = (agora or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    if env.get("GITHUB_RUN_ID"):
        origem = f"gha{env['GITHUB_RUN_ID']}-{env.get('GITHUB_RUN_ATTEMPT', '1')}"
    else:
        origem = "local"
    return f"{carimbo}-{origem}"


def preparar_dados(cfg: dict, raiz: Path = RAIZ, liberar: int = 0) -> dict:
    """Fluxo completo da etapa: (opcional) libera lotes, registra novos e consolida.

    Devolve ``{"df", "manifesto", "data_version", "cycle_id", "novos", "duplicados_ignorados"}``.
    """
    dir_raw = resolver(cfg["dados"]["dir_raw"], raiz)
    dir_incoming = resolver(cfg["dados"]["dir_incoming"], raiz)
    caminho_manifesto = resolver(cfg["dados"]["manifesto"], raiz)

    liberados = liberar_proximo_lote(dir_incoming, dir_raw, liberar) if liberar else []
    resultado = registrar_lotes_novos(dir_raw, caminho_manifesto)
    manifesto = carregar_manifesto(caminho_manifesto)
    df = consolidar_lotes(dir_raw, manifesto)
    return {
        "df": df,
        "manifesto": manifesto,
        "data_version": calcular_data_version(manifesto),
        "cycle_id": gerar_cycle_id(),
        "liberados": liberados,
        **resultado,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Registra lotes novos e mostra a versão dos dados.")
    ap.add_argument("--liberar", type=int, default=0, help="move N lotes de data/incoming para data/raw antes de registrar")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = carregar_config(args.config)
    r = preparar_dados(cfg, liberar=args.liberar)
    resumo = {
        "data_version": r["data_version"],
        "cycle_id": r["cycle_id"],
        "n_lotes": len(r["manifesto"]["lotes"]),
        "n_linhas_brutas": int(len(r["df"])),
        "liberados": r["liberados"],
        "novos_registrados": [l["lote_id"] for l in r["novos"]],
        "duplicados_ignorados": r["duplicados_ignorados"],
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
