"""Feature F6 (critério 6): MLflow e rastreabilidade.

Dono: ver ``config/equipe.yaml``. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Papel: oferecer às demais features uma forma única de registrar um ciclo no MLflow, sem que F3, F4, F5
e F8 precisem conhecer detalhes do MLflow. Parâmetros em ``cfg["mlflow"]``.

Como fica guardado
  Tudo mora numa única pasta, ``cfg["mlflow"]["dir_estado"]`` (padrão ``mlflow_estado/``):
    * ``mlflow.db``  banco SQLite com experimentos, runs, tags, parâmetros e métricas;
    * ``artefatos/`` arquivos anexados aos runs (tabelas, JSONs, modelos).
  Copiar essa pasta é copiar o histórico inteiro. É ela que o workflow (F7) precisa guardar e recuperar
  entre execuções, porque o runner do GitHub Actions é apagado no fim. Proposta em ``docs/f6_mlflow.md``.

Por que existe ``_relocar_artefatos``
  O MLflow grava no banco o caminho ABSOLUTO dos artefatos (``/home/runner/...`` no Actions,
  ``C:/...`` no Windows). Sem correção, o histórico criado numa máquina perde os artefatos ao ser aberto
  em outra. Antes de abrir o banco, os caminhos antigos são trocados pelo caminho atual da pasta.

Contrato para as outras features
  * ``iniciar_run(nome, ctx, cfg)`` abre um run com ``cycle_id`` e ``data_version`` como tags. Use com
    ``with``; o ``run_id`` está em ``run.info.run_id``.
  * Dentro do run: ``registrar_parametros``, ``registrar_metricas``, ``registrar_features``,
    ``registrar_artefatos``, ``registrar_decisao`` (F8) e ``registrar_modelo`` (F5).
  * ``registrar_votacao(...)`` faz o registro completo do notebook 02 (F3 + F4) e devolve o ``run_id``.
  * ``historico_ciclos(cfg)`` devolve os runs de todos os ciclos, para consulta e para o relatório.

Exemplo (F5)::

    with iniciar_run("03_modelo_A", ctx, cfg) as run:
        registrar_parametros({"modelo": cfg["modelos"]["A"]})
        registrar_metricas(resultado["metricas"])
        registrar_features(eleitas)
        registrar_modelo(caminho_joblib)
    run_id = run.info.run_id
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silencia uma dica do mlflow 3.x que polui os logs
import mlflow  # noqa: E402  (precisa vir depois da variável de ambiente acima)

from src.config import RAIZ, resolver  # noqa: E402

NOME_BANCO = "mlflow.db"
DIR_ARTEFATOS = "artefatos"
DIR_EVIDENCIAS = "evidencias/mlflow"
TAGS_OBRIGATORIAS = ("cycle_id", "data_version")


# ---------------------------------------------------------------------------------------------------------
# Configuração do armazenamento
# ---------------------------------------------------------------------------------------------------------
def diretorio_estado(cfg: dict, raiz: Path = RAIZ) -> Path:
    """Pasta que guarda todo o histórico do MLflow (banco + artefatos)."""
    return resolver(cfg["mlflow"]["dir_estado"], raiz)


def _uri_artefatos(estado: Path) -> str:
    """URI no formato que o próprio MLflow grava (``file:C:/...`` no Windows, ``file:///...`` no Linux)."""
    caminho = (estado / DIR_ARTEFATOS).resolve().as_posix()
    return f"file://{caminho}" if caminho.startswith("/") else f"file:{caminho}"


def _relocar_artefatos(banco: Path, uri_nova: str) -> int:
    """Troca o caminho antigo dos artefatos pelo atual em experimentos e runs. Devolve quantos runs mudaram.

    Só mexe em experimentos cuja pasta de artefatos termina em ``/artefatos`` (os criados por este módulo).
    """
    if not banco.exists():
        return 0
    con = sqlite3.connect(banco)
    try:
        tabelas = {t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"experiments", "runs"} <= tabelas:
            return 0
        alterados = 0
        for exp_id, uri_antiga in con.execute("SELECT experiment_id, artifact_location FROM experiments").fetchall():
            if not uri_antiga or uri_antiga == uri_nova or not uri_antiga.endswith(f"/{DIR_ARTEFATOS}"):
                continue
            con.execute("UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?", (uri_nova, exp_id))
            cur = con.execute(
                "UPDATE runs SET artifact_uri = ? || substr(artifact_uri, length(?) + 1) "
                "WHERE experiment_id = ? AND substr(artifact_uri, 1, length(?)) = ?",
                (uri_nova, uri_antiga, exp_id, uri_antiga, uri_antiga),
            )
            alterados += cur.rowcount
        con.commit()
        return alterados
    finally:
        con.close()


def configurar_mlflow(cfg: dict, raiz: Path = RAIZ) -> str:
    """Aponta o MLflow para a pasta de estado e garante o experimento. Devolve o ``experiment_id``."""
    estado = diretorio_estado(cfg, raiz)
    (estado / DIR_ARTEFATOS).mkdir(parents=True, exist_ok=True)
    banco = estado / NOME_BANCO
    uri_artefatos = _uri_artefatos(estado)
    _relocar_artefatos(banco, uri_artefatos)

    mlflow.set_tracking_uri(f"sqlite:///{banco.resolve().as_posix()}")
    nome = cfg["mlflow"]["experimento"]
    experimento = mlflow.get_experiment_by_name(nome)
    if experimento is None:
        exp_id = mlflow.create_experiment(nome, artifact_location=uri_artefatos)
    else:
        exp_id = experimento.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


# ---------------------------------------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------------------------------------
def _commit_git(raiz: Path) -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "desconhecido"


def _tags_do_ciclo(nome: str, ctx: dict, raiz: Path) -> dict[str, str]:
    faltando = [t for t in TAGS_OBRIGATORIAS if not ctx.get(t)]
    if faltando:
        raise ValueError(f"Contexto do ciclo sem {faltando}: rode o notebook 01 antes de registrar no MLflow.")
    tags = {
        "cycle_id": ctx["cycle_id"],
        "data_version": ctx["data_version"],
        "etapa": nome,
        "origem": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
        "git_commit": _commit_git(raiz),
    }
    if os.environ.get("GITHUB_RUN_ID"):
        tags["github_run_id"] = os.environ["GITHUB_RUN_ID"]
    return tags


def iniciar_run(nome: str, ctx: dict, cfg: dict, raiz: Path = RAIZ):
    """Abre um run no MLflow já com ``cycle_id`` e ``data_version`` como tags. Devolve o run ativo.

    Use com ``with``: o run é fechado (com status FAILED se houver exceção) ao sair do bloco.
    O contexto completo do ciclo também é anexado, em ``contexto/ciclo_atual.json``.
    """
    tags = _tags_do_ciclo(nome, ctx, raiz)  # valida antes de tocar no MLflow
    configurar_mlflow(cfg, raiz)
    run = mlflow.start_run(run_name=nome, tags=tags)
    mlflow.log_dict(ctx, "contexto/ciclo_atual.json")
    return run


def _exigir_run_ativo() -> None:
    if mlflow.active_run() is None:
        raise RuntimeError("Nenhum run ativo: chame dentro de `with iniciar_run(...)`.")


def achatar(dados: Mapping, prefixo: str = "") -> dict[str, str]:
    """``{"rf": {"n_estimators": 200}}`` vira ``{"rf.n_estimators": "200"}`` (o MLflow só aceita chave e valor simples)."""
    plano = {}
    for chave, valor in dados.items():
        nome = f"{prefixo}{chave}"
        if isinstance(valor, Mapping):
            plano.update(achatar(valor, f"{nome}."))
        else:
            plano[nome] = str(valor)
    return plano


def registrar_parametros(parametros: Mapping) -> None:
    """Registra parâmetros (aceita dicionários aninhados, como seções do ``config.yaml``)."""
    _exigir_run_ativo()
    mlflow.log_params(achatar(parametros))


def registrar_metricas(metricas: Mapping[str, float]) -> None:
    """Registra métricas numéricas. Valores ``None`` são ignorados."""
    _exigir_run_ativo()
    mlflow.log_metrics({k: float(v) for k, v in metricas.items() if v is not None})


def registrar_features(features: list[str]) -> None:
    """Registra a lista de features usada no run (artefato ``features.json`` e parâmetro ``n_features``)."""
    _exigir_run_ativo()
    mlflow.log_param("n_features", len(features))
    mlflow.log_dict({"features": list(features)}, "features.json")


def registrar_artefatos(caminhos: Iterable[Path] | Mapping[str, Path], cfg: dict | None = None, pasta: str | None = None) -> None:
    """Anexa arquivos (tabela de votos, modelo serializado, métricas) ao run ativo.

    Aceita uma lista de caminhos ou o dicionário devolvido por ``salvar_rankings``/``salvar_votacao``.
    ``cfg`` é aceito por compatibilidade com o esqueleto original e não é usado.
    """
    _exigir_run_ativo()
    lista = list(caminhos.values()) if isinstance(caminhos, Mapping) else list(caminhos)
    for caminho in map(Path, lista):
        if not caminho.is_file():
            raise FileNotFoundError(f"Artefato não encontrado: {caminho}")
        mlflow.log_artifact(str(caminho), artifact_path=pasta)


def registrar_modelo(caminho_modelo: Path) -> None:
    """Anexa o modelo serializado (``.joblib``) em ``modelo/`` (F5)."""
    registrar_artefatos([caminho_modelo], pasta="modelo")


def registrar_decisao(decisao: Mapping) -> None:
    """Registra a decisão de promoção ou manutenção (F8): tag ``decisao`` e o JSON completo."""
    _exigir_run_ativo()
    if "decisao" in decisao:
        mlflow.set_tag("decisao", str(decisao["decisao"]))
    mlflow.log_dict(dict(decisao), "decisao_promocao.json")


# ---------------------------------------------------------------------------------------------------------
# Registro do notebook 02 (seleção F3 + votação F4)
# ---------------------------------------------------------------------------------------------------------
def registrar_votacao(
    ctx: dict,
    cfg: dict,
    tabela_votos: pd.DataFrame,
    eleitas: list[str],
    caminhos_rankings: Mapping[str, Path],
    caminhos_votacao: Mapping[str, Path],
    raiz: Path = RAIZ,
    nome: str = "02_votacao",
) -> str:
    """Registra rankings, votos, features eleitas e parâmetros da votação num run. Devolve o ``run_id``."""
    with iniciar_run(nome, ctx, cfg, raiz) as run:
        registrar_parametros({
            "selecao": cfg["selecao"],
            "seed": cfg["projeto"]["seed"],
            "cv.n_splits": cfg["avaliacao"]["cv"]["n_splits"],
            "dev_n": ctx.get("particao", {}).get("dev_n"),
        })
        votos = tabela_votos["votos"].value_counts()
        n_metodos = sum(c.startswith("voto_") for c in tabela_votos.columns)
        registrar_metricas({
            "n_features_candidatas": len(tabela_votos),
            "n_features_eleitas": len(eleitas),
            **{f"features_com_{v}_votos": int(votos.get(v, 0)) for v in range(n_metodos + 1)},
        })
        registrar_features(eleitas)
        registrar_artefatos(caminhos_rankings, pasta="selecao")
        registrar_artefatos(caminhos_votacao, pasta="votacao")
    return run.info.run_id


# ---------------------------------------------------------------------------------------------------------
# Consulta do histórico
# ---------------------------------------------------------------------------------------------------------
COLUNAS_HISTORICO = ["cycle_id", "data_version", "etapa", "run_id", "status", "inicio", "origem", "decisao"]


def historico_ciclos(cfg: dict, raiz: Path = RAIZ, cycle_id: str | None = None) -> pd.DataFrame:
    """Uma linha por run, com ``cycle_id``, ``data_version``, etapa, ``run_id`` e as métricas registradas.

    Com ``cycle_id``, devolve só os runs daquele ciclo.
    """
    configurar_mlflow(cfg, raiz)
    filtro = f"tags.cycle_id = '{cycle_id}'" if cycle_id else ""
    runs = mlflow.search_runs(experiment_names=[cfg["mlflow"]["experimento"]], filter_string=filtro)
    if runs.empty:
        return pd.DataFrame(columns=COLUNAS_HISTORICO)
    renomear = {f"tags.{t}": t for t in ("cycle_id", "data_version", "etapa", "origem", "decisao")}
    renomear |= {"start_time": "inicio"}
    runs = runs.rename(columns=renomear)
    for coluna in COLUNAS_HISTORICO:
        if coluna not in runs:
            runs[coluna] = None
    metricas = sorted(c for c in runs.columns if c.startswith("metrics."))
    return runs[COLUNAS_HISTORICO + metricas].sort_values("inicio").reset_index(drop=True)


def exportar_historico(cfg: dict, raiz: Path = RAIZ) -> Path:
    """Grava o histórico em ``evidencias/mlflow/historico_runs.csv`` (evidência para o relatório)."""
    destino = resolver(DIR_EVIDENCIAS, raiz)
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / "historico_runs.csv"
    historico_ciclos(cfg, raiz).to_csv(caminho, index=False)
    return caminho


def main() -> int:
    """``python -m src.rastreio``: mostra os ciclos registrados e exporta o histórico."""
    from src.config import carregar_config

    cfg = carregar_config()
    historico = historico_ciclos(cfg)
    if historico.empty:
        print("Nenhum run registrado ainda.")
        return 0
    pd.set_option("display.width", 200)
    print(historico[["cycle_id", "data_version", "etapa", "run_id", "status"]].to_string(index=False))
    print(f"\n{historico['cycle_id'].nunique()} ciclo(s), {len(historico)} run(s)")
    print("exportado:", exportar_historico(cfg).relative_to(RAIZ))
    banco = (diretorio_estado(cfg) / NOME_BANCO).relative_to(RAIZ).as_posix()
    print(f"interface: mlflow ui --backend-store-uri sqlite:///{banco}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
