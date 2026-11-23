"""Feature F8 (critério 8): eleição do vencedor, promoção controlada e previsões.

Dono: ver ``config/equipe.yaml``. Pronto quando: ver ``docs/divisao_de_tarefas.md``.

Fluxo do notebook 04
  1. ``eleger_vencedor``: entre A e B vence a maior média da métrica principal no CV do dev, que a F5 já
     calculou em ``resultados_competicao.json``. Empate: vence o menor desvio entre os folds.
  2. O vencedor (treinado no dev) e o vigente (o arquivo salvo no ciclo em que foi promovido) são
     avaliados no **teste do ciclo atual**, os mesmos registros para os dois (``avaliar_no_teste``).
  3. ``decidir_promocao`` aplica a regra de ``cfg["promocao"]``, fixada antes de rodar:
       * sem vigente (primeiro ciclo): promove se o vencedor passar no piso;
       * com vigente: promove só se passar no piso **e** superar o vigente em pelo menos
         ``min_melhoria_abs`` (``src.avaliacao.melhorou``); senão o vigente é mantido.
  4. ``guardar_modelo``: o vencedor de todo ciclo, promovido ou não, ganha uma cópia comprimida em
     ``models/registro/``. Nada ali é apagado. O workflow (F7) envia esses arquivos para a Release do
     GitHub e baixa de lá o vigente antes do notebook 04, porque o runner é efêmero.
  5. ``atualizar_manifesto`` grava ``models/modelo_vigente.json``: o vigente e o histórico de todas as
     decisões, com identificação, ``data_version``, features, métricas, ``run_id`` e motivo.

Fluxo do notebook 05
  * Carrega o vigente pelo manifesto (o arquivo é o Pipeline completo da F5: pré-processamento,
    seleção das features eleitas e algoritmo).
  * Prevê o **próximo lote de** ``data/incoming``: reservas que ainda não entraram no pipeline e, por
    isso, não participaram da escolha do modelo. Sem lote restante, o notebook avisa e mostra as últimas
    previsões feitas (``ultimas_previsoes``), sem gravar nada novo.
  * Confere as previsões de ciclos anteriores cujo lote já foi incorporado (o rótulo chegou).

Falhas
  Qualquer inconsistência (resultado de outro ciclo, vigente registrado mas sem arquivo, ciclo já
  decidido) levanta exceção antes de gravar o manifesto, então o notebook falha e nada é promovido.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.avaliacao import melhorou, metricas_classificacao
from src.config import RAIZ, resolver
from src.ingestao import _listar_arquivos, _lote_id, carregar_manifesto as carregar_manifesto_lotes
from src.preprocessamento import limpar_e_engenhar, lista_features

DIR_REGISTRO = "models/registro"
DIR_EVIDENCIAS_PROMOCAO = "evidencias/promocao"
DIR_EVIDENCIAS_PREVISOES = "evidencias/previsoes"
COMPRESSAO = 3          # joblib: o modelo B cai de ~50 MB para ~19 MB e carrega em < 1 s
LIMIAR_CLASSE = 0.5     # o mesmo corte usado pela F5 em avaliar_cv
TOLERANCIA = 1e-12      # a mesma de src.avaliacao.melhorou
SCHEMA_MANIFESTO = 1


def _agora_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _regra(cfg: dict) -> dict:
    p = cfg["promocao"]
    if p["direcao"] not in ("maximizar", "minimizar"):
        raise ValueError("promocao.direcao deve ser 'maximizar' ou 'minimizar'")
    return {
        "metrica": p["metrica"],
        "direcao": p["direcao"],
        "min_melhoria_abs": float(p["min_melhoria_abs"]),
        "piso_metrica": float(p["piso_metrica"]),
    }


# ---------------------------------------------------------------------------------------------------------
# Notebook 04: eleição, comparação e decisão
# ---------------------------------------------------------------------------------------------------------

def validar_resultados(resultados: dict, ctx: dict, cfg: dict) -> None:
    """Falha se ``resultados_competicao.json`` é de outro ciclo ou usou outra métrica."""
    for chave in ("cycle_id", "data_version"):
        if resultados.get(chave) != ctx.get(chave):
            raise ValueError(
                f"resultados_competicao.json é de outro ciclo ({chave}: {resultados.get(chave)} != {ctx.get(chave)}). "
                "Rode o notebook 03 neste ciclo."
            )
    metrica = _regra(cfg)["metrica"]
    if resultados.get("metrica_principal") != metrica:
        raise ValueError(f"A F5 comparou por {resultados.get('metrica_principal')}, mas a promoção usa {metrica}")
    if len(resultados.get("modelos", {})) < 2:
        raise ValueError("resultados_competicao.json precisa trazer os dois candidatos (A e B)")


def eleger_vencedor(resultados: dict[str, dict], cfg: dict) -> str:
    """Retorna o nome do candidato (``'A'`` ou ``'B'``) com a melhor média no CV do dev.

    Empate na média (dentro de ``TOLERANCIA``): vence o menor desvio entre os folds. Se o desvio também
    empatar, vence o primeiro em ordem alfabética, para a escolha ser sempre a mesma.
    """
    regra = _regra(cfg)
    metrica, sinal = regra["metrica"], (1 if regra["direcao"] == "maximizar" else -1)
    candidatos = []
    for nome, item in resultados["modelos"].items():
        cv = item["cv"]
        candidatos.append((nome, sinal * float(cv["media"][metrica]), float(cv["desvio"][metrica])))
    melhor = max(c[1] for c in candidatos)
    empatados = [c for c in candidatos if melhor - c[1] <= TOLERANCIA]
    menor_desvio = min(c[2] for c in empatados)
    return sorted(c[0] for c in empatados if c[2] - menor_desvio <= TOLERANCIA)[0]


def avaliar_no_teste(modelo, teste: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """Métricas de ``src.avaliacao`` do modelo já ajustado sobre o teste do ciclo."""
    X, y = teste[lista_features(cfg)], teste[cfg["avaliacao"]["alvo"]]
    y_score = modelo.predict_proba(X)[:, 1]
    return metricas_classificacao(y, (y_score >= LIMIAR_CLASSE).astype(int), y_score)


def _passa_no_piso(valor: float, regra: dict) -> bool:
    if regra["direcao"] == "maximizar":
        return valor >= regra["piso_metrica"] - TOLERANCIA
    return valor <= regra["piso_metrica"] + TOLERANCIA


def decidir_promocao(vencedor: dict, vigente: dict | None, cfg: dict) -> dict:
    """Aplica a regra de ``cfg["promocao"]`` e devolve a decisão com o motivo.

    ``vencedor`` e ``vigente`` trazem ``metricas_teste`` medidas nos mesmos registros (teste do ciclo).
    ``vigente`` é ``None`` no primeiro ciclo. Retorna ``{"decisao": "promovido"|"mantido", "motivo",
    "regra", "valor_vencedor", "valor_vigente"}``.
    """
    regra = _regra(cfg)
    metrica = regra["metrica"]
    valor = float(vencedor["metricas_teste"][metrica])
    valor_vig = None if vigente is None else float(vigente["metricas_teste"][metrica])
    piso_ok = _passa_no_piso(valor, regra)

    if vigente is None:
        if piso_ok:
            decisao, motivo = "promovido", f"primeiro ciclo sem vigente; {metrica}={valor:.4f} passa no piso {regra['piso_metrica']}"
        else:
            decisao, motivo = "mantido", (
                f"primeiro ciclo sem vigente; {metrica}={valor:.4f} não passa no piso {regra['piso_metrica']}: "
                "nenhum modelo foi promovido e o pipeline segue sem vigente"
            )
    elif not piso_ok:
        decisao, motivo = "mantido", f"vencedor com {metrica}={valor:.4f} não passa no piso {regra['piso_metrica']}"
    elif melhorou(valor, valor_vig, regra["direcao"], regra["min_melhoria_abs"]):
        decisao, motivo = "promovido", (
            f"{metrica} {valor:.4f} supera o vigente ({valor_vig:.4f}) em {abs(valor - valor_vig):.4f}, "
            f"acima da melhoria mínima {regra['min_melhoria_abs']}"
        )
    else:
        decisao, motivo = "mantido", (
            f"{metrica} {valor:.4f} contra {valor_vig:.4f} do vigente: a diferença não alcança a melhoria "
            f"mínima de {regra['min_melhoria_abs']} na direção '{regra['direcao']}'"
        )
    return {"decisao": decisao, "motivo": motivo, "regra": regra, "valor_vencedor": valor, "valor_vigente": valor_vig}


# ---------------------------------------------------------------------------------------------------------
# Registro dos modelos e manifesto
# ---------------------------------------------------------------------------------------------------------

def nome_arquivo_modelo(cycle_id: str, nome_modelo: str) -> str:
    """Nome único do vencedor do ciclo. É também o nome do asset na Release do GitHub."""
    return f"modelo_{cycle_id}_{nome_modelo}.joblib"


def guardar_modelo(modelo, cycle_id: str, nome_modelo: str, raiz: Path = RAIZ) -> Path:
    """Grava a cópia comprimida do vencedor em ``models/registro/``. Nunca sobrescreve um arquivo existente."""
    destino = resolver(DIR_REGISTRO, raiz) / nome_arquivo_modelo(cycle_id, nome_modelo)
    if destino.exists():
        raise FileExistsError(f"{destino.name} já existe: modelos registrados não são sobrescritos")
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, destino, compress=COMPRESSAO)
    return destino


def caminho_manifesto(cfg: dict, raiz: Path = RAIZ) -> Path:
    return resolver(cfg["promocao"]["manifesto"], raiz)


def carregar_manifesto(cfg: dict, raiz: Path = RAIZ) -> dict:
    """Lê ``models/modelo_vigente.json``. Sem arquivo, devolve um manifesto vazio (primeiro ciclo)."""
    caminho = caminho_manifesto(cfg, raiz)
    if not caminho.exists():
        return {"schema_version": SCHEMA_MANIFESTO, "vigente": None, "historico": []}
    return json.loads(caminho.read_text(encoding="utf-8"))


def carregar_vigente(manifesto: dict, raiz: Path = RAIZ):
    """Devolve ``(modelo, entrada_do_vigente)`` ou ``(None, None)`` se ainda não há vigente.

    Falha se o manifesto aponta um vigente cujo arquivo não está em ``models/registro/``: comparar sem
    ele levaria a uma promoção indevida.
    """
    vigente = manifesto.get("vigente")
    if not vigente:
        return None, None
    caminho = resolver(DIR_REGISTRO, raiz) / vigente["arquivo"]
    if not caminho.is_file():
        raise FileNotFoundError(
            f"O manifesto aponta o vigente {vigente['arquivo']}, mas o arquivo não está em {DIR_REGISTRO}/. "
            "No Actions, ele é baixado da Release antes do notebook 04."
        )
    return joblib.load(caminho), vigente


def atualizar_manifesto(registro: dict, cfg: dict, raiz: Path = RAIZ) -> dict:
    """Acrescenta a decisão do ciclo ao histórico e, se promovido, troca o vigente. Devolve o manifesto.

    ``registro`` precisa de ``cycle_id``, ``decisao`` e ``candidato`` (com ``arquivo``). O histórico só
    cresce, e um ciclo já decidido não pode ser decidido de novo.
    """
    for chave in ("cycle_id", "data_version", "decisao", "motivo", "candidato"):
        if chave not in registro:
            raise ValueError(f"Registro da decisão sem '{chave}'")
    if registro["decisao"] not in ("promovido", "mantido"):
        raise ValueError(f"Decisão inválida: {registro['decisao']}")
    manifesto = carregar_manifesto(cfg, raiz)
    if any(h["cycle_id"] == registro["cycle_id"] for h in manifesto["historico"]):
        raise ValueError(f"O ciclo {registro['cycle_id']} já tem decisão registrada no manifesto")

    manifesto["historico"].append(registro)
    if registro["decisao"] == "promovido":
        candidato = registro["candidato"]
        manifesto["vigente"] = {
            "cycle_id": registro["cycle_id"],
            "data_version": registro["data_version"],
            "modelo": candidato["modelo"],
            "algoritmo": candidato["algoritmo"],
            "params": candidato["params"],
            "features": candidato["features"],
            "arquivo": candidato["arquivo"],
            "metrica": registro["regra"]["metrica"],
            "valor_teste_na_promocao": registro["valor_vencedor"],
            "run_id_promocao": registro.get("run_id_promocao"),
            "run_id_treino": candidato.get("run_id_treino"),
            "promovido_em": registro.get("decidido_em"),
        }
    manifesto["atualizado_em"] = _agora_iso()

    caminho = caminho_manifesto(cfg, raiz)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifesto


def salvar_evidencia_decisao(registro: dict, raiz: Path = RAIZ) -> Path:
    """Cópia da decisão do ciclo em ``evidencias/promocao/decisao_<cycle_id>.json``."""
    destino = resolver(DIR_EVIDENCIAS_PROMOCAO, raiz) / f"decisao_{registro['cycle_id']}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(registro, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return destino


def tabela_historico(manifesto: dict) -> pd.DataFrame:
    """Uma linha por ciclo decidido: o que mudou de um ciclo para o outro."""
    linhas = []
    for h in manifesto.get("historico", []):
        linhas.append({
            "cycle_id": h["cycle_id"],
            "data_version": h["data_version"],
            "vencedor": h["candidato"]["modelo"],
            "algoritmo": h["candidato"]["algoritmo"],
            "valor_vencedor": h.get("valor_vencedor"),
            "vigente_anterior": (h.get("vigente_anterior") or {}).get("cycle_id"),
            "valor_vigente": h.get("valor_vigente"),
            "decisao": h["decisao"],
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------------------------------------
# Notebook 05: previsões no próximo lote
# ---------------------------------------------------------------------------------------------------------

def proximo_lote(cfg: dict, raiz: Path = RAIZ) -> Path | None:
    """O próximo arquivo de ``data/incoming``, na mesma ordem em que ``src.ingestao`` libera os lotes."""
    arquivos = _listar_arquivos(resolver(cfg["dados"]["dir_incoming"], raiz))
    return arquivos[0] if arquivos else None


def preparar_lote(arquivo: Path, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Lê um lote bruto e aplica o mesmo ETL do notebook 01 (limpeza e atributos derivados).

    O lote simulado traz o rótulo, porque o dataset é histórico. O ETL exige a coluna, mas o rótulo
    não é usado na previsão nem gravado com ela.
    """
    bruto = pd.read_csv(arquivo)
    bruto.insert(0, "linha_origem", range(len(bruto)))
    bruto.insert(0, "lote_id", _lote_id(arquivo))  # mesmo id que src.ingestao dá ao lote quando ele chega
    return limpar_e_engenhar(bruto, cfg)


def prever(modelo, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Probabilidade de cancelamento e classe prevista, com as colunas de rastreio do registro."""
    y_score = modelo.predict_proba(df[lista_features(cfg)])[:, 1]
    meta = [c for c in cfg["features"]["meta"] if c in df.columns]
    saida = df[meta + [cfg["avaliacao"]["coluna_tempo"]]].copy()
    saida["prob_cancelamento"] = y_score
    saida["previsao"] = (y_score >= LIMIAR_CLASSE).astype(int)
    return saida


def salvar_previsoes(previsoes: pd.DataFrame, ctx: dict, vigente: dict, raiz: Path = RAIZ) -> Path:
    """Grava ``evidencias/previsoes/previsoes_<cycle_id>.csv`` com o ciclo e o modelo que previu."""
    destino = resolver(DIR_EVIDENCIAS_PREVISOES, raiz) / f"previsoes_{ctx['cycle_id']}.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    previsoes.assign(
        cycle_id=ctx["cycle_id"],
        data_version=ctx["data_version"],
        vigente_cycle_id=vigente["cycle_id"],
        vigente_modelo=vigente["modelo"],
    ).to_csv(destino, index=False)
    return destino


def ultimas_previsoes(raiz: Path = RAIZ) -> tuple[Path | None, pd.DataFrame | None]:
    """As previsões gravadas pelo ciclo mais recente que ainda tinha lote futuro para prever.

    Serve para o notebook 05 quando ``data/incoming`` já acabou: em vez de ficar sem nada para mostrar, ele
    exibe (avisando) as últimas previsões feitas. O ``cycle_id`` começa pelo horário UTC, então a ordem
    alfabética dos arquivos é a ordem cronológica.
    """
    arquivos = sorted(resolver(DIR_EVIDENCIAS_PREVISOES, raiz).glob("previsoes_*.csv"))
    if not arquivos:
        return None, None
    return arquivos[-1], pd.read_csv(arquivos[-1])


def conferir_previsoes_anteriores(cfg: dict, raiz: Path = RAIZ) -> pd.DataFrame:
    """Compara previsões já gravadas com o rótulo real dos lotes que entraram depois no pipeline.

    Só confere lotes já registrados no manifesto de lotes (``data/raw/manifest.json``), isto é, que
    chegaram em um ciclo posterior ao da previsão. Uma linha por arquivo de previsões conferido.
    """
    pasta = resolver(DIR_EVIDENCIAS_PREVISOES, raiz)
    if not pasta.is_dir():
        return pd.DataFrame()
    lotes = carregar_manifesto_lotes(resolver(cfg["dados"]["manifesto"], raiz))
    incorporados = {l["lote_id"]: l["arquivo"] for l in lotes["lotes"]}
    dir_raw, alvo = resolver(cfg["dados"]["dir_raw"], raiz), cfg["avaliacao"]["alvo"]
    linhas = []
    for arquivo in sorted(pasta.glob("previsoes_*.csv")):
        prev = pd.read_csv(arquivo)
        lote_ids = prev["lote_id"].unique()
        if len(lote_ids) != 1 or lote_ids[0] not in incorporados:
            continue
        reais = pd.read_csv(dir_raw / incorporados[lote_ids[0]], usecols=[alvo])
        reais["linha_origem"] = range(len(reais))
        junto = prev.merge(reais, on="linha_origem", how="inner")
        metricas = metricas_classificacao(junto[alvo], junto["previsao"], junto["prob_cancelamento"])
        linhas.append({
            "arquivo": arquivo.name,
            "cycle_id": prev["cycle_id"].iloc[0],
            "vigente_cycle_id": prev["vigente_cycle_id"].iloc[0],
            "lote_id": lote_ids[0],
            "n": len(junto),
            **metricas,
        })
    return pd.DataFrame(linhas)
