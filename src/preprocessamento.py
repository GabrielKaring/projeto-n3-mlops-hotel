"""ETL, engenharia de atributos, partição temporal e pré-processador (Etapas 1 e 2).

Princípio central: tudo que é "estático" (regras de domínio, imputação por constante, atributos
derivados) é feito aqui, linha a linha, sem olhar para outras linhas. Tudo que "aprende" com os dados
(limites de outliers, médias, categorias) fica dentro de um Pipeline/ColumnTransformer, ajustado
somente no treino — por isso nunca vaza informação do teste.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RAIZ, resolver
from src.schema import (
    ALVO,
    COLUNAS_BRUTAS,
    COLUNAS_NUMERICAS_RESERVA,
    COLUNAS_OPCIONAIS_RESERVA,
    COLUNAS_POS_RESERVA,
    COLUNAS_RESERVA,
    MESES,
    construir_data_chegada,
)

# --------------------------------------------------------------------------- utilidades de config


def colunas_modelagem(cfg: dict) -> dict[str, list[str]]:
    f = cfg["features"]
    return {
        "numericas_cauda_longa": list(f["numericas_cauda_longa"]),
        "numericas_simples": list(f["numericas_simples"]),
        "categoricas": list(f["categoricas"]),
        "binarias": list(f["binarias"]),
    }


def lista_features(cfg: dict) -> list[str]:
    """Todas as colunas de entrada do modelo, antes do pré-processador."""
    return [c for grupo in colunas_modelagem(cfg).values() for c in grupo]


def _colunas_a_remover(cfg: dict) -> list[str]:
    return [c for grupo in cfg["etl"]["remover_colunas"].values() for c in grupo]


# --------------------------------------------------------------------------- validação e diagnóstico


def validar_dados_brutos(df: pd.DataFrame, inferencia: bool = False) -> None:
    """Interrompe o fluxo se o lote não respeitar o esquema mínimo esperado.

    Com ``inferencia=True`` (reserva nova, ainda sem desfecho), só exige as colunas conhecidas no ato da
    reserva (``COLUNAS_RESERVA``) e não olha o alvo.
    """
    exigidas = COLUNAS_RESERVA if inferencia else COLUNAS_BRUTAS
    faltando = [c for c in exigidas if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no dado bruto: {faltando}")
    if not inferencia:
        _validar_alvo(df)
    desconhecidos = set(df["arrival_date_month"].dropna().unique()) - set(MESES)
    if desconhecidos:
        raise ValueError(f"Meses de chegada inválidos: {sorted(desconhecidos)}")


def _validar_alvo(df: pd.DataFrame) -> None:
    if not set(df[ALVO].dropna().unique()) <= {0, 1}:
        raise ValueError(f"A coluna alvo '{ALVO}' deve conter apenas 0 e 1")
    if df[ALVO].isna().any():
        raise ValueError(f"A coluna alvo '{ALVO}' possui valores ausentes")


def completar_reserva_nova(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara reservas novas para o mesmo ETL do treino (modo inferência).

    Uma reserva recém-feita não tem desfecho (``is_canceled``) nem as colunas posteriores à reserva, e pode
    não ter país, agente ou empresa. Essas colunas entram vazias: as posteriores e o alvo são descartados
    pelo ETL antes do modelo, e os campos opcionais recebem a mesma imputação de domínio do treino. Um alvo
    que venha junto é ignorado (apagado), para nunca influenciar a previsão.
    """
    validar_dados_brutos(df, inferencia=True)
    df = df.copy()
    for c in [ALVO, *COLUNAS_POS_RESERVA]:
        df[c] = np.nan
    for c in COLUNAS_OPCIONAIS_RESERVA:
        if c not in df.columns:
            df[c] = np.nan
    vazios = [c for c in COLUNAS_RESERVA if df[c].isna().any()]
    if vazios:
        raise ValueError(f"Campos obrigatórios sem valor: {vazios}")
    for c in COLUNAS_NUMERICAS_RESERVA:
        df[c] = _converter_numerica(df[c])   # "110.5" vira 110.5; null do JSON nos opcionais vira NaN, como no CSV
    _validar_data_chegada(df)
    return df


def _posicoes(mascara: pd.Series) -> list[int]:
    """Posições (a partir de 1) das reservas marcadas, na ordem da entrada."""
    return [i + 1 for i, marcada in enumerate(mascara.to_numpy()) if marcada]


def _converter_numerica(serie: pd.Series) -> pd.Series:
    """Converte um campo numérico de reserva nova; texto que não é número interrompe com o campo e a posição."""
    convertida = pd.to_numeric(serie, errors="coerce")
    invalidas = convertida.isna() & serie.notna()
    if invalidas.any():
        exemplo = serie[invalidas].iloc[0]
        raise ValueError(
            f"O campo '{serie.name}' deve ser numérico (reserva(s) na posição {_posicoes(invalidas)}, "
            f"valor recebido: {exemplo!r})"
        )
    return convertida


def _validar_data_chegada(df: pd.DataFrame) -> None:
    """Ano e dia inteiros e uma data que existe no calendário (31 de setembro, por exemplo, é recusado)."""
    partes = df[["arrival_date_year", "arrival_date_day_of_month"]]
    nao_inteiras = (partes % 1 != 0).any(axis=1)
    if nao_inteiras.any():
        raise ValueError(
            f"Ano e dia da chegada devem ser números inteiros (reserva(s) na posição {_posicoes(nao_inteiras)})"
        )
    datas = pd.to_datetime(
        pd.DataFrame({
            "year": df["arrival_date_year"].to_numpy(),
            "month": df["arrival_date_month"].map(MESES).to_numpy(),
            "day": df["arrival_date_day_of_month"].to_numpy(),
        }),
        errors="coerce",
    )
    impossiveis = pd.Series(datas.isna().to_numpy())
    if impossiveis.any():
        i = impossiveis.to_numpy().argmax()
        linha = df.iloc[i]
        raise ValueError(
            f"Data de chegada impossível na(s) posição(ões) {_posicoes(impossiveis)}: dia "
            f"{linha['arrival_date_day_of_month']:g} de {linha['arrival_date_month']} de {linha['arrival_date_year']:g}"
        )


def diagnosticar(df: pd.DataFrame, colunas_numericas: list[str] | None = None) -> dict:
    """Retrato do dado (antes ou depois do ETL): volume, ausentes, duplicatas, alvo e extremos."""
    colunas_dados = [c for c in df.columns if c not in ("lote_id", "linha_origem")]
    nulos = df[colunas_dados].isna().sum()
    diag: dict = {
        "n_linhas": int(len(df)),
        "n_colunas": int(len(colunas_dados)),
        "nulos_por_coluna": {c: int(n) for c, n in nulos.items() if n > 0},
        "duplicatas_exatas": int(df.duplicated(subset=colunas_dados).sum()),
    }
    if ALVO in df.columns:
        diag["taxa_cancelamento"] = round(float(df[ALVO].mean()), 4)
    extremos = {}
    for c in colunas_numericas or ["lead_time", "adr", "adults", "babies", "days_in_waiting_list"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            extremos[c] = {
                "min": float(s.min()), "p99": float(s.quantile(0.99)), "max": float(s.max()),
            }
    diag["extremos"] = extremos
    return diag


# --------------------------------------------------------------------------- limpeza + engenharia


def remover_duplicatas(df: pd.DataFrame, estrategia: str) -> pd.DataFrame:
    """Remove duplicatas conforme a estratégia da config (ver comentário em ``etl.duplicatas``).

    Em ``entre_lotes``, a k-ésima cópia de uma linha dentro de um lote só é descartada se a k-ésima
    cópia já existir em lote anterior. Assim, 5 cópias legítimas no lote A continuam 5, e um lote B que
    reentregue 3 delas perde essas 3.
    """
    if estrategia == "nenhuma":
        return df.copy()
    if estrategia == "todas":
        return df.drop_duplicates(subset=COLUNAS_BRUTAS, keep="first").copy()
    if estrategia != "entre_lotes":
        raise ValueError(f"Estratégia de duplicatas desconhecida: {estrategia}")
    chave = df[COLUNAS_BRUTAS].copy()
    lote = df["lote_id"] if "lote_id" in df.columns else pd.Series("unico", index=df.index)
    chave["_lote"] = lote
    chave["_copia"] = chave.groupby(COLUNAS_BRUTAS + ["_lote"], dropna=False).cumcount()
    repetida = chave.drop(columns="_lote").duplicated(keep="first")
    return df.loc[~repetida].copy()


def limpar_e_engenhar(df_raw: pd.DataFrame, cfg: dict, inferencia: bool = False) -> tuple[pd.DataFrame, dict]:
    """Aplica as regras de limpeza e cria os atributos derivados.

    Nada aqui depende de estatísticas globais do conjunto, então é seguro rodar antes da partição.
    Retorna ``(df_limpo, relatorio)``; o relatório conta cada linha removida e o motivo.

    ``inferencia=True`` é o modo de uso do modelo com reservas novas (``src.prever``): aceita entradas sem
    o alvo e sem as colunas posteriores à reserva (``completar_reserva_nova``), não remove duplicatas (cada
    linha é uma reserva a prever), mantém a ordem da entrada e devolve a saída sem a coluna alvo.
    As regras de domínio e os atributos derivados são exatamente os do treino.
    """
    if inferencia:
        df_raw = completar_reserva_nova(df_raw)
    else:
        validar_dados_brutos(df_raw)
    etl = cfg["etl"]
    rel: dict = {"n_entrada": int(len(df_raw))}

    # 1) Duplicatas. Não há ID de reserva; a estratégia (config) decide o que é "duplicado".
    rel["duplicatas_identicas_no_dado"] = int(df_raw.duplicated(subset=COLUNAS_BRUTAS).sum())  # informativo
    df = df_raw.copy() if inferencia else remover_duplicatas(df_raw, etl["duplicatas"]["estrategia"])
    rel["duplicatas_removidas"] = rel["n_entrada"] - len(df)
    rel["duplicatas_estrategia"] = "nenhuma (inferência)" if inferencia else etl["duplicatas"]["estrategia"]

    df["arrival_date"] = construir_data_chegada(df)

    # 2) Regras de domínio (cada contagem é independente; uma linha pode quebrar mais de uma)
    hospedes = df["adults"] + df["children"].fillna(0) + df["babies"]
    sem_hospedes = hospedes <= 0
    adr_negativo = df["adr"] < 0
    remover = pd.Series(False, index=df.index)
    if etl["regras_dominio"]["remover_sem_hospedes"]:
        remover |= sem_hospedes
    if etl["regras_dominio"]["remover_adr_negativo"]:
        remover |= adr_negativo
    rel["regras_dominio"] = {"sem_hospedes": int(sem_hospedes.sum()), "adr_negativo": int(adr_negativo.sum())}
    rel["regras_dominio_removidas"] = int(remover.sum())
    if inferencia:
        rel["indices_removidos_dominio"] = [int(i) for i in df.index[remover]]
    df = df.loc[~remover].copy()

    # 3) Ausentes: imputação por constante de domínio (não aprende nada dos dados)
    rel["imputados"] = {
        "children": int(df["children"].isna().sum()),
        "country": int(df["country"].isna().sum()),
        "agent": int(df["agent"].isna().sum()),
        "company": int(df["company"].isna().sum()),
    }
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("UNK")
    df["tem_agente"] = df["agent"].notna().astype(int)
    df["tem_empresa"] = df["company"].notna().astype(int)
    df["agent"] = df["agent"].map(lambda v: "SEM_AGENTE" if pd.isna(v) else str(int(v)))
    df["meal"] = df["meal"].replace({"Undefined": "SC"})  # a documentação do dataset trata ambos como "sem pacote"

    # 4) Atributos derivados
    df["total_noites"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_hospedes"] = df["adults"] + df["children"] + df["babies"]
    df["tem_criancas"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["valor_estimado_estadia"] = df["adr"] * df["total_noites"]
    df["mes_chegada"] = df["arrival_date"].dt.month
    df["dia_semana_chegada"] = df["arrival_date"].dt.day_of_week
    historico = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["razao_cancel_previo"] = np.where(historico > 0, df["previous_cancellations"] / historico.where(historico > 0, 1), 0.0)

    # 5) Colunas finais: meta + tempo + alvo + features. Tudo o mais (vazamento etc.) é descartado.
    meta = [c for c in cfg["features"]["meta"] if c in df.columns]
    features = lista_features(cfg)
    ausentes = [c for c in features if c not in df.columns]
    if ausentes:
        raise ValueError(f"Features previstas na config mas não geradas pelo ETL: {ausentes}")
    removidas = [c for c in _colunas_a_remover(cfg) if c in df.columns]
    rel["colunas_removidas"] = removidas
    alvo = [] if inferencia else [ALVO]
    df = df[meta + ["arrival_date"] + alvo + features].copy()

    for c in cfg["features"]["categoricas"]:
        df[c] = df[c].astype(str)

    # 6) Ordem cronológica estável (empate: lote e linha de origem). Na inferência, a ordem da entrada.
    if not inferencia:
        ordenar = ["arrival_date"] + [c for c in ("lote_id", "linha_origem") if c in df.columns]
        df = df.sort_values(ordenar, kind="mergesort").reset_index(drop=True)

    rel["n_saida"] = int(len(df))
    rel["removidas_total"] = rel["n_entrada"] - rel["n_saida"]
    return df, rel


# --------------------------------------------------------------------------- partição temporal


def dividir_temporal(df: pd.DataFrame, teste_fracao: float, coluna_tempo: str = "arrival_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa desenvolvimento (passado) e teste (futuro) cortando em uma data.

    O corte cai sempre na fronteira de um dia: nenhuma data aparece nos dois lados. Por isso a fração
    de teste obtida é aproximada. O teste não entra no treino nem na seleção; só na avaliação final do ciclo.
    """
    if not 0 < teste_fracao < 1:
        raise ValueError("teste_fracao deve estar entre 0 e 1")
    ordenado = df.sort_values(coluna_tempo, kind="mergesort")
    posicao = int(len(ordenado) * (1 - teste_fracao))
    corte = ordenado[coluna_tempo].iloc[posicao]
    dev = ordenado[ordenado[coluna_tempo] < corte].reset_index(drop=True)
    teste = ordenado[ordenado[coluna_tempo] >= corte].reset_index(drop=True)
    if dev.empty or teste.empty:
        raise ValueError("Partição temporal resultou em conjunto vazio; verifique os dados do ciclo")
    return dev, teste


def caminhos_particoes(cfg: dict, raiz: Path = RAIZ) -> dict[str, Path]:
    base = resolver(cfg["dados"]["dir_processed"], raiz)
    return {"dev": base / "dev.parquet", "teste": base / "teste.parquet"}


def salvar_particoes(dev: pd.DataFrame, teste: pd.DataFrame, cfg: dict, raiz: Path = RAIZ) -> dict[str, Path]:
    caminhos = caminhos_particoes(cfg, raiz)
    caminhos["dev"].parent.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(caminhos["dev"], index=False)
    teste.to_parquet(caminhos["teste"], index=False)
    return caminhos


def carregar_particao(nome: str, cfg: dict, raiz: Path = RAIZ) -> pd.DataFrame:
    """Lê ``dev`` ou ``teste``. O teste só é lido no notebook 04 (vencedor × vigente e análise de erros)."""
    if nome not in ("dev", "teste"):
        raise ValueError("nome deve ser 'dev' ou 'teste'")
    return pd.read_parquet(caminhos_particoes(cfg, raiz)[nome])


# --------------------------------------------------------------------------- pré-processador


class Winsorizador(TransformerMixin, BaseEstimator):
    """Limita cada coluna aos quantis aprendidos no ``fit`` (somente treino).

    Valores ausentes são preservados para o imputador seguinte tratar.
    """

    def __init__(self, q_inf: float = 0.001, q_sup: float = 0.999):
        self.q_inf = q_inf
        self.q_sup = q_sup

    def fit(self, X, y=None):
        valores = np.asarray(X, dtype=float)
        self.n_features_in_ = valores.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object) if hasattr(X, "columns") else None
        self.limite_inf_, self.limite_sup_ = np.nanquantile(valores, [self.q_inf, self.q_sup], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.limite_inf_, self.limite_sup_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.feature_names_in_ is not None:
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


def construir_preprocessador(cfg: dict) -> ColumnTransformer:
    """ColumnTransformer NÃO ajustado. Quem treina é responsável por ajustá-lo só com dados de treino
    (idealmente dentro de um Pipeline, para que a validação cruzada reajuste a cada fold)."""
    cols = colunas_modelagem(cfg)
    q = cfg["etl"]["outliers"]
    numerica_cauda = Pipeline([
        ("winsor", Winsorizador(q_inf=q["q_inf"], q_sup=q["q_sup"])),
        ("imputador", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    numerica_simples = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    categorica = Pipeline([
        ("imputador", SimpleImputer(strategy="constant", fill_value="UNK")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=cfg["etl"]["ohe_min_frequencia"],
            sparse_output=False,
        )),
    ])
    pre = ColumnTransformer(
        [
            ("num_cauda", numerica_cauda, cols["numericas_cauda_longa"]),
            ("num_simples", numerica_simples, cols["numericas_simples"]),
            ("cat", categorica, cols["categoricas"]),
            ("bin", "passthrough", cols["binarias"]),
        ],
        verbose_feature_names_out=False,
    )
    pre.set_output(transform="pandas")
    return pre
