"""
Módulo de scoring de telefones por CPF.

Expõe as funções usadas no Notebook 03 para calcular o score de cada telefone
e selecionar os N melhores para um CPF. Separado aqui para facilitar importação
em produção sem depender do notebook.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson_lb(s: int, n: int, z: float = 1.96) -> float:
    """Limite inferior do intervalo de Wilson para proporção s/n."""
    if n == 0:
        return 0.0
    p = s / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (c - m) / d


def decay_factor(dias, lam: float) -> float:
    """Fator de decaimento exponencial exp(-lambda * dias).

    Retorna 0.5 (penalidade conservadora) quando a data não está disponível.
    """
    if dias is None or (isinstance(dias, float) and np.isnan(dias)) or dias < 0:
        return 0.5
    return float(np.exp(-lam * dias))


# ---------------------------------------------------------------------------
# Score por telefone
# ---------------------------------------------------------------------------

def score_telefone(
    row_tel: dict,
    aparicoes: list,
    sistema_scores: dict,
    lam: float,
    data_referencia=None,
) -> dict:
    """Calcula o score de um telefone.

    Parameters
    ----------
    row_tel:
        Campos da dim_telefone para o número (tipo, qualidade, proprietários).
    aparicoes:
        Lista de dicts com ``id_sistema`` e ``registro_data_atualizacao``.
    sistema_scores:
        Mapa {str(id_sistema_hash) -> score} gerado pelo ranking de sistemas.
    lam:
        Taxa de decaimento lambda ajustada no NB02.
    data_referencia:
        Timestamp de referência para calcular dias_desde_atualizacao.
        Usa o momento atual se não fornecido.

    Returns
    -------
    dict com score final e componentes intermediários.
    """
    if data_referencia is None:
        data_referencia = pd.Timestamp.now()

    melhor_score_base = 0.0
    melhor_sistema = None

    for ap in aparicoes:
        sistema = ap.get("id_sistema")
        s_score = sistema_scores.get(str(sistema), 0.0) if sistema is not None else 0.0

        data_upd = ap.get("registro_data_atualizacao")
        if data_upd:
            try:
                dias = (data_referencia - pd.to_datetime(data_upd)).days
            except Exception:
                dias = None
        else:
            dias = None

        score_ap = s_score * decay_factor(dias, lam)

        if score_ap > melhor_score_base:
            melhor_score_base = score_ap
            melhor_sistema = sistema

    # Bônus por tipo (WhatsApp exige celular)
    tipo = str(row_tel.get("telefone_tipo", "")).lower()
    if "cel" in tipo or "mobile" in tipo:
        bonus_tipo = 1.10
    elif "fixo" in tipo or "fix" in tipo:
        bonus_tipo = 0.50
    else:
        bonus_tipo = 1.00

    # Bônus por qualidade interna
    qualidade = str(row_tel.get("telefone_qualidade", "")).upper()
    bonus_qual = {"VALIDO": 1.05, "SUSPEITO": 1.00, "INVALIDO": 0.90}.get(qualidade, 1.00)

    # Penalidade por múltiplos proprietários
    n_prop = row_tel.get("telefone_proprietarios_quantidade", 1)
    penalidade_prop = 0.85 if (n_prop and n_prop > 1) else 1.00

    score_final = melhor_score_base * bonus_tipo * bonus_qual * penalidade_prop

    return {
        "score": score_final,
        "melhor_sistema": melhor_sistema,
        "score_base": melhor_score_base,
        "bonus_tipo": bonus_tipo,
        "bonus_qualidade": bonus_qual,
        "penalidade_proprietarios": penalidade_prop,
    }


# ---------------------------------------------------------------------------
# Seleção dos N melhores por CPF
# ---------------------------------------------------------------------------

def selecionar_melhores_telefones(
    cpf_tel_df: pd.DataFrame,
    sistema_scores: dict,
    lam: float,
    data_referencia=None,
    n_escolhas: int = 2,
    diversidade_threshold: float = 0.10,
) -> pd.DataFrame:
    """Seleciona os N melhores telefones de um CPF.

    Parameters
    ----------
    cpf_tel_df:
        DataFrame com uma linha por telefone do CPF. Deve conter as colunas
        da dim_telefone (telefone_aparicoes, telefone_tipo, telefone_qualidade,
        telefone_proprietarios_quantidade).
    sistema_scores:
        Mapa {str(id_sistema_hash) -> score}.
    lam:
        Lambda do modelo de decaimento.
    data_referencia:
        Timestamp de referência. Padrão: agora.
    n_escolhas:
        Quantos telefones retornar (padrão 2).
    diversidade_threshold:
        Se o 2º candidato for do mesmo sistema que o 1º com diferença relativa
        de score < threshold, prefere o próximo de sistema diferente — evita
        ponto único de falha sistêmico.

    Returns
    -------
    DataFrame com os N telefones selecionados, scores e componentes.
    """
    if data_referencia is None:
        data_referencia = pd.Timestamp.now()

    resultados = []
    for _, row in cpf_tel_df.iterrows():
        aparicoes = row.get("telefone_aparicoes")
        if aparicoes is None or (isinstance(aparicoes, float) and np.isnan(aparicoes)):
            aparicoes = []
        elif not isinstance(aparicoes, list):
            aparicoes = list(aparicoes)

        info = score_telefone(row.to_dict(), aparicoes, sistema_scores, lam, data_referencia)
        info["telefone_id"] = row.get("telefone_numero", row.get("telefone_mascarado", ""))
        info["telefone_tipo"] = row.get("telefone_tipo", "")
        info["telefone_qualidade"] = row.get("telefone_qualidade", "")
        resultados.append(info)

    df_res = (
        pd.DataFrame(resultados)
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    if len(df_res) <= n_escolhas:
        return df_res

    escolhidos = [df_res.iloc[0]]
    sistema_1 = df_res.iloc[0]["melhor_sistema"]
    score_1 = df_res.iloc[0]["score"]

    for i in range(1, len(df_res)):
        if len(escolhidos) >= n_escolhas:
            break
        candidato = df_res.iloc[i]
        diff_rel = (score_1 - candidato["score"]) / (score_1 + 1e-9)
        if (
            candidato["melhor_sistema"] == sistema_1
            and diff_rel < diversidade_threshold
            and i < len(df_res) - 1
        ):
            continue
        escolhidos.append(candidato)

    if len(escolhidos) < n_escolhas:
        escolhidos.append(df_res.iloc[1])

    return pd.DataFrame(escolhidos).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Carregamento de artefatos
# ---------------------------------------------------------------------------

def carregar_artefatos(data_dir: str | Path = "../data"):
    """Carrega os artefatos gerados pelo NB02.

    Returns
    -------
    tuple: (sistema_scores dict, lambda float, alias_map dict)
    """
    data_dir = Path(data_dir)

    ranking = pd.read_csv(data_dir / "ranking_sistemas.csv")
    sistema_scores = dict(
        zip(ranking["id_sistema_hash"].astype(str), ranking["score_sistema"])
    )

    with open(data_dir / "decay_params.json") as f:
        params = json.load(f)
    lam = params["lambda"]

    with open(data_dir / "alias_map.json") as f:
        alias_map = json.load(f)

    return sistema_scores, lam, alias_map
