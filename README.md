# Desafio Técnico - Cientista de Dados Pleno - Squad WhatsApp

**Prefeitura do Rio de Janeiro**

---

## Solução

Este repositório contém minha solução para o desafio de criar a **Inteligência de Escolha** para disparos WhatsApp — identificar quais fontes de dados são mais confiáveis ("quentes") para garantir que mensagens críticas cheguem ao cidadão com eficiência e menor custo.

---

## Abordagem

### Premissas

1. **"Quente" = alta taxa de entrega**: definido como DELIVERED + READ sobre total de tentativas.
2. **Viés de seleção é real**: sistemas mais usados historicamente acumulam mais registros sem necessariamente serem superiores — a análise bruta seria enganosa.
3. **Dado envelhece**: telefones com registros desatualizados têm menor probabilidade de entrega; modelamos isso com decaimento exponencial.
4. **WhatsApp é mobile-first**: telefones fixos têm baixíssima chance de sucesso e recebem penalidade no score.
5. **Múltiplos proprietários = dado ambíguo**: se um número está associado a mais de um CPF, pode ter sido reutilizado — penalizamos levemente.

### Estrutura da Solução

```
📁 notebooks/
├── 01_eda_carregamento.ipynb        → Carregamento, inspeção, EDA inicial
├── 02_qualidade_por_sistema.ipynb   → Taxas por sistema + decaimento temporal
├── 03_ranking_e_algoritmo.ipynb     → Ranking + algoritmo de escolha + backtest
└── 04_experimento_ab.ipynb          → Desenho do experimento A/B
```

---

## Parte 1 — Análise Exploratória e Qualidade de Fontes

### 1.1 Correlação Sistema × Performance

**Pipeline analítico**:
1. Join `base_disparo` ⟕ `dim_telefone` via `contato_telefone = telefone_mascarado`
2. Explode do array `telefone_aparicoes` → uma linha por (disparo × sistema)
3. Agregação por sistema: taxa de entrega bruta e corrigida

**Tratamento do viés de seleção**:
- O scatter volume × taxa revela se sistemas preferidos são genuinamente melhores ou apenas mais usados
- Correção via **Wilson Score Lower Bound** (IC 95%): penaliza sistemas com pouca evidência

$$\text{score}_{j} = \text{Wilson LB}(\hat{p}_j, n_j) = \frac{\hat{p}_j + \frac{z^2}{2n_j} - z\sqrt{\frac{\hat{p}_j(1-\hat{p}_j)}{n_j} + \frac{z^2}{4n_j^2}}}{1 + \frac{z^2}{n_j}}$$

### 1.2 Janela de Atualidade (Decaimento Temporal)

- Calculado: `dias_desde_atualizacao = data_disparo − registro_data_atualizacao`
- Análise por faixas: <30d, 30–90d, 90–180d, 180–365d, 1–2 anos, >2 anos
- Modelo ajustado: $P(\text{entrega} \mid t) = P_0 \cdot e^{-\lambda t}$
- **Meia-vida** do dado: estimada a partir dos dados históricos
- Heatmap sistema × faixa temporal revela quais sistemas são mais resilientes ao envelhecimento

---

## Parte 2 — Inteligência de Priorização

### 2.1 Ranking de Sistemas

O ranking usa o **Wilson Score Lower Bound** como score final. Por que sistema X > sistema Y:

> Sistema X tem score maior que Y se e somente se, com 95% de confiança, a taxa de entrega real de X é superior à de Y. Isso pode ser por taxa genuinamente maior, volume muito maior (IC mais estreito → LB mais alto), ou ambos.

### 2.2 Algoritmo de Escolha

Para N telefones de um mesmo CPF, o score de cada telefone é:

$$\text{score}_{\text{tel}} = \max_{j \in \text{sistemas}(i)}\left(\text{score}_j \times e^{-\lambda \cdot t_{ij}}\right) \times \text{bonus}_{\text{tipo}} \times \text{bonus}_{\text{qualidade}} \times \text{penalidade}_{\text{proprietários}}$$

| Componente | Valor |
|---|---|
| `score_j` | Wilson LB do sistema j |
| `decay(t)` | `exp(−λ × dias_desde_atualizacao)` |
| `bonus_tipo` | Celular: ×1.10 · Fixo: ×0.50 |
| `bonus_qualidade` | ALTA: ×1.05 · MEDIA: ×1.00 · BAIXA: ×0.90 |
| `penalidade_proprietários` | N > 1: ×0.85 |

**Seleção dos 2 melhores**:
1. Escolher o telefone com maior score
2. Para o 2º: se o próximo na fila for do mesmo sistema e Δscore < 10%, preferir um de sistema diferente (diversidade — evita ponto único de falha)

---

## Parte 3 — Experimento A/B

| Parâmetro | Valor |
|---|---|
| H₀ | `p_tratamento ≤ p_controle` |
| H₁ | `p_tratamento > p_controle + 2pp` |
| α (erro tipo I) | 0.05 |
| Poder | 0.80 |
| MDE | 2 pp absolutos |
| Unidade de randomização | CPF |
| Estratificação | `categoria_hsm` |
| Métrica primária | Taxa DELIVERED+READ |
| Métricas secundárias | Taxa READ, custo por entrega, tempo até entrega |
| Guardrail | Taxa de FAILED técnico não aumenta |

O tamanho de amostra é calculado com base no volume real de disparos históricos, e o notebook fornece a curva de poder × MDE e o cronograma estimado do experimento.

---

## Como Reproduzir

### Requisitos

```bash
pip install -r requirements.txt
```

### Execução

Os dados são lidos diretamente do bucket público GCS — nenhum download prévio é necessário:

```python
import gcsfs
fs = gcsfs.GCSFileSystem(token='anon')
```

Execute os notebooks em ordem:

```bash
jupyter notebook notebooks/01_eda_carregamento.ipynb
jupyter notebook notebooks/02_qualidade_por_sistema.ipynb
jupyter notebook notebooks/03_ranking_e_algoritmo.ipynb
jupyter notebook notebooks/04_experimento_ab.ipynb
```

O NB02 exporta artefatos para `data/` (ignorado pelo git) que são consumidos pelo NB03.

### Dependências Principais

| Pacote | Uso |
|---|---|
| `pandas` + `pyarrow` | Leitura de Parquet e manipulação de dados |
| `gcsfs` | Acesso ao bucket GCS sem download |
| `scipy` | Ajuste do modelo de decaimento exponencial |
| `matplotlib` + `seaborn` | Visualizações |
| `statsmodels` | Disponível para análises adicionais |

---

## Contato do Desafio

Dúvidas: `patricia.catandi@prefeitura.rio` com título `[CASE DS]`

Entrega: `selecao.pcrj@gmail.com`
