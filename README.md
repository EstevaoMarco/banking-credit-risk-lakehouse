# Banking Credit Risk Lakehouse

**[English](#english)** · **[Português](#português)**

## English

### Introduction

This project builds an end-to-end **credit risk pipeline on Databricks**. Raw banking data is ingested into a **medallion lakehouse** (bronze, silver and gold) with **Delta Lake** and **Unity Catalog**, turned into a loan-level feature table, and used to train a model that estimates the probability that a loan will not be repaid. Behavioral features come only from the account's transactions **before** the loan was granted, so the model does not see the future.

### Dataset

Public data from a Czech bank, released for the **PKDD'99 Discovery Challenge** (known as the "Berka" dataset): accounts, clients, districts, loans and about 1 million transactions from 1993 to 1998.

| File | Content | Rows |
|---|---|---|
| account | accounts | 4,500 |
| client | clients | 5,369 |
| disp | client to account links | 5,369 |
| district | district demographics | 77 |
| loan | loans, with status | 682 |
| trans | transactions | 1,056,320 |

The data files are not part of this repository. To reproduce, download the dataset and upload the CSVs to the `banking_project.bronze.raw_files` Volume.

### Architecture

```
CSVs (Volume)
   │
   ▼
BRONZE   raw data in Delta, all columns as text
   │
   ▼
SILVER   typed columns, parsed dates, Czech codes translated, data quality checks
   │
   ▼
GOLD     loan_risk_features: one row per loan, 40+ features
   │
   ▼
ML       trained on closed contracts (A/B), evaluated, then used to score active contracts (C/D)
```

### Repository structure

```
notebooks/
├── 00_setup.py                       # Unity Catalog catalog and schemas
├── 01_bronze/bronze_ingestion.py     # CSV -> Delta
├── 02_silver/                        # cleaning, one notebook per table
│   ├── 01_account.py ... 06_trans.py
├── 03_gold/gold_loan_risk_features.py
└── 04_ml/
    ├── 01_data_preparation.py        # target filter and feature selection
    ├── 02_model_training.py          # pipeline and training
    ├── 03_model_evaluation.py        # metrics, cross-validation, curves
    └── 04_score_active_loans.py      # scoring of C and D contracts
```

### Key decisions

**No temporal leakage.** Behavioral features use only transactions dated before the loan, in 30, 90 and 180 day windows: transaction count, credits, debits (including cash withdrawals), cash flow, average and minimum balance, transactions with negative balance and balance volatility.

**Target.** Loan status in the dataset:

| Status | Meaning | Use |
|---|---|---|
| A | finished, paid | training, default = 0 |
| B | finished, not paid | training, default = 1 |
| C | running, up to date | scored by the model |
| D | running, in debt | scored by the model |

Only finished contracts have a final outcome, so supervised training uses A and B (234 loans, 31 defaults). Running contracts are used as an out-of-sample check.

**Model.** Logistic regression with `class_weight="balanced"`, inside a scikit-learn `Pipeline` with imputation, scaling and one-hot encoding. Because the labeled sample is small, the main metric is stratified 5-fold cross-validation.

### Results

**Cross-validation (5 folds, A and B contracts)**

| Metric | Mean | Range across folds |
|---|---|---|
| ROC-AUC | 0.75 | 0.60 to 0.92 |
| Recall | 0.50 | 0.14 to 0.83 |
| Precision | 0.25 | 0.09 to 0.36 |
| F1 | 0.32 | 0.11 to 0.50 |
| Accuracy | 0.73 | 0.65 to 0.81 |

**Check on running contracts (C and D)**

The model never saw these contracts during training. If it ranks risk correctly, contracts in debt (D) should score higher than contracts that are up to date (C).

| Status | Contracts | Mean score | Flagged as risky |
|---|---|---|---|
| C (up to date) | 403 | 0.45 | 44% |
| D (in debt) | 45 | 0.83 | 82% |

ROC-AUC using D as a proxy target is **0.83**, which confirms that the score separates both groups well.

### Limitations

- The labeled sample is small (234 contracts, 31 defaults), so metrics vary a lot across folds.
- With `class_weight="balanced"`, scores are not calibrated probabilities. This is why 44% of up-to-date contracts are above the 0.5 cutoff.
- A few clients are younger than 18 at the loan date. These cases were kept and remain open for investigation.

### Next steps

- Compare with tree-based models (Random Forest, XGBoost) and add interpretability (coefficients and SHAP)
- Calibrate probabilities and choose the cutoff based on error cost
- Track experiments and the model in MLflow
- Orchestrate the pipeline as a Databricks Job

### How to run

1. Create the `banking_project.bronze.raw_files` Volume and upload the dataset CSVs
2. Run the notebooks in order: `00_setup`, `01_bronze`, `02_silver` (01 to 06), `03_gold`
3. Run `04_ml/03_model_evaluation` and `04_ml/04_score_active_loans` (both run training through `%run`)

**Stack:** Databricks, PySpark, Delta Lake, Unity Catalog, pandas, scikit-learn, matplotlib

---

## Português

### Introdução

Este projeto constrói um **pipeline de risco de crédito de ponta a ponta no Databricks**. Os dados bancários brutos são ingeridos em um **lakehouse medallion** (bronze, silver e gold) com **Delta Lake** e **Unity Catalog**, transformados em uma tabela de features por empréstimo e usados para treinar um modelo que estima a probabilidade de um empréstimo não ser pago. As features de comportamento vêm só das transações da conta **antes** da concessão do empréstimo, então o modelo não enxerga o futuro.

### Dataset

Dados públicos de um banco tcheco, do desafio **PKDD'99 Discovery Challenge** (conhecido como dataset "Berka"): contas, clientes, distritos, empréstimos e cerca de 1 milhão de transações entre 1993 e 1998.

| Arquivo | Conteúdo | Linhas |
|---|---|---|
| account | contas | 4.500 |
| client | clientes | 5.369 |
| disp | relação entre clientes e contas | 5.369 |
| district | dados demográficos dos distritos | 77 |
| loan | empréstimos, com o status | 682 |
| trans | transações | 1.056.320 |

Os arquivos não fazem parte deste repositório. Para reproduzir, baixe o dataset e carregue os CSVs no Volume `banking_project.bronze.raw_files`.

### Arquitetura

```
CSVs (Volume)
   │
   ▼
BRONZE   dados brutos em Delta, todas as colunas como texto
   │
   ▼
SILVER   tipos corrigidos, datas convertidas, códigos em tcheco traduzidos, checagens de qualidade
   │
   ▼
GOLD     loan_risk_features: uma linha por empréstimo, com 40+ features
   │
   ▼
ML       treino em contratos encerrados (A/B), avaliação e pontuação dos contratos ativos (C/D)
```

### Estrutura do repositório

```
notebooks/
├── 00_setup.py                       # catálogo e schemas no Unity Catalog
├── 01_bronze/bronze_ingestion.py     # CSV -> Delta
├── 02_silver/                        # limpeza de cada tabela
│   ├── 01_account.py ... 06_trans.py
├── 03_gold/gold_loan_risk_features.py
└── 04_ml/
    ├── 01_data_preparation.py        # filtro do alvo e seleção de features
    ├── 02_model_training.py          # pipeline e treino
    ├── 03_model_evaluation.py        # métricas, validação cruzada, curvas
    └── 04_score_active_loans.py      # pontuação dos contratos C e D
```

### Principais decisões

**Features sem vazamento temporal.** As features de comportamento usam só transações anteriores à data do empréstimo, em janelas de 30, 90 e 180 dias: quantidade de transações, créditos, débitos (incluindo saques), fluxo de caixa, saldo médio e mínimo, transações com saldo negativo e volatilidade do saldo.

**Alvo.** O status do empréstimo no dataset é:

| Status | Significado | Uso |
|---|---|---|
| A | encerrado, pago | treino, default = 0 |
| B | encerrado, não pago | treino, default = 1 |
| C | em andamento, em dia | pontuado pelo modelo |
| D | em andamento, em atraso | pontuado pelo modelo |

Só os contratos encerrados têm desfecho final, então o treino supervisionado usa A e B (234 empréstimos, 31 inadimplentes). Os contratos em andamento são usados como validação fora da amostra.

**Modelo.** Regressão logística com `class_weight="balanced"`, dentro de um `Pipeline` do scikit-learn com imputação, padronização e one-hot encoding. Por causa da amostra pequena, a métrica principal é a validação cruzada estratificada com 5 folds.

### Resultados

**Validação cruzada (5 folds, contratos A e B)**

| Métrica | Média | Variação entre folds |
|---|---|---|
| ROC-AUC | 0,75 | 0,60 a 0,92 |
| Recall | 0,50 | 0,14 a 0,83 |
| Precision | 0,25 | 0,09 a 0,36 |
| F1 | 0,32 | 0,11 a 0,50 |
| Accuracy | 0,73 | 0,65 a 0,81 |

**Validação nos contratos em andamento (C e D)**

O modelo nunca viu esses contratos no treino. Se ele ordena o risco corretamente, os contratos em atraso (D) devem receber scores maiores que os em dia (C).

| Status | Contratos | Score médio | Classificados como risco |
|---|---|---|---|
| C (em dia) | 403 | 0,45 | 44% |
| D (em atraso) | 45 | 0,83 | 82% |

O ROC-AUC usando D como alvo aproximado é **0,83**, o que confirma que o score separa bem os dois grupos.

### Limitações

- A amostra rotulada é pequena (234 contratos, 31 inadimplentes), então as métricas variam bastante entre folds.
- Com `class_weight="balanced"`, os scores não são probabilidades calibradas. Isso explica por que 44% dos contratos em dia aparecem acima do corte de 0,5.
- Existem alguns clientes com idade menor que 18 anos na data do empréstimo. Os casos foram mantidos e ficam como ponto de investigação.

### Próximos passos

- Comparar com modelos de árvore (Random Forest, XGBoost) e analisar interpretabilidade (coeficientes e SHAP)
- Calibrar as probabilidades e escolher o corte pelo custo do erro
- Registrar experimentos e modelo no MLflow
- Orquestrar o pipeline como um Job do Databricks

### Como executar

1. Crie o Volume `banking_project.bronze.raw_files` e carregue os CSVs do dataset
2. Execute os notebooks na ordem: `00_setup`, `01_bronze`, `02_silver` (01 a 06), `03_gold`
3. Execute `04_ml/03_model_evaluation` e `04_ml/04_score_active_loans` (os dois rodam o treino via `%run`)

**Stack:** Databricks, PySpark, Delta Lake, Unity Catalog, pandas, scikit-learn, matplotlib
