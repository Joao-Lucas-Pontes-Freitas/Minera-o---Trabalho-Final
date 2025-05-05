# Trabalho Final - Mneração

# DBSCAN vs HDBSCAN

Comparativo de **DBSCAN** e **HDBSCAN** para detecção de anomalias no dataset de Fraude de Cartão de Crédito.

---

## Sumário rápido

1. Instalar dependências
2. Garantir que `dataset.csv` esteja na pasta
3. Executar `python anomalias.py`
4. Analisar métricas e gráficos

---

## 1 – Requisitos

| Pacote       | Versão mínima |
| ------------ | ------------- |
| Python       | 3.9 +         |
| scikit‑learn | 1.4 +         |

Dependências Python:

```bash
pip install pandas numpy matplotlib scikit-learn tqdm
```

---

## 2 – Colocar o dataset

Copie ou mova seu arquivo **dataset.csv** (exportado do CATS) para a pasta do projeto:

```
./
├─ anomalias.py
├─ creditcard_modified.csv        
└─ README.md
```

O CSV deve conter pelo menos:

* Coluna **`y`** (1 = anomalia, 0 = normal)
* Demais colunas numéricas que serão usadas como features

---

## 3 – Executar

```bash
python anomalias.py                  # usa dataset.csv padrão
```

### Parâmetros opcionais

```bash
python anomalias.py --arquivo outro.csv --eps 0.3 --min_samples 10 --min_cluster_size 25
```

* `--arquivo` → CSV customizado
* `--eps` & `--min_samples` → DBSCAN
* `--min_cluster_size` → HDBSCAN

---

## 4 – Saída esperada

1. Impressão na tela de:

   * **Matriz de Confusão** (2×2)
   * **Acurácia**
   * **Coeficiente de Silhueta**

---
