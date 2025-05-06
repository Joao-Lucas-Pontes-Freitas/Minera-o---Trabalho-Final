# Trabalho Final – Mineração de Dados

## Comparativo DBSCAN vs HDBSCAN na Detecção de Fraudes

Este projeto realiza a detecção de anomalias (fraudes) em transações financeiras, comparando os algoritmos **DBSCAN** e **HDBSCAN** sobre o dataset de Fraude de Cartão de Crédito.

---

## Sumário

- **Requisitos do Sistema**
- **Instalação e Configuração**
- **Uso e Execução**
- **Parâmetros Opcionais**
- **Saída Esperada**
- **Estrutura do Projeto**
- **Dependências**
- **Autores e Licença**

---

## Requisitos do Sistema

- Ubuntu 20.04+
- Python 3.9+

---

## Instalação e Configuração

1. **Instale o UV:**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   > Este comando baixa e executa um script de instalação diretamente do site [oficial](https://astral.sh/blog/uv).

2. **Verifique a instalação:**

   ```bash
   which uv
   ```

3. **Atualize as dependências:**

   ```bash
   uv sync
   ```

   > Informações estão no `pyproject.toml`.

---

## Uso e Execução

Execute o código diretamente via `uv`:

```bash
uv run src/main.py
```

---

## Parâmetros Opcionais

Você pode ajustar os algoritmos usando flags:

```bash
uv run src/main.py \
  --arquivo datasets/outro.csv \
  --eps 0.3 \
  --min_samples 10 \
  --min_cluster_size 25
```

- `--arquivo` → caminho para um CSV alternativo (padrão: `datasets/creditcardfraud_modified.csv`)
- `--eps`, `--min_samples` → parâmetros do **DBSCAN**
- `--min_cluster_size` → parâmetro do **HDBSCAN**

---

## Dependências

Listadas em `requirements.txt`:

---

## Autores

Trabalho final da disciplina de **Mineração de Dados**
Instituição: Universidade Federal de Uberlândia - UFU
Ano: 2025
Alunos: Enzo Lazzarini, João Lucas, João Pedro, Thiago Pacheco e Wallace Geraldo
