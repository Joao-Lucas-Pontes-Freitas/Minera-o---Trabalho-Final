from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, HDBSCAN


# ============================================================
# 1) PRÉ-PROCESSAMENTO
# ============================================================


def carregar_csv(caminho: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(caminho)
    df = df.sample(frac=1 / 1, random_state=42).reset_index(drop=True)
    if "y" not in df.columns:
        raise ValueError("Coluna 'y' não encontrada")
    y = df["y"].astype(int).to_numpy()
    X = StandardScaler().fit_transform(df.drop(columns=["y"]))
    return X, y


def sugerir_eps_kdistance(X: np.ndarray, k: int = 5) -> float:
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist = nn.kneighbors(X)[0][:, -1]
    d_k = np.sort(dist)
    return float(np.median(d_k) + d_k.std())


def plot_kdistance_curve(X: np.ndarray, k: int = 5, n_max: int = 10_000) -> None:
    """
    Sempre exibe a curva k-distance em uma sub-amostra (≤ n_max) e
    destaca a linha 'mediana + σ' — valor usado como eps.
    """
    idx = np.random.choice(len(X), size=min(n_max, len(X)), replace=False)
    Xsub = X[idx]

    nn = NearestNeighbors(n_neighbors=k).fit(Xsub)
    dist = nn.kneighbors(Xsub)[0][:, -1]
    d_k = np.sort(dist)
    cutoff = np.median(d_k) + d_k.std()

    plt.figure(figsize=(5, 3))
    plt.plot(d_k, label=f"{k}-distance (ordenada)")
    plt.axhline(cutoff, ls="--", color="red", label=f"mediana + σ = {cutoff:.4f}")
    plt.xlabel("Pontos ordenados")
    plt.ylabel(f"{k}ª distância")
    plt.title(f"Curva {k}-distance (subamostra {len(Xsub)}/{len(X)})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 2) MINERAÇÃO DE DADOS
# ============================================================


def _metricas_binarias(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    return (
        confusion_matrix(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0),
        matthews_corrcoef(y_true, y_pred),
        average_precision_score(y_true, y_score),
    )


def avaliar_dbscan(X, y_true, eps, min_samples):
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X)
    scores = (labels == -1).astype(float)
    cm, p, r, f1, mcc, ap = _metricas_binarias(y_true, scores)
    return {
        "nome": f"DBSCAN (eps={eps:.3f})",
        "cm": cm,
        "prec": p,
        "rec": r,
        "f1": f1,
        "mcc": mcc,
        "auprc": ap,
        "scores": scores,
    }


def _scores_hdb(modelo: HDBSCAN) -> np.ndarray:
    return (
        1.0 - modelo.probabilities_
        if not hasattr(modelo, "outlier_scores_")
        else (modelo.outlier_scores_ - modelo.outlier_scores_.min())
        / (modelo.outlier_scores_.ptp() + 1e-12)
    )


def avaliar_hdbscan(X, y_true, min_cluster_size, q_out):
    modelo = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=-1).fit(X)
    scores = _scores_hdb(modelo)
    thr = np.quantile(scores, q_out)
    cm, p, r, f1, mcc, ap = _metricas_binarias(y_true, scores, thr)
    return {
        "nome": f"HDBSCAN (min_sz={min_cluster_size})",
        "cm": cm,
        "prec": p,
        "rec": r,
        "f1": f1,
        "mcc": mcc,
        "auprc": ap,
        "scores": scores,
    }


# ============================================================
# 3) PÓS-PROCESSAMENTO
# ============================================================


def plot_matriz(cm, titulo, ax=None):
    """
    Plota matriz de confusão em ax. Se ax for None, cria um novo.
    """
    # 1) Cria axes se não foi passado
    if ax is None:
        fig, ax = plt.subplots()
    # 2) Plota no ax fornecido
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(titulo)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Legítimo", "Fraude"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Legítimo", "Fraude"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    # 3) Adiciona colorbar ao redor daquele ax
    ax.figure.colorbar(im, ax=ax, fraction=0.045)
    return ax


def plot_pr(resultados, y_true, ax=None):
    """
    Plota curvas Precision-Recall em ax. Se ax for None, cria um novo.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    for r in resultados:
        p, rc, _ = precision_recall_curve(y_true, r["scores"])
        ax.plot(rc, p, lw=2, label=f"{r['nome']} (AP={r['auprc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend()
    ax.grid(True)
    return ax


def plot_barras(resultados, ax=None):
    """
    Plota barras de F1 x AUPRC em ax. Se ax for None, cria um novo.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    nomes = [r["nome"] for r in resultados]
    f1s = [r["f1"] for r in resultados]
    aps = [r["auprc"] for r in resultados]
    x = np.arange(len(nomes))
    w = 0.35
    ax.bar(x - w / 2, f1s, w, label="F1 (fraude)")
    ax.bar(x + w / 2, aps, w, label="AUPRC")
    ax.set_xticks(x)
    ax.set_xticklabels(nomes, rotation=15, ha="right")
    ax.set_ylabel("Pontuação")
    ax.set_title("Comparação de Métricas")
    ax.legend()
    return ax


# ---------------------------  MAIN  ---------------------------


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fraude - DBSCAN vs HDBSCAN")
    parser.add_argument(
        "--arquivo", type=Path, default="./datasets/creditcardfraud_modified.csv"
    )
    parser.add_argument("--pct_outlier", type=float, default=0.95)
    args = parser.parse_args(argv)

    # ── Pré-processamento ─────────────────────────────────────
    X, y_true = carregar_csv(args.arquivo)

    # Curva k-distance + eps
    eps = sugerir_eps_kdistance(X)
    print(f"[info] eps escolhido (mediana + σ): {eps:.4f}")
    #plot_kdistance_curve(X, k=5)

    # ── Mineração de Dados ───────────────────────────────────
    db = avaliar_dbscan(X, y_true, eps, min_samples=X.shape[1] * 2)
    hdb = avaliar_hdbscan(X, y_true, int(np.sqrt(len(X))), args.pct_outlier)
    resultados = [db, hdb]

    # ── Pós-processamento ────────────────────────────────────
    for r in resultados:
        print(f"\n===> {r['nome']}")
        print(
            pd.DataFrame(
                r["cm"], index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]
            )
        )
        print(
            f"Precision: {r['prec']:.4f}  Recall: {r['rec']:.4f}  "
            f"F1: {r['f1']:.4f}  MCC: {r['mcc']:.4f}  AUPRC: {r['auprc']:.4f}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    plot_matriz(db["cm"], f"DBSCAN ({db['nome']})", ax=axes[0, 0])
    plot_matriz(hdb["cm"], f"HDBSCAN ({hdb['nome']})", ax=axes[0, 1])
    plot_pr(resultados, y_true, ax=axes[1, 0])
    plot_barras(resultados, ax=axes[1, 1])

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()