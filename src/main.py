from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from sklearn.cluster import DBSCAN, HDBSCAN


def carregar_csv(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê *dataset.csv* (CATS) e devolve (X, y).

    O CSV deve conter a coluna numérica `y` (1 = anomalia, 0 = normal) e
    pelo menos uma feature numérica para detecção.
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo {caminho} não encontrado. Coloque 'dataset.csv' na pasta do projeto "
            "ou especifique outro caminho com --arquivo."
        )

    print("Carregando dados…")
    df = pd.read_csv(caminho)

    if "y" not in df.columns:
        raise ValueError("Coluna obrigatória 'y' não encontrada em dataset.csv!")

    y = df["y"].astype(int).to_numpy()

    # Remove colunas não numéricas ou não usadas
    drop_cols = {"timestamp", "y", "category"}
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Normaliza features
    X = StandardScaler().fit_transform(X_df.to_numpy())
    return X, y


def avaliar(nome: str, y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> dict:
    pred_anomalia = (y_pred == -1).astype(int)
    cm = confusion_matrix(y_true, pred_anomalia)
    acc = accuracy_score(y_true, pred_anomalia)
    try:
        sil = silhouette_score(X, y_pred)
    except ValueError:
        sil = float("nan")

    print(f"\n===> {nome}")
    print("Matriz de Confusão:\n", cm)
    print(f"Acurácia:  {acc:.4f}")
    print(f"Silhueta:  {sil:.4f}")

    return {"nome": nome, "matriz": cm, "acuracia": acc, "silhueta": sil}


def plotar_matriz(cm: np.ndarray, titulo: str) -> None:
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_title(titulo)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomalia"])
    ax.set_yticklabels(["Normal", "Anomalia"])
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    plt.tight_layout()


def plotar_comparacao(resultados: list[dict]) -> None:

    nomes = [r["nome"] for r in resultados]

    acuracias = [r["acuracia"] for r in resultados]

    silhuetas = [r["silhueta"] for r in resultados]

    x = np.arange(len(nomes))

    largura = 0.35

    fig, ax = plt.subplots()

    ax.bar(x - largura / 2, acuracias, largura, label="Acurácia")

    ax.bar(x + largura / 2, silhuetas, largura, label="Silhueta")

    ax.set_ylabel("Pontuação")

    ax.set_title("Comparação entre DBSCAN e HDBSCAN")

    ax.set_xticks(x)

    ax.set_xticklabels(nomes)

    ax.legend()

    plt.tight_layout()

    plt.show()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Comparação DBSCAN vs HDBSCAN (CATS)")
    parser.add_argument(
        "--arquivo",
        type=Path,
        default="./datasets/creditcardfraud_modified.csv",
        help="Caminho do CSV",
    )
    parser.add_argument("--eps", type=float, default=0.5, help="eps para DBSCAN")
    parser.add_argument(
        "--min_samples", type=int, default=5, help="min_samples para DBSCAN"
    )
    parser.add_argument(
        "--min_cluster_size", type=int, default=5, help="min_cluster_size para HDBSCAN"
    )
    args = parser.parse_args(argv)

    X, y_true = carregar_csv(args.arquivo)

    print("Rodando DBSCAN…")
    db_pred = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=-1).fit_predict(
        X
    )

    print("Rodando HDBSCAN…")
    hdb_pred = HDBSCAN(min_cluster_size=args.min_cluster_size, n_jobs=-1).fit_predict(X)

    resultados = [
        avaliar("DBSCAN", y_true, db_pred, X),
        avaliar("HDBSCAN", y_true, hdb_pred, X),
    ]

    plt.figure(figsize=(10, 4))
    for i, res in enumerate(resultados, start=1):
        plt.subplot(1, 2, i)
        plotar_matriz(res["matriz"], res["nome"])
    plt.show()

    plotar_comparacao(resultados)


if __name__ == "__main__":
    main()
