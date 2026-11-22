#!/usr/bin/env python3
"""
Etapa 1: Análise Exploratória de Dados (EDA), Verificação de Integridade,
Divisão Estratificada Fixa (70% Treino / 15% Validação / 15% Teste, seed=42)
e Geração de Figuras de Distribuição e Exemplos.
"""

import os
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Ordem canônica exigida pela proposta da atividade (Seção 4/5):
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
CLASS_LABELS_PT = {
    "glass": "glass (vidro)",
    "paper": "paper (papel)",
    "cardboard": "cardboard (papelão)",
    "plastic": "plastic (plástico)",
    "metal": "metal (metal)",
    "trash": "trash (outros resíduos)"
}
SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dataset-resized"
SPLITS_DIR = BASE_DIR / "data" / "splits"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_eda_and_split():
    records = []
    corrupted = []
    hashes = {}
    duplicates = []

    for label_idx, cls_name in enumerate(CLASSES):
        cls_folder = DATA_DIR / cls_name
        if not cls_folder.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {cls_folder}")

        files = sorted([f for f in cls_folder.iterdir() if f.is_file() and not f.name.startswith(".")])
        for fpath in files:
            try:
                with Image.open(fpath) as img:
                    img.verify()
                with Image.open(fpath) as img:
                    width, height = img.size
                    mode = img.mode
                    arr = np.asarray(img, dtype=np.float32)
                    mean_rgb = arr.mean(axis=(0, 1)).tolist() if arr.ndim == 3 else [float(arr.mean())] * 3
                    std_rgb = arr.std(axis=(0, 1)).tolist() if arr.ndim == 3 else [float(arr.std())] * 3

                file_hash = hashlib.md5(fpath.read_bytes()).hexdigest()
                if file_hash in hashes:
                    duplicates.append((str(fpath.relative_to(DATA_DIR)), hashes[file_hash]))
                else:
                    hashes[file_hash] = str(fpath.relative_to(DATA_DIR))

                records.append({
                    "relative_path": str(fpath.relative_to(DATA_DIR)),
                    "filename": fpath.name,
                    "class_name": cls_name,
                    "class_idx": label_idx,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "file_size_kb": round(fpath.stat().st_size / 1024.0, 2),
                    "md5": file_hash,
                    "r_mean": round(mean_rgb[0], 2),
                    "g_mean": round(mean_rgb[1], 2),
                    "b_mean": round(mean_rgb[2], 2),
                    "r_std": round(std_rgb[0], 2),
                    "g_std": round(std_rgb[1], 2),
                    "b_std": round(std_rgb[2], 2),
                })
            except Exception as e:
                corrupted.append({"path": str(fpath), "error": str(e)})

    df = pd.DataFrame(records)
    original_count = len(df)
    print(f"Total de imagens válidas inspecionadas: {original_count}")
    print(f"Arquivos corrompidos: {len(corrupted)}")
    print(f"Pares de imagens idênticas (MD5) com rótulos distintos: {len(duplicates)}")

    # B2: Remover todos os arquivos pertencentes aos pares duplicados MD5 com rótulos conflitantes
    # antes da divisão estratificada para eliminar ruído de rotulagem e vazamento entre splits.
    conflicting_paths = set()
    for p1, p2 in duplicates:
        conflicting_paths.add(p1)
        conflicting_paths.add(p2)

    removed_duplicates = sorted(conflicting_paths)
    df = df[~df["relative_path"].isin(conflicting_paths)].reset_index(drop=True)
    print(f"Arquivos removidos por duplicidade MD5 conflitante: {len(removed_duplicates)} -> {removed_duplicates}")
    print(f"Total limpo de imagens utilizado na divisão estratificada: {len(df)}")

    # Divisão estratificada: 70% treino, 15% validação, 15% teste (seed = 42)
    # Primeiro separa 70% treino e 30% temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["class_idx"]
    )
    # Divide os 30% exatamente ao meio (15% val, 15% test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["class_idx"]
    )

    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    # Salvar CSV completo da divisão estratificada e arquivos por split
    df.to_csv(SPLITS_DIR / "dataset_split_seed42.csv", index=False)
    for sp in ["train", "val", "test"]:
        df[df["split"] == sp].to_csv(SPLITS_DIR / f"{sp}_seed42.csv", index=False)

    # Calcular estatísticas de normalização (média e desvio padrão) APENAS no treino
    train_subset = df[df["split"] == "train"]
    train_rgb_mean = [
        round(float(train_subset["r_mean"].mean() / 255.0), 4),
        round(float(train_subset["g_mean"].mean() / 255.0), 4),
        round(float(train_subset["b_mean"].mean() / 255.0), 4),
    ]
    train_rgb_std = [
        round(float(train_subset["r_std"].mean() / 255.0), 4),
        round(float(train_subset["g_std"].mean() / 255.0), 4),
        round(float(train_subset["b_std"].mean() / 255.0), 4),
    ]

    # Pesos de classe inversamente proporcionais à frequência (calculados SOMENTE no treino)
    # w_c = N_train / (n_classes * N_{train, c})
    n_train = len(train_subset)
    n_classes = len(CLASSES)
    class_weights = {}
    class_counts_split = {}

    for idx, cls_name in enumerate(CLASSES):
        c_train = int((train_subset["class_name"] == cls_name).sum())
        c_val = int(((df["split"] == "val") & (df["class_name"] == cls_name)).sum())
        c_test = int(((df["split"] == "test") & (df["class_name"] == cls_name)).sum())
        c_total = int((df["class_name"] == cls_name).sum())
        w_c = round(n_train / (n_classes * c_train), 4)
        class_weights[cls_name] = {
            "class_idx": idx,
            "weight": w_c,
            "train_count": c_train
        }
        class_counts_split[cls_name] = {
            "total": c_total,
            "train": c_train,
            "val": c_val,
            "test": c_test,
            "pct_total": round(100.0 * c_total / len(df), 2),
            "class_weight": w_c
        }

    eda_summary = {
        "total_images_original": original_count,
        "removed_conflicting_duplicates": removed_duplicates,
        "total_images": len(df),
        "corrupted_files": len(corrupted),
        "duplicate_md5_pairs_original": duplicates,
        "dimensions_summary": df.groupby(["width", "height"]).size().to_dict() if False else
            [{"width": int(w), "height": int(h), "count": int(c)} for (w, h), c in df.groupby(["width", "height"]).size().items()],
        "color_modes": df["mode"].value_counts().to_dict(),
        "seed": SEED,
        "split_totals": {
            "train": int((df["split"] == "train").sum()),
            "val": int((df["split"] == "val").sum()),
            "test": int((df["split"] == "test").sum()),
        },
        "class_distribution": class_counts_split,
        "class_weights_train": class_weights,
        "train_rgb_mean_0_1": train_rgb_mean,
        "train_rgb_std_0_1": train_rgb_std,
    }

    with open(RESULTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, ensure_ascii=False)

    print("Resumo da divisão estratificada salvo em:", RESULTS_DIR / "eda_summary.json")
    generate_eda_figures(df, class_counts_split)
    return df, eda_summary


def generate_eda_figures(df: pd.DataFrame, class_counts_split: dict):
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150
    })

    # 1. Gráfico de barras da distribuição de imagens por classe em treino, validação e teste
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(CLASSES))
    width = 0.26

    train_counts = [class_counts_split[c]["train"] for c in CLASSES]
    val_counts = [class_counts_split[c]["val"] for c in CLASSES]
    test_counts = [class_counts_split[c]["test"] for c in CLASSES]

    b1 = ax.bar(x - width, train_counts, width, label=f"Treino (70% - {sum(train_counts)} imgs)", color="#1f77b4", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, val_counts, width, label=f"Validação (15% - {sum(val_counts)} imgs)", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + width, test_counts, width, label=f"Teste (15% - {sum(test_counts)} imgs)", color="#2ca02c", edgecolor="black", linewidth=0.5)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

    ax.set_ylabel("Quantidade de Imagens")
    ax.set_title("Distribuição Estratificada por Classe no Dataset TrashNet (Treino 70%, Validação 15%, Teste 15% - Seed 42)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\nTotal: {class_counts_split[c]['total']}\nw={class_counts_split[c]['class_weight']:.2f}" for c in CLASSES],
        fontsize=9.5,
    )
    ax.set_ylim(0, max(train_counts) * 1.16)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_distribuicao_classes.png", dpi=200)
    plt.close(fig)

    # 2. Painel com ao menos 2 imagens de cada uma das 6 classes (2x6 grid)
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.8))
    for col_idx, cls_name in enumerate(CLASSES):
        subset = df[(df["class_name"] == cls_name) & (df["split"] == "train")].sort_values("filename")
        samples = [subset.iloc[0], subset.iloc[min(15, len(subset) - 1)]]
        for row_idx in range(2):
            sample = samples[row_idx]
            img_path = DATA_DIR / sample["relative_path"]
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                axes[row_idx, col_idx].imshow(im_rgb)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f"{cls_name}\n({sample['filename']})", fontsize=9.5, fontweight="bold")
            else:
                axes[row_idx, col_idx].set_xlabel(f"{sample['filename']}", fontsize=8.5)

    fig.suptitle("Exemplos Representativos das Seis Classes do Dataset TrashNet (Conjunto de Treinamento)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_exemplos_classes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Uma mesma imagem comparada nas três resoluções exigidas: 128x96, 256x192 e 512x384 (largura x altura)
    sample_row = df[(df["class_name"] == "metal") & (df["split"] == "train")].sort_values("filename").iloc[5]
    img_path = DATA_DIR / sample_row["relative_path"]
    resolutions = [(128, 96), (256, 192), (512, 384)]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    with Image.open(img_path) as im_orig:
        im_orig = im_orig.convert("RGB")
        for idx, (w, h) in enumerate(resolutions):
            im_res = im_orig.resize((w, h), Image.Resampling.BILINEAR)
            axes[idx].imshow(im_res)
            n_pixels = w * h * 3
            axes[idx].set_title(f"Resolução {w} × {h} (L × A)\nTensor: {h}×{w}×3 ({n_pixels:,} valores)".replace(",", "."), fontsize=10, fontweight="bold")
            axes[idx].set_xlabel(f"Amostra: {sample_row['filename']} ({sample_row['class_name']})", fontsize=9)
            axes[idx].set_xticks([0, w // 2, w - 1])
            axes[idx].set_yticks([0, h // 2, h - 1])

    fig.suptitle("Comparação Visual de Uma Mesma Imagem nas Três Resoluções Experimentais", fontsize=11.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_comparacao_resolucoes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Figuras de EDA geradas com sucesso em:", FIGURES_DIR)


if __name__ == "__main__":
    run_eda_and_split()
