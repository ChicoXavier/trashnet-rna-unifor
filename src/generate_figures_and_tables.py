#!/usr/bin/env python3
"""
Etapa 3: Geração de Todas as Figuras e Tabelas Obrigatórias (Seção 4 e Seção 7).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dataset-resized"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_all_results():
    with open(RESULTS_DIR / "experiments_results.json", "r", encoding="utf-8") as f:
        exp_data = json.load(f)
    with open(RESULTS_DIR / "yolo_bonus_results.json", "r", encoding="utf-8") as f:
        yolo_data = json.load(f)
    return exp_data, yolo_data


def build_comparative_tables(exp_data, yolo_data):
    rows_comp = []
    rows_per_cls = []

    all_exps = exp_data["experiments"] + [yolo_data]
    for e in all_exps:
        is_yolo = e["exp_id"].startswith("exp7")
        t = e["test_metrics"]
        if not is_yolo:
            v_f1 = e["val_metrics"]["f1_macro"]
            v_acc = e["val_metrics"]["accuracy"]
            ep_str = f"{e['best_epoch']}/{e['epochs_executed']}"
            model_name = e["model_type"]
            weights_str = "Sim" if e["use_class_weights"] else "Não"
        else:
            yolo_csv = pd.read_csv(BASE_DIR / e["history_csv"])
            yolo_csv.columns = [c.strip() for c in yolo_csv.columns]
            v_acc = round(float(yolo_csv["metrics/accuracy_top1"].max()), 4)
            v_f1 = None
            ep_str = f"{e['hyperparameters']['epochs_executed']}/{e['hyperparameters']['epochs_max']}"
            model_name = "YOLO11n-cls (Bonus)"
            weights_str = "Não (Pré-treinado)"

        rows_comp.append({
            "Experimento": e["exp_label"],
            "Modelo": model_name,
            "Resolucao_LxA": e["resolution_wh"],
            "Pesos_Classe": weights_str,
            "Parametros": e["parameters_count"],
            "Tamanho_MB": e["model_size_mb"],
            "Epocas_Melhor_Total": ep_str,
            "Tempo_Total_s": e["total_train_time_sec"],
            "Tempo_Medio_Epoca_s": e["avg_epoch_time_sec"],
            "Inferencia_ms_img": e.get("inference_ms_per_image", "-"),
            "Pico_RAM_MB": e["peak_ram_mb"],
            "Val_Acc": v_acc,
            "Val_F1_Macro": v_f1 if v_f1 is not None else "-",
            "Teste_Acc": t["accuracy"],
            "Teste_F1_Macro": t["f1_macro"],
        })

        for c in CLASSES:
            pc = t["per_class"][c]
            rows_per_cls.append({
                "Experimento": e["exp_label"],
                "Classe": c,
                "Precisao": pc["precision"],
                "Recall": pc["recall"],
                "F1": pc["f1"],
                "Suporte": pc["support"],
            })

    df_comp = pd.DataFrame(rows_comp)
    df_cls = pd.DataFrame(rows_per_cls)
    df_comp.to_csv(RESULTS_DIR / "tabela_comparativa_geral.csv", index=False)
    df_cls.to_csv(RESULTS_DIR / "tabela_metricas_por_classe.csv", index=False)
    print("Tabelas comparativas salvas em results/.")
    return df_comp, df_cls


def plot_learning_curves_6exp(exp_data):
    exps = exp_data["experiments"]
    fig, axes = plt.subplots(6, 2, figsize=(13.5, 21), sharex=False)

    for idx, e in enumerate(exps):
        hist = pd.DataFrame(e["history"])
        epochs = hist["epoch"]
        best_ep = e["best_epoch"]

        # Coluna 0: Perda (Loss)
        ax_l = axes[idx, 0]
        ax_l.plot(epochs, hist["train_loss"], label="Treino Loss", color="#1f77b4", linewidth=2.2)
        ax_l.plot(epochs, hist["val_loss"], label="Validação Loss", color="#d62728", linewidth=2.2, linestyle="--")
        ax_l.axvline(best_ep, color="green", linestyle=":", linewidth=1.8, label=f"Melhor Época ({best_ep})")
        ax_l.set_title(f"{e['exp_label']} — Perda (Loss)", fontsize=12, fontweight="bold")
        ax_l.set_ylabel("Loss", fontsize=11)
        ax_l.tick_params(labelsize=10)
        ax_l.grid(True, linestyle="--", alpha=0.45)
        ax_l.legend(fontsize=9.5, loc="upper right")
        if idx == 5:
            ax_l.set_xlabel("Época", fontsize=11)

        # Coluna 1: Acurácia
        ax_a = axes[idx, 1]
        ax_a.plot(epochs, hist["train_acc"], label="Treino Acurácia", color="#1f77b4", linewidth=2.2)
        ax_a.plot(epochs, hist["val_acc"], label="Validação Acurácia", color="#ff7f0e", linewidth=2.2, linestyle="--")
        ax_a.axvline(best_ep, color="green", linestyle=":", linewidth=1.8, label=f"Melhor Época ({best_ep})")
        ax_a.set_title(f"{e['exp_label']} — Acurácia", fontsize=12, fontweight="bold")
        ax_a.set_ylabel("Acurácia", fontsize=11)
        ax_a.set_ylim(0, 1.0)
        ax_a.tick_params(labelsize=10)
        ax_a.grid(True, linestyle="--", alpha=0.45)
        ax_a.legend(fontsize=9.5, loc="lower right")
        if idx == 5:
            ax_a.set_xlabel("Época", fontsize=11)

    fig.suptitle("Curvas de Aprendizagem (Perda e Acurácia por Época) — Seis Experimentos Obrigatórios", fontsize=14.5, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIGURES_DIR / "fig4_curvas_aprendizagem_6exp.png", dpi=200)
    plt.close(fig)
    print("fig4_curvas_aprendizagem_6exp.png gerada.")


def draw_confusion_matrix_ax(ax, cm_counts, cm_pct, title):
    cm_counts = np.array(cm_counts)
    cm_pct = np.array(cm_pct)
    annot = np.empty_like(cm_counts, dtype=object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            annot[i, j] = f"{cm_counts[i, j]}\n({cm_pct[i, j]:.1f}%)"

    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Classe Prevista", fontsize=9)
    ax.set_ylabel("Classe Real", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8.5)


def plot_confusion_matrices_6exp(exp_data):
    exps = exp_data["experiments"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 15.5))
    axes_flat = axes.flatten()

    for idx, e in enumerate(exps):
        t = e["test_metrics"]
        title = f"{e['exp_label']}\nTeste Acc={t['accuracy']:.4f} | F1 Macro={t['f1_macro']:.4f}"
        draw_confusion_matrix_ax(axes_flat[idx], t["confusion_matrix_counts"], t["confusion_matrix_row_pct"], title)

    fig.suptitle("Matrizes de Confusão no Conjunto de Teste (Contagens e % por Linha) — Seis Experimentos", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIGURES_DIR / "fig5_matrizes_confusao_6exp.png", dpi=180)
    plt.close(fig)
    print("fig5_matrizes_confusao_6exp.png gerada.")


def plot_resolution_effect(exp_data):
    # Exp 3 (128x96), Exp 5 (256x192), Exp 6 (512x384)
    cnn_res_exps = [e for e in exp_data["experiments"] if e["model_type"] == "CNN" and not e["use_class_weights"]]
    labels = [e["resolution_wh"] for e in cnn_res_exps]
    accs = [e["test_metrics"]["accuracy"] * 100.0 for e in cnn_res_exps]
    f1s = [e["test_metrics"]["f1_macro"] * 100.0 for e in cnn_res_exps]
    avg_ep_times = [e["avg_epoch_time_sec"] for e in cnn_res_exps]
    total_times = [e["total_train_time_sec"] for e in cnn_res_exps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Gráfico (a): Acurácia e F1 Macro (%)
    x = np.arange(len(labels))
    w = 0.34
    b1 = ax1.bar(x - w/2, accs, w, label="Acurácia no Teste (%)", color="#1f77b4", edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(x + w/2, f1s, w, label="F1 Macro no Teste (%)", color="#2ca02c", edgecolor="black", linewidth=0.6)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Desempenho (%)")
    ax1.set_ylim(0, 90)
    ax1.set_title("(a) Desempenho no Teste por Resolução (CNN Sem Pesos)", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.45)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Gráfico (b): Tempo médio por época e tempo total
    ax2_twin = ax2.twinx()
    b3 = ax2.bar(x - w/2, total_times, w, label="Tempo Total de Treino (s)", color="#ff7f0e", edgecolor="black", linewidth=0.6)
    b4 = ax2_twin.bar(x + w/2, avg_ep_times, w, label="Tempo Médio por Época (s)", color="#9467bd", edgecolor="black", linewidth=0.6)
    for bar in b3:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}s", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#b34d00")
    for bar in b4:
        h = bar.get_height()
        ax2_twin.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#5e3c82")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Tempo Total de Treinamento (s)", color="#b34d00")
    ax2_twin.set_ylabel("Tempo Médio por Época (s)", color="#5e3c82")
    ax2.set_ylim(0, max(total_times) * 1.18)
    ax2_twin.set_ylim(0, max(avg_ep_times) * 1.18)
    ax2.set_title("(b) Custo Temporal por Resolução (28.358 parâmetros fixos)", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.45)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5)

    fig.suptitle("Efeito da Resolução de Entrada na CNN (Sem Pesos de Classe)", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_efeito_resolucao_cnn.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("fig6_efeito_resolucao_cnn.png gerada.")


def plot_error_analysis(exp_data):
    # Selecionar o modelo obrigatório com maior F1 macro de validação
    best_exp = max(exp_data["experiments"], key=lambda e: e["val_metrics"]["f1_macro"])
    exp_id = best_exp["exp_id"]
    preds_df = pd.read_csv(RESULTS_DIR / f"predictions_{exp_id}.csv")
    errors_df = preds_df[~preds_df["correct"]].copy()

    # Selecionar até 6 erros representativos (diversificando classes reais quando possível)
    selected = errors_df.sort_values("pred_prob", ascending=False).drop_duplicates(subset=["class_name"]).head(6)
    if len(selected) < 6:
        remaining = errors_df.drop(selected.index).sort_values("pred_prob", ascending=False).head(6 - len(selected))
        selected = pd.concat([selected, remaining])

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    axes_flat = axes.flatten()

    for idx, row in enumerate(selected.itertuples(index=False)):
        ax = axes_flat[idx]
        img_path = DATA_DIR / row.relative_path
        with Image.open(img_path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{row.filename}\nReal: {row.class_name} | Previsto: {row.pred_class} ({row.pred_prob*100:.1f}%)",
            fontsize=9.5,
            fontweight="bold",
            color="#a61c1c",
        )

    fig.suptitle(
        f"Análise dos Erros — Melhor Modelo Obrigatório na Validação: {best_exp['exp_label']} (Val F1={best_exp['val_metrics']['f1_macro']:.4f})",
        fontsize=11.5,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURES_DIR / "fig7_analise_erros_melhor_modelo.png", dpi=180)
    plt.close(fig)
    print(f"fig7_analise_erros_melhor_modelo.png gerada (total de erros no teste: {len(errors_df)}/{len(preds_df)}).")


def plot_yolo_bonus_figures(yolo_data):
    yolo_csv = pd.read_csv(BASE_DIR / yolo_data["history_csv"])
    yolo_csv.columns = [c.strip() for c in yolo_csv.columns]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # 1. Curva de perda
    axes[0].plot(yolo_csv["epoch"], yolo_csv["train/loss"], label="Treino Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(yolo_csv["epoch"], yolo_csv["val/loss"], label="Validação Loss", color="#d62728", linewidth=2, linestyle="--")
    axes[0].set_title("YOLO11n-cls — Curva de Perda", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=8.5)

    # 2. Curva de acurácia top1 na validação
    axes[1].plot(yolo_csv["epoch"], yolo_csv["metrics/accuracy_top1"], label="Validação Top-1 Acc", color="#2ca02c", linewidth=2)
    axes[1].set_title("YOLO11n-cls — Acurácia na Validação", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia Top-1")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=8.5, loc="lower right")

    # 3. Matriz de confusão no teste
    t = yolo_data["test_metrics"]
    draw_confusion_matrix_ax(
        axes[2],
        t["confusion_matrix_counts"],
        t["confusion_matrix_row_pct"],
        f"YOLO11n-cls — Matriz de Confusão (Teste)\nAcc={t['accuracy']:.4f} | F1 Macro={t['f1_macro']:.4f}",
    )

    fig.suptitle("Objetivo Extra (Seção 7): Avaliação do YOLO11n-cls (224×224, Transfer Learning)", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_yolo_bonus_avaliacao.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("fig8_yolo_bonus_avaliacao.png gerada.")


def main():
    exp_data, yolo_data = load_all_results()
    build_comparative_tables(exp_data, yolo_data)
    plot_learning_curves_6exp(exp_data)
    plot_confusion_matrices_6exp(exp_data)
    plot_resolution_effect(exp_data)
    plot_error_analysis(exp_data)
    plot_yolo_bonus_figures(yolo_data)
    print("\nTodas as figuras e tabelas foram geradas com sucesso!")


if __name__ == "__main__":
    main()
