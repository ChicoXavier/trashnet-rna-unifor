#!/usr/bin/env python3
"""
Busca controlada de hiperparâmetros da MLP (Exp 1: 128x96, sem pesos de classe)
usando EXCLUSIVAMENTE o conjunto de validação (F1 macro de validação).
Nenhuma métrica de teste é calculada ou exibida neste script.
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from train_experiments import (
    CLASSES,
    SEED,
    SPLITS_DIR,
    RESULTS_DIR,
    set_all_seeds,
    preload_split_tensors,
    TrashNetMemoryDataset,
    TrashNetMLP,
    count_parameters,
    evaluate_model,
)


def tune_single_config(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    lr: float,
    normalize: bool,
    mean_rgb: list,
    std_rgb: list,
    max_epochs: int = 60,
    patience: int = 10,
    batch_size: int = 32,
):
    set_all_seeds(SEED)
    device = torch.device("cpu")

    train_dataset = TrashNetMemoryDataset(x_train, y_train, augment=True)
    val_dataset = TrashNetMemoryDataset(x_val, y_val, augment=False)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TrashNetMLP(
        height=96,
        width=128,
        num_classes=len(CLASSES),
        normalize=normalize,
        mean=mean_rgb,
        std=std_rgb,
    ).to(device)

    n_params = count_parameters(model)
    assert n_params == 9471110, f"Número de parâmetros treináveis inesperado: {n_params}"

    criterion = nn.CrossEntropyLoss()
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 1
    epochs_no_improve = 0
    epochs_trained = 0

    print(f"\n{'='*70}")
    print(f"Tuning MLP (Exp 1, 128x96, sem pesos) | lr={lr} | normalize={normalize} | params={n_params:,}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        val_loss, val_acc, val_f1, _, _, _ = evaluate_model(model, val_loader, val_criterion, device)
        scheduler.step(val_f1)
        epochs_trained = epoch

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            marker = "* [MELHOR]"
        else:
            epochs_no_improve += 1
            marker = ""

        if epoch == 1 or epoch % 5 == 0 or marker or epochs_no_improve >= patience:
            print(
                f"Época {epoch:02d}/{max_epochs:02d} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} {marker}"
            )

        if epochs_no_improve >= patience:
            print(f"Early Stopping na época {epoch} (melhor época: {best_epoch}, Val F1={best_val_f1:.4f}).")
            break

    elapsed = time.perf_counter() - t0
    print(
        f"--> Config lr={lr}, normalize={normalize} concluída em {elapsed:.1f}s | "
        f"Melhor época: {best_epoch}/{epochs_trained} | Val Acc: {best_val_acc:.4f} | Val F1: {best_val_f1:.4f}"
    )

    return {
        "lr": lr,
        "normalize": normalize,
        "best_epoch": best_epoch,
        "epochs_trained": epochs_trained,
        "val_acc": round(float(best_val_acc), 4),
        "val_f1_macro": round(float(best_val_f1), 4),
    }


def main():
    df_train = pd.read_csv(SPLITS_DIR / "train_seed42.csv")
    df_val = pd.read_csv(SPLITS_DIR / "val_seed42.csv")

    with open(RESULTS_DIR / "eda_summary.json", "r", encoding="utf-8") as f:
        eda = json.load(f)

    mean_rgb = eda["train_rgb_mean_0_1"]
    std_rgb = eda["train_rgb_std_0_1"]
    print(f"Estatísticas RGB do treino: mean={mean_rgb}, std={std_rgb}")

    print("Pré-carregando tensores 128x96 (treino e validação apenas)...")
    x_train, y_train = preload_split_tensors(df_train, 128, 96)
    x_val, y_val = preload_split_tensors(df_val, 128, 96)

    lrs = [1e-4, 3e-4, 1e-3]
    norms = [False, True]

    rows = []
    for lr in lrs:
        for norm in norms:
            row = tune_single_config(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                lr=lr,
                normalize=norm,
                mean_rgb=mean_rgb,
                std_rgb=std_rgb,
                max_epochs=60,
                patience=10,
                batch_size=32,
            )
            rows.append(row)

    df_res = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "mlp_tuning.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nTabela salva em: {out_csv}")
    print(df_res.to_string(index=False))

    # Seleção pela regra: maior Val F1 Macro; empate (<0.005) -> menor lr e normalize=False
    max_f1 = df_res["val_f1_macro"].max()
    tied = df_res[df_res["val_f1_macro"] >= max_f1 - 0.005].sort_values(
        by=["normalize", "lr"], ascending=[True, True]
    )
    chosen = tied.iloc[0]
    print(
        f"\nCONFIGURAÇÃO ESCOLHIDA (validação): lr={chosen['lr']}, normalize={bool(chosen['normalize'])}, "
        f"val_f1_macro={chosen['val_f1_macro']:.4f}, val_acc={chosen['val_acc']:.4f}, "
        f"best_epoch={int(chosen['best_epoch'])}/{int(chosen['epochs_trained'])}"
    )


if __name__ == "__main__":
    main()
