#!/usr/bin/env python3
"""
Script de Avaliação Sem Novo Treinamento (Seção 6 - Reprodutibilidade).
Carrega os 6 modelos obrigatórios (models/exp1..exp6.pt) e o modelo bônus (models/exp7_yolo11n_cls_bonus.pt)
e recalcula as métricas diretamente sobre data/splits/test_seed42.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from ultralytics import YOLO

from train_experiments import (
    CLASSES,
    TrashNetMLP,
    TrashNetCNN,
    TrashNetMemoryDataset,
    preload_split_tensors,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dataset-resized"
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODELS_DIR = BASE_DIR / "models"


def evaluate_pytorch_checkpoints():
    df_test = pd.read_csv(SPLITS_DIR / "test_seed42.csv")
    device = torch.device("cpu")
    tensor_cache = {}

    pt_files = sorted([p for p in MODELS_DIR.glob("exp[1-6]_*.pt")])
    if not pt_files:
        raise FileNotFoundError("Nenhum checkpoint exp1..exp6 encontrado em models/.")

    print("=" * 76)
    print(f"AVALIAÇÃO REPRODUTÍVEL SEM NOVO TREINAMENTO (CONJUNTO DE TESTE: {len(df_test)} IMAGENS)")
    print("=" * 76)

    for pt_path in pt_files:
        ckpt = torch.load(pt_path, map_location=device, weights_only=False)
        arch_type = ckpt["arch_type"]
        w, h = tuple(ckpt["resolution_wh"])

        if (w, h) not in tensor_cache:
            tensor_cache[(w, h)] = preload_split_tensors(df_test, w, h)
        x_test, y_test = tensor_cache[(w, h)]
        loader = DataLoader(TrashNetMemoryDataset(x_test, y_test, augment=False), batch_size=32, shuffle=False)

        if arch_type == "MLP":
            model = TrashNetMLP(height=h, width=w, num_classes=len(CLASSES)).to(device)
        else:
            model = TrashNetCNN(num_classes=len(CLASSES)).to(device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                logits = model(xb.to(device))
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())

        y_pred = np.concatenate(preds)
        y_true = y_test.numpy()
        acc = accuracy_score(y_true, y_pred)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"{ckpt['exp_id']:32s} | Melhor Época: {ckpt['best_epoch']:02d} | Teste Acc: {acc:.4f} | Teste F1 Macro: {f1_m:.4f}")

    # Avaliar bônus YOLO se disponível
    yolo_pt = MODELS_DIR / "exp7_yolo11n_cls_bonus.pt"
    if yolo_pt.exists():
        yolo_model = YOLO(str(yolo_pt))
        yolo_names = yolo_model.names
        name_to_idx = {c: i for i, c in enumerate(CLASSES)}
        test_paths = [str(DATA_DIR / p) for p in df_test["relative_path"]]
        results = yolo_model.predict(test_paths, imgsz=224, device="cpu", verbose=False)
        y_pred_yolo = np.array([name_to_idx[yolo_names[int(r.probs.top1)]] for r in results])
        y_true_yolo = df_test["class_idx"].to_numpy()
        acc_y = accuracy_score(y_true_yolo, y_pred_yolo)
        f1_y = f1_score(y_true_yolo, y_pred_yolo, average="macro", zero_division=0)
        print(f"{'exp7_yolo11n_cls_bonus':32s} | Transfer Learning | Teste Acc: {acc_y:.4f} | Teste F1 Macro: {f1_y:.4f}")


if __name__ == "__main__":
    evaluate_pytorch_checkpoints()
