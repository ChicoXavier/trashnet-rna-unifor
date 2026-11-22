#!/usr/bin/env python3
"""
Objetivo Extra Opcional (Seção 7): Transferência de Aprendizagem com YOLO (modo classify)
- Preserva integralmente as 6 classes e a divisão estratificada original (seed=42)
  em data/splits/{train,val,test}_seed42.csv, evitando qualquer nova divisão automática.
- Avalia o modelo ajustado no mesmo conjunto de teste (380 imagens) e registra métricas
  de desempenho (acurácia, F1 macro, precisão, recall, F1 e suporte por classe, matriz de confusão)
  e métricas de custo computacional (parâmetros, tamanho do modelo, épocas, tempo, RAM).
"""

import os
import time
import json
import shutil
import platform
import subprocess
from pathlib import Path

import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from ultralytics import YOLO

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dataset-resized"
SPLITS_DIR = BASE_DIR / "data" / "splits"
YOLO_DATA_DIR = BASE_DIR / "data" / "yolo_trashnet_split42"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def prepare_yolo_directory_structure():
    """
    Cria a estrutura de pastas data/yolo_trashnet_split42/{train,val,test}/{class_name}
    usando os arquivos exatos de train_seed42.csv, val_seed42.csv e test_seed42.csv.
    """
    if YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)

    for split_name in ["train", "val", "test"]:
        df_split = pd.read_csv(SPLITS_DIR / f"{split_name}_seed42.csv")
        for cls_name in CLASSES:
            (YOLO_DATA_DIR / split_name / cls_name).mkdir(parents=True, exist_ok=True)
        for row in df_split.itertuples(index=False):
            src = DATA_DIR / row.relative_path
            dst = YOLO_DATA_DIR / split_name / row.class_name / row.filename
            os.symlink(src, dst)

    print(f"Estrutura YOLO preparada em {YOLO_DATA_DIR} preservando exatamente a divisão seed={SEED}.")


def main():
    prepare_yolo_directory_structure()

    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024.0 * 1024.0)

    # Modelo pré-treinado no ImageNet (modo classify)
    base_weights = "yolo11n-cls.pt"
    imgsz = 224
    epochs = 20
    batch = 32
    lr0 = 0.001

    print(f"Carregando {base_weights} (pesos pré-treinados ImageNet) para ajuste fino no TrashNet...")
    model = YOLO(base_weights)

    t_start = time.perf_counter()
    results = model.train(
        data=str(YOLO_DATA_DIR),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        optimizer="AdamW",
        seed=SEED,
        deterministic=True,
        device="cpu",
        workers=4,
        patience=7,
        project=str(RESULTS_DIR / "yolo_runs"),
        name="yolo11n_cls_trashnet",
        exist_ok=True,
        verbose=True,
    )
    total_train_time = time.perf_counter() - t_start

    curr_ram_mb = process.memory_info().rss / (1024.0 * 1024.0)
    peak_ram_mb = max(peak_ram_mb, curr_ram_mb)

    run_dir = RESULTS_DIR / "yolo_runs" / "yolo11n_cls_trashnet"
    best_pt_src = run_dir / "weights" / "best.pt"
    best_pt_dst = MODELS_DIR / "exp7_yolo11n_cls_bonus.pt"
    shutil.copy2(best_pt_src, best_pt_dst)
    model_size_kb = round(best_pt_dst.stat().st_size / 1024.0, 2)
    model_size_mb = round(model_size_kb / 1024.0, 3)

    # Ler histórico de treinamento do YOLO (results.csv)
    yolo_csv = pd.read_csv(run_dir / "results.csv")
    yolo_csv.columns = [c.strip() for c in yolo_csv.columns]
    epochs_executed = len(yolo_csv)
    avg_epoch_time = total_train_time / max(1, epochs_executed)

    # Carregar melhor modelo salvo e avaliar exatamente no mesmo conjunto de teste (380 imagens)
    best_model = YOLO(str(best_pt_dst))
    n_params = sum(p.numel() for p in best_model.model.parameters())

    # Mapeamento de nomes do YOLO (ordem alfabética) para nossa ordem canônica CLASSES
    yolo_names = best_model.names  # dict {idx: name}

    df_test = pd.read_csv(SPLITS_DIR / "test_seed42.csv")
    test_paths = [str(DATA_DIR / p) for p in df_test["relative_path"]]

    t_inf_start = time.perf_counter()
    pred_results = best_model.predict(test_paths, imgsz=imgsz, device="cpu", verbose=False)
    t_inf_total = time.perf_counter() - t_inf_start
    inference_ms_per_image = round((t_inf_total / max(1, len(test_paths))) * 1000.0, 4)

    y_true = df_test["class_idx"].to_numpy()
    y_pred = []
    probs_canonical = np.zeros((len(df_test), len(CLASSES)), dtype=np.float32)

    name_to_canonical_idx = {c: i for i, c in enumerate(CLASSES)}

    for i, r in enumerate(pred_results):
        probs_vec = r.probs.data.cpu().numpy()
        # Reordenar probabilidades da ordem do YOLO para a ordem canônica CLASSES
        for y_idx, y_name in yolo_names.items():
            c_idx = name_to_canonical_idx[y_name]
            probs_canonical[i, c_idx] = float(probs_vec[y_idx])
        top1_yolo_idx = int(r.probs.top1)
        top1_name = yolo_names[top1_yolo_idx]
        y_pred.append(name_to_canonical_idx[top1_name])

    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASSES))), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_pct = (cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0).round(2)

    per_class = {}
    for idx, cls_name in enumerate(CLASSES):
        per_class[cls_name] = {
            "precision": round(float(prec[idx]), 4),
            "recall": round(float(rec[idx]), 4),
            "f1": round(float(f1_cls[idx]), 4),
            "support": int(supp[idx]),
        }

    # Salvar previsões no mesmo formato dos experimentos obrigatórios
    preds_df = df_test[["relative_path", "filename", "class_name", "class_idx"]].copy()
    preds_df["pred_idx"] = y_pred
    preds_df["pred_class"] = [CLASSES[i] for i in y_pred]
    preds_df["correct"] = (preds_df["class_idx"] == preds_df["pred_idx"])
    preds_df["pred_prob"] = [round(float(probs_canonical[i, y_pred[i]]), 4) for i in range(len(y_pred))]
    for c_idx, c_name in enumerate(CLASSES):
        preds_df[f"prob_{c_name}"] = np.round(probs_canonical[:, c_idx], 4)
    preds_df.to_csv(RESULTS_DIR / "predictions_exp7_yolo11n_cls_bonus.csv", index=False)

    # Salvar resumo do experimento bônus
    cpu_name = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass

    bonus_summary = {
        "exp_id": "exp7_yolo11n_cls_bonus",
        "exp_label": "Exp 7 (Bônus): YOLO11n-cls (224×224, Transfer Learning)",
        "yolo_version": "ultralytics 8.4.216",
        "base_model": base_weights,
        "weights_origin": "Pré-treinado no ImageNet-1K (Ultralytics)",
        "resolution_wh": f"{imgsz} × {imgsz}",
        "transformations": "Redimensionamento para 224×224, normalização [0,1] e aumentação padrão do YOLO classify (randaugment/flip/hsv) apenas no treino",
        "hyperparameters": {
            "epochs_max": epochs,
            "epochs_executed": epochs_executed,
            "batch_size": batch,
            "optimizer": "AdamW",
            "lr0": lr0,
            "patience": 7,
            "seed": SEED,
        },
        "parameters_count": n_params,
        "model_file": "models/exp7_yolo11n_cls_bonus.pt",
        "model_size_kb": model_size_kb,
        "model_size_mb": model_size_mb,
        "total_train_time_sec": round(total_train_time, 2),
        "avg_epoch_time_sec": round(avg_epoch_time, 3),
        "inference_ms_per_image": inference_ms_per_image,
        "peak_ram_mb": round(peak_ram_mb, 2),
        "cpu": cpu_name,
        "test_metrics": {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "per_class": per_class,
            "confusion_matrix_counts": cm.tolist(),
            "confusion_matrix_row_pct": cm_pct.tolist(),
        },
        "history_csv": "results/yolo_runs/yolo11n_cls_trashnet/results.csv",
    }

    with open(RESULTS_DIR / "yolo_bonus_results.json", "w", encoding="utf-8") as f:
        json.dump(bonus_summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n--> Resultado YOLO Bônus: Teste Acc = {acc:.4f} | Teste F1 Macro = {f1_macro:.4f} | "
        f"Tempo Total = {total_train_time:.2f}s | Parâmetros = {n_params:,}"
    )


if __name__ == "__main__":
    main()
