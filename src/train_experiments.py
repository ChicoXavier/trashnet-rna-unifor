#!/usr/bin/env python3
"""
Etapa 2: Treinamento e Avaliação Controlada dos 6 Experimentos Obrigatórios
Dataset TrashNet - Atividade Final de Redes Neurais Artificiais (EAD Unifor)

Experimentos:
  1. MLP_128x96_no_weights     : MLP | 128 x 96 | Pesos de classe: Não
  2. MLP_128x96_class_weights  : MLP | 128 x 96 | Pesos de classe: Sim
  3. CNN_128x96_no_weights     : CNN | 128 x 96 | Pesos de classe: Não
  4. CNN_128x96_class_weights  : CNN | 128 x 96 | Pesos de classe: Sim
  5. CNN_256x192_no_weights    : CNN | 256 x 192 | Pesos de classe: Não
  6. CNN_512x384_no_weights    : CNN | 512 x 384 | Pesos de classe: Não
"""

import os
import gc
import time
import json
import random
import argparse
import platform
import subprocess
from pathlib import Path

import psutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

# Configurar 4 threads físicas para processamento em CPU
torch.set_num_threads(4)

CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
SEED = 42

# Escolhidos na validação, ver results/mlp_tuning.csv
MLP_LR = 1e-4
MLP_NORMALIZE = True

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dataset-resized"
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preload_split_tensors(df_split: pd.DataFrame, width: int, height: int):
    """
    Carrega todas as imagens de um subconjunto (train, val ou test),
    converte para RGB, redimensiona para (width, height) [largura x altura]
    a partir da versão de referência (512 x 384) e normaliza dividindo por 255.0.
    Retorna tensores X de formato (N, 3, height, width) [canais, altura, largura]
    e y de formato (N,).
    """
    n_samples = len(df_split)
    x_tensor = torch.empty((n_samples, 3, height, width), dtype=torch.float32)
    y_tensor = torch.empty((n_samples,), dtype=torch.long)

    for i, row in enumerate(df_split.itertuples(index=False)):
        img_path = DATA_DIR / row.relative_path
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            if (width, height) != img_rgb.size:
                img_rgb = img_rgb.resize((width, height), Image.Resampling.BILINEAR)
            arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
            # Transpor de (Altura, Largura, Canais) para (Canais, Altura, Largura)
            x_tensor[i] = torch.from_numpy(arr.transpose(2, 0, 1))
            y_tensor[i] = int(row.class_idx)

    return x_tensor, y_tensor


class TrashNetMemoryDataset(Dataset):
    """Dataset em memória com aumento de dados leve opcional APENAS para treinamento."""

    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor, augment: bool = False):
        self.x = x_tensor
        self.y = y_tensor
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        if self.augment:
            # Espelhamento horizontal aleatório (p=0.5)
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            # Pequena rotação aleatória entre -10 e +10 graus (p=0.5)
            if torch.rand(1).item() < 0.5:
                angle = (torch.rand(1).item() * 20.0) - 10.0
                img = TF.rotate(img, angle=angle)
        return img, label


class TrashNetMLP(nn.Module):
    """
    Arquitetura MLP conforme Seção 3 da proposta e Percurso 2:
    - Entrada: pixels RGB organizados como vetor (altura * largura * 3 = 96 * 128 * 3 = 36.864)
    - Opcionalmente normaliza a entrada por canal RGB com média e desvio do treino (buffers, sem alterar parâmetros)
    - Duas camadas ocultas totalmente conectadas (256 e 128 neurônios) com ativação ReLU e Dropout(0.2)
    - Camada de saída: 6 neurônios (1 para cada classe de resíduo)
    """

    def __init__(
        self,
        height: int = 96,
        width: int = 128,
        num_classes: int = 6,
        normalize: bool = False,
        mean=None,
        std=None,
    ):
        super().__init__()
        self.normalize = normalize
        if self.normalize:
            mean_vals = [0.6727, 0.6397, 0.6047] if mean is None else list(mean)
            std_vals = [0.1837, 0.1828, 0.1934] if std is None else list(std)
            self.register_buffer("mean", torch.tensor(mean_vals, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std_vals, dtype=torch.float32).view(1, 3, 1, 1))
        input_dim = height * width * 3
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        return self.network(x)


class TrashNetCNN(nn.Module):
    """
    Arquitetura CNN conforme Seção 3 da proposta e Percurso 4:
    - Três blocos de convolução + BatchNorm + ReLU + MaxPool2d(2x2)
      Bloco 1: 16 filtros 3x3 (stride=2, padding=1) + ReLU + MaxPool(2x2)
      Bloco 2: 32 filtros 3x3 (stride=1, padding=1) + ReLU + MaxPool(2x2)
      Bloco 3: 64 filtros 3x3 (stride=1, padding=1) + ReLU + MaxPool(2x2)
    - Global Average Pooling (AdaptiveAvgPool2d(1)) antes da classificação:
      garante exatamente a MESMA quantidade de parâmetros nas três resoluções
      (128x96, 256x192 e 512x384).
    - Camada oculta densa de 64 neurônios com ReLU + Dropout(0.2) e saída para 6 classes.
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            # Bloco Convolucional 1
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco Convolucional 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco Convolucional 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    avg_loss = total_loss / max(1, total_samples)
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return avg_loss, acc, f1_macro, y_true, y_pred, y_prob


def compute_detailed_test_metrics(y_true: np.ndarray, y_pred: np.ndarray):
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

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": per_class,
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_row_pct": cm_pct.tolist(),
    }


def run_single_experiment(
    exp_id: str,
    exp_label: str,
    arch_type: str,
    resolution_wh: tuple,
    use_class_weights: bool,
    data_cache: dict,
    class_weights_tensor: torch.Tensor,
    max_epochs: int = 35,
    patience: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    mlp_normalize: bool = False,
    mlp_mean=None,
    mlp_std=None,
):
    print(f"\n{'='*72}")
    print(f"Iniciando {exp_id}: {exp_label}")
    print(f"Arquitetura: {arch_type} | Resolução (LxA): {resolution_wh[0]}x{resolution_wh[1]} | Pesos de classe: {use_class_weights}")
    print(f"{'='*72}")

    # Reiniciar semente fixa antes da criação do modelo e dos DataLoaders para isolar efeitos
    set_all_seeds(SEED)
    device = torch.device("cpu")
    process = psutil.Process(os.getpid())

    width, height = resolution_wh
    x_train, y_train, x_val, y_val, x_test, y_test = data_cache[resolution_wh]

    # Aumento de dados idêntico no treinamento de MLP e CNN (protocolo justo)
    augment_train = True
    train_dataset = TrashNetMemoryDataset(x_train, y_train, augment=augment_train)
    val_dataset = TrashNetMemoryDataset(x_val, y_val, augment=False)
    test_dataset = TrashNetMemoryDataset(x_test, y_test, augment=False)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if arch_type == "MLP":
        model = TrashNetMLP(
            height=height,
            width=width,
            num_classes=len(CLASSES),
            normalize=mlp_normalize,
            mean=mlp_mean,
            std=mlp_std,
        ).to(device)
    else:
        model = TrashNetCNN(num_classes=len(CLASSES)).to(device)

    n_params = count_parameters(model)

    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Critério de validação sem pesos para curva comparável e seleção por F1 macro de validação
    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1_macro": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "epoch_time_sec": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 1
    best_state_dict = None
    epochs_no_improve = 0
    peak_ram_mb = process.memory_info().rss / (1024.0 * 1024.0)

    train_start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        ep_start = time.perf_counter()
        model.train()

        running_loss = 0.0
        running_samples = 0
        train_preds = []
        train_targets = []

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            bs = y_batch.size(0)
            running_loss += loss.item() * bs
            running_samples += bs

            preds = torch.argmax(logits.detach(), dim=1)
            train_preds.append(preds.cpu().numpy())
            train_targets.append(y_batch.cpu().numpy())

        curr_ram_mb = process.memory_info().rss / (1024.0 * 1024.0)
        if curr_ram_mb > peak_ram_mb:
            peak_ram_mb = curr_ram_mb

        train_y_true = np.concatenate(train_targets)
        train_y_pred = np.concatenate(train_preds)
        train_loss = running_loss / max(1, running_samples)
        train_acc = float(accuracy_score(train_y_true, train_y_pred))
        train_f1 = float(f1_score(train_y_true, train_y_pred, average="macro", zero_division=0))

        val_loss, val_acc, val_f1, _, _, _ = evaluate_model(model, val_loader, val_criterion, device)
        ep_time = time.perf_counter() - ep_start
        curr_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_f1)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["train_f1_macro"].append(round(train_f1, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1_macro"].append(round(val_f1, 4))
        history["epoch_time_sec"].append(round(ep_time, 3))
        history["lr"].append(curr_lr)

        # Seleção do melhor modelo pelo F1 macro de validação (conforme Seção 4 da proposta)
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = "* [MELHOR]"
        else:
            epochs_no_improve += 1
            marker = ""

        print(
            f"Época {epoch:02d}/{max_epochs:02d} ({ep_time:5.2f}s) | "
            f"Treino Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} {marker}"
        )

        if epochs_no_improve >= patience:
            print(f"Early Stopping acionado na época {epoch} (melhor época: {best_epoch} com Val F1 Macro = {best_val_f1:.4f}).")
            break

    total_train_time = time.perf_counter() - train_start_time
    epochs_trained = len(history["epoch"])
    avg_epoch_time = total_train_time / max(1, epochs_trained)

    # Restaurar os melhores pesos obtidos na validação
    model.load_state_dict(best_state_dict)

    # Salvar o modelo e medir tamanho em disco
    model_path = MODELS_DIR / f"{exp_id}.pt"
    ckpt_payload = {
        "exp_id": exp_id,
        "arch_type": arch_type,
        "resolution_wh": resolution_wh,
        "use_class_weights": use_class_weights,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_f1_macro": round(best_val_f1, 4),
        "model_state_dict": best_state_dict,
        "classes": CLASSES,
    }
    if arch_type == "MLP":
        ckpt_payload["mlp_normalize"] = mlp_normalize
        ckpt_payload["learning_rate_initial"] = lr
    torch.save(ckpt_payload, model_path)
    model_size_kb = round(model_path.stat().st_size / 1024.0, 2)
    model_size_mb = round(model_size_kb / 1024.0, 3)

    # Avaliação final no conjunto de validação e de teste com o melhor modelo salvo
    val_loss_best, val_acc_best, val_f1_best, _, _, _ = evaluate_model(model, val_loader, val_criterion, device)
    t_inf_start = time.perf_counter()
    test_loss, test_acc, test_f1, y_true_test, y_pred_test, y_prob_test = evaluate_model(
        model, test_loader, val_criterion, device
    )
    t_inf_total = time.perf_counter() - t_inf_start
    inference_ms_per_image = round((t_inf_total / max(1, len(y_true_test))) * 1000.0, 4)
    test_metrics = compute_detailed_test_metrics(y_true_test, y_pred_test)

    # Salvar histórico por época em CSV
    pd.DataFrame(history).to_csv(RESULTS_DIR / f"history_{exp_id}.csv", index=False)

    # Salvar previsões individuais no conjunto de teste para reprodutibilidade e análise de erros
    df_test_meta = pd.read_csv(SPLITS_DIR / "test_seed42.csv")
    preds_df = df_test_meta[["relative_path", "filename", "class_name", "class_idx"]].copy()
    preds_df["pred_idx"] = y_pred_test
    preds_df["pred_class"] = [CLASSES[i] for i in y_pred_test]
    preds_df["correct"] = (preds_df["class_idx"] == preds_df["pred_idx"])
    preds_df["pred_prob"] = [round(float(y_prob_test[i, y_pred_test[i]]), 4) for i in range(len(y_pred_test))]
    for c_idx, c_name in enumerate(CLASSES):
        preds_df[f"prob_{c_name}"] = np.round(y_prob_test[:, c_idx], 4)
    preds_df.to_csv(RESULTS_DIR / f"predictions_{exp_id}.csv", index=False)

    print(
        f"--> Resultado Final {exp_id} (Época Salva {best_epoch}/{epochs_trained}): "
        f"Val F1 Macro = {val_f1_best:.4f} | Teste Acc = {test_metrics['accuracy']:.4f} | "
        f"Teste F1 Macro = {test_metrics['f1_macro']:.4f} | Tempo Total = {total_train_time:.2f}s"
    )

    result_dict = {
        "exp_id": exp_id,
        "exp_label": exp_label,
        "model_type": arch_type,
        "resolution_wh": f"{width} × {height}",
        "input_shape_hwc": f"{height} × {width} × 3",
        "width": width,
        "height": height,
        "use_class_weights": use_class_weights,
        "seed": SEED,
        "batch_size": batch_size,
        "learning_rate_initial": lr,
        "optimizer": "Adam (weight_decay=1e-4)",
        "loss_function": "CrossEntropyLoss (com pesos)" if use_class_weights else "CrossEntropyLoss",
        "parameters_count": n_params,
        "model_file": f"models/{exp_id}.pt",
        "model_size_kb": model_size_kb,
        "model_size_mb": model_size_mb,
        "epochs_executed": epochs_trained,
        "best_epoch": best_epoch,
        "total_train_time_sec": round(total_train_time, 2),
        "avg_epoch_time_sec": round(avg_epoch_time, 3),
        "inference_ms_per_image": inference_ms_per_image,
        "peak_ram_mb": round(peak_ram_mb, 2),
        "val_metrics": {
            "loss": round(val_loss_best, 4),
            "accuracy": round(val_acc_best, 4),
            "f1_macro": round(val_f1_best, 4),
        },
        "test_metrics": test_metrics,
        "history": history,
    }
    if arch_type == "MLP":
        result_dict["mlp_normalize"] = mlp_normalize
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Treinamento controlado dos experimentos TrashNet.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Lista separada por vírgula de exp_id para executar (mescla no JSON existente).",
    )
    args = parser.parse_args()
    only_ids = None
    if args.only:
        only_ids = {s.strip() for s in args.only.split(",") if s.strip()}

    df_train = pd.read_csv(SPLITS_DIR / "train_seed42.csv")
    df_val = pd.read_csv(SPLITS_DIR / "val_seed42.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test_seed42.csv")

    with open(RESULTS_DIR / "eda_summary.json", "r", encoding="utf-8") as f:
        eda_summary = json.load(f)

    mean_rgb = eda_summary["train_rgb_mean_0_1"]
    std_rgb = eda_summary["train_rgb_std_0_1"]
    weights_list = [eda_summary["class_weights_train"][c]["weight"] for c in CLASSES]
    class_weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
    print("Pesos de classe (calculados somente no treino):", dict(zip(CLASSES, weights_list)))

    experiments_config = [
        ("exp1_mlp_128x96_sem_pesos", "Exp 1: MLP (128×96, Sem Pesos)", "MLP", (128, 96), False, 60, 10),
        ("exp2_mlp_128x96_com_pesos", "Exp 2: MLP (128×96, Com Pesos)", "MLP", (128, 96), True, 60, 10),
        ("exp3_cnn_128x96_sem_pesos", "Exp 3: CNN (128×96, Sem Pesos)", "CNN", (128, 96), False, 60, 10),
        ("exp4_cnn_128x96_com_pesos", "Exp 4: CNN (128×96, Com Pesos)", "CNN", (128, 96), True, 60, 10),
        ("exp5_cnn_256x192_sem_pesos", "Exp 5: CNN (256×192, Sem Pesos)", "CNN", (256, 192), False, 60, 10),
        ("exp6_cnn_512x384_sem_pesos", "Exp 6: CNN (512×384, Sem Pesos)", "CNN", (512, 384), False, 60, 10),
    ]

    json_path = RESULTS_DIR / "experiments_results.json"
    existing_by_id = {}
    existing_hw = None
    if only_ids is not None:
        if not json_path.exists():
            raise FileNotFoundError("--only exige results/experiments_results.json existente para mesclagem.")
        with open(json_path, "r", encoding="utf-8") as f:
            existing_payload = json.load(f)
        existing_hw = existing_payload.get("hardware_info")
        for item in existing_payload.get("experiments", []):
            existing_by_id[item["exp_id"]] = item
        valid_ids = {cfg[0] for cfg in experiments_config}
        unknown = only_ids - valid_ids
        if unknown:
            raise ValueError(f"exp_id desconhecido em --only: {sorted(unknown)}")
        print(f"Modo --only ativo: executando apenas {sorted(only_ids)} e preservando os demais experimentos.")

    data_cache = {}
    all_results_by_id = dict(existing_by_id) if only_ids is not None else {}

    # Detectar CPU dinamicamente
    cpu_name = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        try:
            cpu_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    current_os = f"{platform.system()} {platform.release()}"
    current_hw = {
        "cpu": cpu_name,
        "gpu": "Ausente (treinamento executado em CPU; pico de RAM registrado via psutil)",
        "ram_total_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
        "os": current_os,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "measurement_method": (
            "Tempo cronometrado com time.perf_counter() cobrindo exclusivamente os loops de "
            "treinamento e validação por época (excluindo download, leitura de disco e redimensionamento prévio). "
            "Pico de RAM (RSS) monitorado via psutil.Process().memory_info().rss."
        ),
    }

    if existing_hw is not None:
        if existing_hw.get("cpu") == cpu_name and existing_hw.get("os") == current_os:
            final_hw = existing_hw
        else:
            print(f"[AVISO] Hardware diferente do JSON anterior: {cpu_name} / {current_os}")
            final_hw = existing_hw
    else:
        final_hw = current_hw

    for exp_id, exp_label, arch_type, res_wh, use_weights, max_ep, pat in experiments_config:
        if only_ids is not None and exp_id not in only_ids:
            continue

        if res_wh not in data_cache:
            # Liberar resoluções anteriores para medição limpa de RAM
            data_cache.clear()
            gc.collect()
            w, h = res_wh
            print(f"\nPré-carregando tensores em {w}x{h} (excluído da medição de tempo de treinamento)...")
            x_tr, y_tr = preload_split_tensors(df_train, w, h)
            x_va, y_va = preload_split_tensors(df_val, w, h)
            x_te, y_te = preload_split_tensors(df_test, w, h)
            data_cache[res_wh] = (x_tr, y_tr, x_va, y_va, x_te, y_te)

        lr_used = MLP_LR if arch_type == "MLP" else 1e-3
        norm_used = MLP_NORMALIZE if arch_type == "MLP" else False

        res = run_single_experiment(
            exp_id=exp_id,
            exp_label=exp_label,
            arch_type=arch_type,
            resolution_wh=res_wh,
            use_class_weights=use_weights,
            data_cache=data_cache,
            class_weights_tensor=class_weights_tensor,
            max_epochs=max_ep,
            patience=pat,
            batch_size=32,
            lr=lr_used,
            mlp_normalize=norm_used,
            mlp_mean=mean_rgb,
            mlp_std=std_rgb,
        )
        all_results_by_id[exp_id] = res

        ordered_results = [
            all_results_by_id[cfg[0]]
            for cfg in experiments_config
            if cfg[0] in all_results_by_id
        ]

        # Salvar checkpoint incremental de resultados
        summary_payload = {
            "hardware_info": final_hw,
            "experiments": ordered_results,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print("\nTodos os 6 experimentos obrigatórios foram concluídos e salvos em results/experiments_results.json!")


if __name__ == "__main__":
    main()
