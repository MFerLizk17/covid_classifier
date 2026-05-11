"""
Lógica de entrenamiento del Autoencoder y cálculo de centroides.

Separa el ciclo de entrenamiento de la definición del modelo
para mantener atomicidad: cada función tiene una única responsabilidad.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config import UNetConfig
from src.loss import ContrastiveLoss
from src.model import Autoencoder


def train_one_epoch(
    modelo: Autoencoder,
    dataloader: DataLoader,
    cfg: UNetConfig,
    device: str,
) -> Dict[str, List[float]]:
    """
    Ejecuta el ciclo completo de entrenamiento por épocas.

    En cada época:
        1. Forward pass → reconstruida, Z
        2. Calcula loss_recon (MSE) + LAMBDA * loss_contrastiva
        3. Backward + optimizer step
        4. Ajusta LR con ReduceLROnPlateau si la pérdida no mejora

    Args:
        modelo: Autoencoder a entrenar (ya en el device correcto).
        dataloader: DataLoader del conjunto de entrenamiento.
        cfg: Instancia de UNetConfig con hiperparámetros.
        device: 'cuda' o 'cpu'.

    Returns:
        Diccionario con listas de pérdidas por época:
            {
                'total':     [...],
                'recon':     [...],
                'contrast':  [...],
            }
    """
    criterio_recon       = nn.MSELoss()
    criterio_contrastivo = ContrastiveLoss(margin=cfg.MARGIN)
    optimizer            = optim.Adam(modelo.parameters(), lr=cfg.LR)
    scheduler            = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=cfg.LR_PATIENCE,
        factor=cfg.LR_FACTOR,
    )

    historial: Dict[str, List[float]] = {
        "total":    [],
        "recon":    [],
        "contrast": [],
    }

    for epoch in range(1, cfg.EPOCHS + 1):
        modelo.train()

        loss_epoch     = 0.0
        recon_epoch    = 0.0
        contrast_epoch = 0.0

        for imgs, labels in tqdm(
            dataloader,
            desc=f"Época {epoch:>3}/{cfg.EPOCHS}",
            leave=False,
        ):
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            reconstruidas, z = modelo(imgs)

            loss_recon    = criterio_recon(reconstruidas, imgs)
            loss_contrast = criterio_contrastivo(z, labels)
            loss          = loss_recon + cfg.LAMBDA * loss_contrast

            loss.backward()
            optimizer.step()

            loss_epoch     += loss.item()
            recon_epoch    += loss_recon.item()
            contrast_epoch += loss_contrast.item()

        n_batches       = len(dataloader)
        loss_epoch     /= n_batches
        recon_epoch    /= n_batches
        contrast_epoch /= n_batches

        historial["total"].append(loss_epoch)
        historial["recon"].append(recon_epoch)
        historial["contrast"].append(contrast_epoch)

        scheduler.step(loss_epoch)

        if epoch % 5 == 0 or epoch == 1:
            lr_actual = optimizer.param_groups[0]["lr"]
            print(
                f"Época {epoch:>3}/{cfg.EPOCHS} | "
                f"Total: {loss_epoch:.4f} | "
                f"Recon: {recon_epoch:.4f} | "
                f"Contrast: {contrast_epoch:.4f} | "
                f"LR: {lr_actual:.2e}"
            )

    print("\n✓ Entrenamiento completado")
    return historial


def compute_centroids(
    modelo: Autoencoder,
    dataset: Dataset,
    device: str,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula los centroides COVID y No-COVID en el espacio latente.

    Recorre todo el dataset en modo eval, extrae los vectores Z
    y promedia por clase para obtener el centroide de cada una.

    Args:
        modelo: Autoencoder entrenado.
        dataset: Dataset completo (train + test) para centroides globales.
        device: 'cuda' o 'cpu'.
        batch_size: Tamaño de batch para la extracción de vectores Z.

    Returns:
        Tupla (centroide_covid, centroide_no_covid), ambos np.ndarray
        de forma (latent_dim,).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    todos_z:      List[np.ndarray] = []
    todos_labels: List[int]        = []

    modelo.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extrayendo vectores Z"):
            imgs = imgs.to(device)
            _, z = modelo(imgs)
            todos_z.append(z.cpu().numpy())
            todos_labels.extend(labels.numpy())

    todos_z_np      = np.vstack(todos_z)
    todos_labels_np = np.array(todos_labels)

    centroide_covid    = todos_z_np[todos_labels_np == 1].mean(axis=0)
    centroide_no_covid = todos_z_np[todos_labels_np == 0].mean(axis=0)

    dist = np.linalg.norm(centroide_covid - centroide_no_covid)
    print(f"✓ Centroides calculados | Distancia entre clases: {dist:.4f}")

    return centroide_covid, centroide_no_covid