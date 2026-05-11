"""
Evaluación del Autoencoder entrenado sobre el conjunto de test.

Responsabilidades:
    - Clasificar imágenes por distancia euclidiana al centroide
    - Calcular métricas: accuracy, classification report, AUC-ROC
    - Analizar distribución de confianza por zona (alta / media / baja)
    - Clasificar una sola imagen nueva desde disco
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from config import UNetConfig
from model import Autoencoder


# ─────────────────────────────────────────────
# FUNCIONES ATÓMICAS DE CLASIFICACIÓN
# ─────────────────────────────────────────────

def compute_distance(
    z: np.ndarray,
    centroide: np.ndarray,
) -> float:
    """
    Calcula la distancia euclidiana entre un vector Z y un centroide.

    Args:
        z: Vector latente de una imagen, forma (latent_dim,).
        centroide: Centroide de clase, forma (latent_dim,).

    Returns:
        Distancia euclidiana como float.
    """
    return float(np.linalg.norm(z - centroide))


def predict_from_distances(
    dist_covid: float,
    dist_no_covid: float,
) -> Tuple[int, float]:
    """
    Determina clase y confianza a partir de distancias a centroides.

    La confianza refleja qué tan más cercano está el vector al
    centroide ganador respecto al total de distancias.

    Args:
        dist_covid: Distancia euclidiana al centroide COVID.
        dist_no_covid: Distancia euclidiana al centroide No-COVID.

    Returns:
        Tupla (prediccion, confianza_porcentaje).
            prediccion: 1 si COVID, 0 si No-COVID.
            confianza: valor en [0, 100].
    """
    dist_total = dist_covid + dist_no_covid
    if dist_covid < dist_no_covid:
        return 1, (dist_no_covid / dist_total) * 100.0
    return 0, (dist_covid / dist_total) * 100.0


def extract_latent_vectors(
    modelo: Autoencoder,
    loader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae todos los vectores Z y etiquetas de un DataLoader.

    Args:
        modelo: Autoencoder entrenado en modo eval.
        loader: DataLoader del conjunto a evaluar.
        device: 'cuda' o 'cpu'.

    Returns:
        Tupla (z_array, labels_array):
            z_array: forma (N, latent_dim).
            labels_array: forma (N,).
    """
    z_list:     List[np.ndarray] = []
    label_list: List[int]        = []

    modelo.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extrayendo vectores Z (test)"):
            imgs = imgs.to(device)
            _, z = modelo(imgs)
            z_list.append(z.cpu().numpy())
            label_list.extend(labels.numpy())

    return np.vstack(z_list), np.array(label_list)


def classify_batch(
    z_array: np.ndarray,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clasifica un batch de vectores Z por distancia al centroide.

    Args:
        z_array: Vectores latentes, forma (N, latent_dim).
        centroide_covid: Centroide COVID, forma (latent_dim,).
        centroide_no_covid: Centroide No-COVID, forma (latent_dim,).

    Returns:
        Tupla (predicciones, scores_covid):
            predicciones: array de 0/1, forma (N,).
            scores_covid: confianza normalizada para AUC, forma (N,).
    """
    predicciones: List[int]   = []
    scores_covid: List[float] = []

    for z in z_array:
        d_covid    = compute_distance(z, centroide_covid)
        d_no_covid = compute_distance(z, centroide_no_covid)
        pred, _    = predict_from_distances(d_covid, d_no_covid)
        predicciones.append(pred)
        scores_covid.append(d_no_covid / (d_covid + d_no_covid))

    return np.array(predicciones), np.array(scores_covid)


# ─────────────────────────────────────────────
# EVALUACIÓN COMPLETA SOBRE TEST SET
# ─────────────────────────────────────────────

def evaluate_test_set(
    modelo: Autoencoder,
    dataset_test: Dataset,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    device: str,
    cfg: UNetConfig,
) -> Dict:
    """
    Evalúa el modelo sobre el conjunto de test y reporta métricas completas.

    Métricas calculadas:
        - Classification report (precision, recall, F1 por clase)
        - AUC-ROC
        - Accuracy train vs test
        - Distribución de confianza por zona (alta / media / baja)

    Args:
        modelo: Autoencoder entrenado.
        dataset_test: Subset de test.
        centroide_covid: Centroide COVID en espacio latente.
        centroide_no_covid: Centroide No-COVID en espacio latente.
        device: 'cuda' o 'cpu'.
        cfg: Instancia de UNetConfig con umbrales de confianza.

    Returns:
        Diccionario con predicciones, labels, scores y métricas.
    """
    loader_test = DataLoader(
        dataset_test, batch_size=32, shuffle=False, num_workers=2
    )

    z_test, labels_test = extract_latent_vectors(modelo, loader_test, device)
    predicciones, scores_covid = classify_batch(
        z_test, centroide_covid, centroide_no_covid
    )

    # ── Reporte de clasificación ────────────────────────────────────
    sep = "=" * 55
    print(f"\n{sep}")
    print("     RESULTADOS EN DATOS DE PRUEBA (nunca vistos)")
    print(f"{sep}")
    print(classification_report(
        labels_test,
        predicciones,
        target_names=["No-COVID", "COVID"],
    ))

    try:
        auc = roc_auc_score(labels_test, scores_covid)
        print(f"AUC-ROC: {auc:.4f}")
    except ValueError:
        pass

    # ── Análisis de confianza por zona ──────────────────────────────
    _report_confidence_zones(
        z_test, labels_test, predicciones,
        centroide_covid, centroide_no_covid, cfg,
    )

    # ── Métricas de separación en espacio latente ───────────────────
    _report_latent_separation(
        z_test, labels_test, centroide_covid, centroide_no_covid
    )

    return {
        "z_test":        z_test,
        "labels_test":   labels_test,
        "predicciones":  predicciones,
        "scores_covid":  scores_covid,
        "accuracy":      accuracy_score(labels_test, predicciones),
    }


def _report_confidence_zones(
    z_array: np.ndarray,
    labels: np.ndarray,
    predicciones: np.ndarray,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    cfg: UNetConfig,
) -> None:
    """
    Imprime tabla de correctos/errores segmentada por zona de confianza.

    Zonas definidas en UNetConfig:
        Alta   : confianza >= CONFIDENCE_HIGH
        Media  : CONFIDENCE_MED <= confianza < CONFIDENCE_HIGH
        Baja   : confianza < CONFIDENCE_MED

    Args:
        z_array: Vectores latentes del test, forma (N, latent_dim).
        labels: Etiquetas reales, forma (N,).
        predicciones: Predicciones del modelo, forma (N,).
        centroide_covid: Centroide COVID.
        centroide_no_covid: Centroide No-COVID.
        cfg: UNetConfig con umbrales CONFIDENCE_HIGH y CONFIDENCE_MED.
    """
    conteos = {
        "alta":  {"correctos": 0, "errores": 0},
        "media": {"correctos": 0, "errores": 0},
        "baja":  {"correctos": 0, "errores": 0},
    }

    for z, label, pred in zip(z_array, labels, predicciones):
        d_c   = compute_distance(z, centroide_covid)
        d_n   = compute_distance(z, centroide_no_covid)
        _, conf = predict_from_distances(d_c, d_n)
        clave   = "correctos" if pred == label else "errores"

        if conf >= cfg.CONFIDENCE_HIGH:
            conteos["alta"][clave] += 1
        elif conf >= cfg.CONFIDENCE_MED:
            conteos["media"][clave] += 1
        else:
            conteos["baja"][clave] += 1

    print("\nANÁLISIS DE CONFIANZA — TEST SET")
    print("=" * 55)
    print(f"{'Zona':22} {'Correctos':>10} {'Errores':>10} {'Precisión':>10}")
    print("-" * 54)

    for nombre, limites in [
        (f"Alta  (>={cfg.CONFIDENCE_HIGH:.0f}%)", "alta"),
        (f"Media ({cfg.CONFIDENCE_MED:.0f}-{cfg.CONFIDENCE_HIGH:.0f}%)", "media"),
        (f"Baja  (<{cfg.CONFIDENCE_MED:.0f}%)",  "baja"),
    ]:
        c     = conteos[limites]["correctos"]
        e     = conteos[limites]["errores"]
        total = c + e
        prec  = (c / total * 100) if total > 0 else 0.0
        print(f"{nombre:22} {c:>10} {e:>10} {prec:>9.1f}%")

    total_err = sum(v["errores"] for v in conteos.values())
    print(f"\n💡 Casos con confianza <{cfg.CONFIDENCE_MED:.0f}% "
          f"deberían revisarse manualmente en producción.")
    print(f"   Total errores: {total_err}")


def _report_latent_separation(
    z_array: np.ndarray,
    labels: np.ndarray,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
) -> None:
    """
    Imprime métricas de separación geométrica en el espacio latente.

    Métricas:
        - Distancia entre centroides
        - Dispersión interna por clase
        - Índice de separación: dist_centroides / dispersión_promedio
          > 1.0 indica buena separación de clases

    Args:
        z_array: Todos los vectores Z, forma (N, latent_dim).
        labels: Etiquetas reales, forma (N,).
        centroide_covid: Centroide COVID.
        centroide_no_covid: Centroide No-COVID.
    """
    z_covid    = z_array[labels == 1]
    z_no_covid = z_array[labels == 0]

    dist_centroides   = np.linalg.norm(centroide_covid - centroide_no_covid)
    dispersion_covid  = np.mean([
        compute_distance(z, centroide_covid) for z in z_covid
    ])
    dispersion_nocovid = np.mean([
        compute_distance(z, centroide_no_covid) for z in z_no_covid
    ])
    separacion = dist_centroides / ((dispersion_covid + dispersion_nocovid) / 2)

    print("\n── SEPARACIÓN EN ESPACIO LATENTE ──────────────────")
    print(f"  Distancia entre centroides : {dist_centroides:.4f}")
    print(f"  Dispersión interna COVID   : {dispersion_covid:.4f}")
    print(f"  Dispersión interna No-COVID: {dispersion_nocovid:.4f}")
    print(f"  Índice de separación       : {separacion:.4f}")
    print(f"  {'✅ Clases bien separadas' if separacion > 1.0 else '❌ Clases mezcladas'}")


# ─────────────────────────────────────────────
# CLASIFICACIÓN DE UNA IMAGEN NUEVA
# ─────────────────────────────────────────────

def classify_image(
    ruta_imagen: str,
    modelo: Autoencoder,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    img_size: int = 128,
    device: str = "cpu",
    mostrar: bool = True,
) -> Dict:
    """
    Clasifica una imagen TAC nueva como COVID o No-COVID.

    Carga la imagen desde disco, la transforma al mismo formato
    usado en entrenamiento, obtiene su vector Z y clasifica
    por distancia euclidiana al centroide más cercano.

    Args:
        ruta_imagen: Ruta al archivo de imagen a clasificar.
        modelo: Autoencoder entrenado.
        centroide_covid: Centroide COVID en espacio latente.
        centroide_no_covid: Centroide No-COVID en espacio latente.
        img_size: Tamaño de entrada al modelo (debe coincidir con entrenamiento).
        device: 'cuda' o 'cpu'.
        mostrar: Si True, muestra la imagen con el resultado via matplotlib.

    Returns:
        Diccionario con:
            clase (str), confianza (float), dist_covid (float),
            dist_no_covid (float), z (np.ndarray).
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    img        = Image.open(ruta_imagen).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    modelo.eval()
    with torch.no_grad():
        _, z = modelo(img_tensor)
    z_np = z.cpu().numpy().squeeze()

    dist_covid    = compute_distance(z_np, centroide_covid)
    dist_no_covid = compute_distance(z_np, centroide_no_covid)
    pred, confianza = predict_from_distances(dist_covid, dist_no_covid)

    clase = "COVID" if pred == 1 else "No-COVID"

    if mostrar:
        _plot_prediction(img_tensor, clase, confianza, dist_covid, dist_no_covid)

    return {
        "clase":        clase,
        "confianza":    confianza,
        "dist_covid":   dist_covid,
        "dist_no_covid": dist_no_covid,
        "z":            z_np,
    }


def _plot_prediction(
    img_tensor: torch.Tensor,
    clase: str,
    confianza: float,
    dist_covid: float,
    dist_no_covid: float,
) -> None:
    """
    Muestra la imagen clasificada con su resultado y distancias.

    Args:
        img_tensor: Tensor de la imagen (1, 1, H, W).
        clase: Clase predicha ('COVID' o 'No-COVID').
        confianza: Confianza en porcentaje.
        dist_covid: Distancia al centroide COVID.
        dist_no_covid: Distancia al centroide No-COVID.
    """
    import matplotlib.pyplot as plt

    color = "#F44336" if clase == "COVID" else "#2196F3"
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(img_tensor.cpu().squeeze(), cmap="gray")
    ax.set_title(
        f"Predicción: {clase}\nConfianza: {confianza:.1f}%\n"
        f"d(COVID)={dist_covid:.2f} | d(No-COVID)={dist_no_covid:.2f}",
        color=color, fontsize=10, fontweight="bold",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()