"""
Centraliza todos los hiperparámetros del pipeline CT y del modelo UNET.
Modificar aquí afecta todo el sistema sin tocar lógica de negocio.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    """
    Configuración global del pipeline de preprocesamiento CT.

    Justificación de target_size=224:
        Estándar consolidado en literatura de clasificación de imágenes médicas
        con CNN. He et al. (2016) ResNet, Huang et al. (2017) DenseNet y
        Rajpurkar et al. (2017) CheXNet usan 224×224 como entrada. Este tamaño
        balancea resolución diagnóstica y costo computacional en GPU con 16GB.

    Justificación SNR_LIMIT=20.0:
        Por debajo de 20 dB el ruido comienza a enmascarar texturas finas como
        opacidades en vidrio esmerilado características de COVID-19
        (Firmeza et al., 2021).

    Justificación SFM_LIMIT=0.65:
        Spectral Flatness Measure > 0.65 indica espectro de potencia plano,
        característico de ruido blanco térmico de equipos CT (Welch, 1967).

    Justificación SP_DENSITY_LIMIT=0.005:
        Densidad de outliers > 0.5% indica presencia estadísticamente
        significativa de ruido impulsivo sal y pimienta.

    Justificación normalización min-max:
        Para CNN con activación ReLU, la normalización min-max [0,1] preserva
        la distribución relativa de intensidades Hounsfield, crítica para
        distinguir tejido pulmonar (HU: -700 a -600) de consolidaciones
        (HU: -100 a 100). LeCun et al. (2012) recomiendan escalar a [0,1]
        para imágenes médicas con rangos dinámicos variables entre equipos.
    """

    # Geometría
    target_size: Tuple[int, int] = (224, 224)

    # Umbrales de detección de ruido
    SNR_LIMIT: float = 20.0
    SFM_LIMIT: float = 0.65
    SP_DENSITY_LIMIT: float = 0.005

    # Umbral de descarte por calidad
    SNR_DISCARD: float = 10.0

    # Parámetros NLM
    NLM_H: int = 4
    NLM_TEMPLATE_WINDOW: int = 7
    NLM_SEARCH_WINDOW: int = 21

    # Parámetros CLAHE
    CLAHE_CLIP_LIMIT: float = 1.1
    CLAHE_TILE_GRID: Tuple[int, int] = (8, 8)

    # Augmentation
    ROTATION_RANGE: float = 15.0
    ZOOM_RANGE: float = 0.10
    AUG_SEED: int = 42

    # Pipeline
    META_TASA_EXITO: float = 95.0


@dataclass(frozen=True)
class UNetConfig:
    """
    Configuración del Autoencoder convolucional para clasificación COVID/No-COVID.

    Justificación IMG_SIZE=128:
        Las imágenes ya fueron preprocesadas a 224×224 por el pipeline CT.
        Se reduce a 128×128 en el dataset loader para balancear resolución
        y costo de memoria en el espacio latente: 4 poolings de ×2 producen
        un feature map de 8×8, manejable con LATENT_DIM=256.

    Justificación LATENT_DIM=256:
        Dimensión suficiente para capturar patrones de opacidad en vidrio
        esmerilado y consolidaciones pulmonares sin sobreajuste en datasets
        medianos (~1000 imágenes). Valores mayores (512+) requieren más datos.

    Justificación LAMBDA=0.1:
        Peso de la pérdida contrastiva frente a la de reconstrucción.
        Un valor muy alto colapsa el espacio latente; muy bajo no separa clases.
        0.1 mantiene la reconstrucción como objetivo principal y la separación
        de clases como regularizador.

    Justificación MARGIN=2.0:
        Margen de la ContrastiveLoss. Con vectores normalizados por BatchNorm
        y LATENT_DIM=256, una distancia euclidiana de 2.0 es un objetivo
        razonable para separar clases sin forzar separación excesiva.

    Justificación LR=1e-3 con ReduceLROnPlateau:
        Adam con LR inicial 1e-3 converge rápido en las primeras épocas.
        El scheduler reduce a la mitad si la pérdida no mejora en 5 épocas,
        evitando oscilaciones al final del entrenamiento.
    """

    # Geometría de entrada al modelo (distinto a target_size del preprocesamiento)
    IMG_SIZE: int = 128

    # Espacio latente
    LATENT_DIM: int = 256

    # Entrenamiento
    BATCH_SIZE: int = 16
    EPOCHS: int = 50
    LR: float = 1e-3

    # Pérdida combinada: loss = loss_recon + LAMBDA * loss_contrastiva
    LAMBDA: float = 0.1
    MARGIN: float = 2.0

    # Scheduler
    LR_PATIENCE: int = 5
    LR_FACTOR: float = 0.5

    # Evaluación
    CONFIDENCE_HIGH: float = 85.0
    CONFIDENCE_MED: float = 70.0

    # Persistencia
    MODEL_PATH: str = "autoencoder_covid.pth"
    CENTROIDS_PATH: str = "centroides_covid.pkl"

    # Clases (orden importa: índice 0 = No-COVID, índice 1 = COVID)
    CLASSES: Tuple[str, ...] = ("Non-COVID-19", "COVID-19")


CONFIG      = PipelineConfig()
UNET_CONFIG = UNetConfig()