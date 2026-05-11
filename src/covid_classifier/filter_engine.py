"""
Motor de filtrado condicional para imágenes CT.

Responsabilidad única: aplicar filtros según el perfil de ruido.

Justificación del orden de filtros:
    1. Mediana: elimina outliers extremos (sal y pimienta) antes de
       que contaminen el análisis de parches del NLM y el CLAHE.
    2. CLAHE: estandariza el contraste local entre equipos CT
       antes de aplicar NLM para que opere sobre intensidades
       ya normalizadas.
    3. NLM: opera sobre imagen ya libre de outliers y con contraste
       estandarizado, mejorando la estimación de parches similares.
    4. Morfológico: aísla la región pulmonar al final para no
       interferir con los filtros de ruido previos.

Uso del experimento:
    python -m src.filter_engine
"""

import logging
from pathlib import Path
from typing import Dict, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from config import CONFIG

logger = logging.getLogger(__name__)

CLASES        = ["COVID-19", "Non-COVID-19"]
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
RAW_DIR       = Path("./data/raw")


# ──────────────────────────────────────────────────────────────────────────────
# Funciones de filtrado
# ──────────────────────────────────────────────────────────────────────────────

def apply_median_filter(img_norm: np.ndarray) -> np.ndarray:
    """
    Aplica filtro de mediana para eliminar ruido impulsivo (sal y pimienta).

    Justificación kernel 3×3:
        Suficiente para outliers aislados sin difuminar bordes de
        consolidaciones y opacidades características de COVID-19.
        Kernels mayores degradan bordes diagnósticos relevantes.

    Args:
        img_norm: Array float32 normalizado en [0.0, 1.0].

    Returns:
        Array float32 filtrado, normalizado en [0.0, 1.0].
    """
    output_u8 = (img_norm * 255).astype(np.uint8)
    return cv2.medianBlur(output_u8, 3).astype(np.float32) / 255.0


def apply_clahe(img_norm: np.ndarray) -> np.ndarray:
    """
    Aplica CLAHE para estandarizar contraste local entre equipos CT.

    Justificación clipLimit=2.0:
        Limita amplificación de contraste para evitar que el ruido
        residual se amplifique como artefactos. Valores >3.0 generan
        halos en bordes de consolidaciones.
        tileGridSize=(8,8) sobre 224×224 px produce regiones de
        28×28 px, granularidad suficiente para compensar variaciones
        locales entre equipos del dataset de Teherán.

    Args:
        img_norm: Array float32 normalizado en [0.0, 1.0].

    Returns:
        Array float32 con contraste estandarizado, normalizado en [0.0, 1.0].
    """
    clahe     = cv2.createCLAHE(
        clipLimit=CONFIG.CLAHE_CLIP_LIMIT,
        tileGridSize=CONFIG.CLAHE_TILE_GRID
    )
    output_u8 = (img_norm * 255).astype(np.uint8)
    return clahe.apply(output_u8).astype(np.float32) / 255.0


def apply_nlm_filter(img_norm: np.ndarray) -> np.ndarray:
    """
    Aplica Non-Local Means para ruido blanco gaussiano.

    Justificación parámetros:
        h=4: conservador para preservar texturas finas diagnósticas.
        templateWindowSize=7: captura contexto estructural suficiente.
        searchWindowSize=21: ventana amplia para parches similares
                             en tejido pulmonar homogéneo.

    Args:
        img_norm: Array float32 normalizado en [0.0, 1.0].

    Returns:
        Array float32 filtrado, normalizado en [0.0, 1.0].
    """
    output_u8 = (img_norm * 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(
        output_u8, None,
        h=CONFIG.NLM_H,
        templateWindowSize=CONFIG.NLM_TEMPLATE_WINDOW,
        searchWindowSize=CONFIG.NLM_SEARCH_WINDOW
    ).astype(np.float32) / 255.0


def apply_morphological_lung_mask(img_norm: np.ndarray) -> np.ndarray:
    """
    Segmenta la región pulmonar replicando el pipeline de referencia.

    Aplica internamente windowing simulado y filtros para garantizar
    que Otsu reciba una imagen uint8 con el rango correcto [1, 254],
    equivalente al pipeline validado experimentalmente.

    Args:
        img_norm: Array float32 normalizado en [0.0, 1.0].

    Returns:
        Array float32 con región pulmonar aislada, fondo en negro.
    """
    # Recuperar uint8 desde float32
    img_u8 = (img_norm * 255).astype(np.uint8)

    # Replicar windowing simulado del pipeline de referencia
    lower  = 25
    upper  = 175
    img_u8 = np.clip(img_u8, lower, upper)
    img_u8 = ((img_u8 - lower) / (upper - lower) * 255).astype(np.uint8)

    # Replicar filtros del pipeline de referencia sobre uint8
    f1    = cv2.medianBlur(img_u8, 3)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))
    f2    = clahe.apply(f1)
    f3    = cv2.bilateralFilter(f2, 3, 15, 15)
    img_u8 = cv2.fastNlMeansDenoising(
        f3, None, h=4, templateWindowSize=7, searchWindowSize=21
    )

    # Segmentación con Otsu sobre imagen uint8 correcta
    _, mascara = cv2.threshold(
        img_u8, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel  = np.ones((3, 3), np.uint8)
    mascara = cv2.dilate(mascara, kernel, iterations=3)
    mascara = ndimage.binary_fill_holes(mascara).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mascara, connectivity=4
    )

    nueva_mascara = np.zeros_like(mascara)
    if num_labels > 1:
        areas   = stats[1:, cv2.CC_STAT_AREA]
        indices = np.argsort(areas)[::-1] + 1
        count   = 0
        for idx in indices:
            x, y, w, h, area = stats[idx]
            if not (x == 0 or y == 0) and area > (img_u8.size * 0.0005):
                nueva_mascara[labels == idx] = 255
                count += 1
            if count == 3:
                break

    result = cv2.bitwise_and(img_u8, img_u8, mask=nueva_mascara)
    return result.astype(np.float32) / 255.0


def apply_filters(
    img_norm: np.ndarray,
    noise_profile: Dict[str, Union[float, bool]]
) -> Optional[np.ndarray]:
    output = img_norm.copy()

    if noise_profile["has_sp"]:
        logger.info("Aplicando filtro de mediana (sal y pimienta detectado)")
        output = apply_median_filter(output)

    output = apply_clahe(output)

    if noise_profile["is_white"] and noise_profile["snr_db"] < CONFIG.SNR_LIMIT:
        logger.info("Aplicando NLM (ruido blanco con SNR degradado)")
        output = apply_nlm_filter(output)

    result = apply_morphological_lung_mask(output)
    if result is None:
        return None

    return result

# ──────────────────────────────────────────────────────────────────────────────
# Experimento de filtros y SNR
# ──────────────────────────────────────────────────────────────────────────────

def _compute_snr(img_norm: np.ndarray) -> float:
    """
    Calcula SNR en dB de una imagen normalizada.

    Args:
        img_norm: Array float32 normalizado en [0.0, 1.0].

    Returns:
        SNR en decibeles.
    """
    low_pass = gaussian_filter(img_norm, sigma=2.0)
    noise    = img_norm - low_pass
    return float(10 * np.log10(np.var(low_pass) / (np.var(noise) + 1e-12)))


def _normalize(img: np.ndarray) -> np.ndarray:
    """
    Normaliza imagen uint8 a float32 en [0.0, 1.0].

    Args:
        img: Array uint8.

    Returns:
        Array float32 normalizado.
    """
    img_float = img.astype(np.float32)
    return (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)


def _get_random_image(clase: str) -> np.ndarray:
    """
    Carga una imagen aleatoria de la clase especificada.

    Args:
        clase: Nombre de la clase (COVID-19 o Non-COVID-19).

    Returns:
        Array uint8 de la imagen en escala de grises.

    Raises:
        FileNotFoundError: Si la carpeta de la clase no existe.
        ValueError: Si no hay imágenes válidas en la carpeta.
    """
    clase_dir = RAW_DIR / clase
    if not clase_dir.exists():
        raise FileNotFoundError(f"Carpeta no encontrada: {clase_dir}")

    image_files = [
        f for f in clase_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    ]

    if not image_files:
        raise ValueError(f"No hay imágenes en: {clase_dir}")

    rng    = np.random.default_rng(seed=42)
    sample = image_files[rng.integers(0, len(image_files))]
    img    = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"No se pudo leer: {sample}")

    logger.info(f"Imagen seleccionada [{clase}]: {sample.name}")
    return img


def run_filter_experiment() -> None:
    """
    Experimento visual mostrando cada etapa del pipeline por clase.
    Visualiza: Original → Windowing → Filtrada → Máscara → Segmentada
    """
    n_clases = len(CLASES)
    fig, axes = plt.subplots(n_clases, 5, figsize=(22, n_clases * 5))
    fig.suptitle(
        "Pipeline de Preprocesamiento CT por Clase\n"
        "Original → Windowing → Filtrada → Máscara → Segmentada",
        fontsize=12, fontweight="bold"
    )

    titulos = ["Original", "Windowing", "Filtrada", "Máscara", "Segmentada"]
    rng = np.random.default_rng(seed=2000)

    for row, clase in enumerate(CLASES):
        clase_dir = RAW_DIR / clase
        if not clase_dir.exists():
            logger.error(f"Carpeta no encontrada: {clase_dir}")
            continue

        image_files = [
            f for f in clase_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXT
        ]
        sample_path = image_files[rng.integers(0, len(image_files))]
        img_raw = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)

        if img_raw is None:
            logger.error(f"No se pudo leer: {sample_path}")
            continue

        logger.info(f"Imagen seleccionada [{clase}]: {sample_path.name}")

        # Etapa 1: Windowing simulado → uint8
        lower = 25
        upper = 175
        img_win = np.clip(img_raw, lower, upper)
        img_win = ((img_win - lower) / (upper - lower) * 255).astype(np.uint8)

        # Etapa 2: Filtros → uint8
        f1 = cv2.medianBlur(img_win, 3)
        clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))
        f2 = clahe.apply(f1)
        f3 = cv2.bilateralFilter(f2, 3, 15, 15)
        img_filtrada = cv2.fastNlMeansDenoising(
            f3, None, h=4, templateWindowSize=7, searchWindowSize=21
        )

        # Etapa 3: Máscara morfológica
        _, mascara = cv2.threshold(
            img_filtrada, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel  = np.ones((3, 3), np.uint8)
        mascara = cv2.dilate(mascara, kernel, iterations=3)
        mascara = ndimage.binary_fill_holes(mascara).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mascara, connectivity=4
        )
        nueva_mascara = np.zeros_like(mascara)
        if num_labels > 1:
            areas   = stats[1:, cv2.CC_STAT_AREA]
            indices = np.argsort(areas)[::-1] + 1
            count   = 0
            for idx in indices:
                x, y, w, h, area = stats[idx]
                if not (x == 0 or y == 0) and area > (img_filtrada.size * 0.0005):
                    nueva_mascara[labels == idx] = 255
                    count += 1
                if count == 3:
                    break

        # Etapa 4: Segmentada
        img_segmentada = cv2.bitwise_and(
            img_filtrada, img_filtrada, mask=nueva_mascara
        )

        imgs = [img_raw, img_win, img_filtrada, nueva_mascara, img_segmentada]

        for col, (titulo, img) in enumerate(zip(titulos, imgs)):
            ax = axes[row, col] if n_clases > 1 else axes[col]
            ax.imshow(img, cmap="gray")
            ax.set_title(titulo, fontsize=10)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(clase, fontsize=10)

    plt.tight_layout()
    plt.savefig("experimento_filtros_visual.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Gráfico guardado: experimento_filtros_visual.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_filter_experiment()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    run_filter_experiment()