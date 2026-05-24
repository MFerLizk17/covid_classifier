"""
Orquestador principal del pipeline de preprocesamiento CT.

Responsabilidad única: coordinar las etapas del pipeline delegando
cada responsabilidad a su módulo correspondiente.

Pipeline aplicado:
    Carga → CLAHE → NLM → Binarización adaptativa → Flood-fill
    → Componentes conexas → Morfología → Máscara pulmonar → Resize
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage import img_as_float
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.restoration import denoise_nl_means, estimate_sigma

from config import CONFIG

logger = logging.getLogger(__name__)

# ── Códigos de descarte ───────────────────────────────────────────
DESCARTE_CALIDAD = "calidad"
DESCARTE_MASCARA = "mascara"

# ── Parámetros CLAHE ──────────────────────────────────────────────
CLIP_LIMIT = 2.5
TILE_SIZE  = (4, 4)

# ── Parámetros NLM ────────────────────────────────────────────────
H_FACTOR   = 1.3
PATCH_SIZE = 7
PATCH_DIST = 13

# ── Parámetros binarización ───────────────────────────────────────
PX_THRESHOLD = 80

# ── Parámetros morfología ─────────────────────────────────────────
EROSION_RADIUS  = 1
DILATION_RADIUS = 3


def _load_image(image_path: Path) -> np.ndarray:
    """
    Carga imagen y convierte a escala de grises.

    Maneja formatos L, RGB y RGBA correctamente.

    Args:
        image_path: Ruta al archivo de imagen.

    Returns:
        Array numpy uint8 en escala de grises.

    Raises:
        ValueError: Si el formato de imagen no es soportado.
    """
    img_raw = np.array(Image.open(image_path))

    if img_raw.ndim == 2:
        return img_raw.astype(np.uint8)
    elif img_raw.ndim == 3 and img_raw.shape[2] == 4:
        return cv2.cvtColor(img_raw, cv2.COLOR_RGBA2GRAY)
    elif img_raw.ndim == 3 and img_raw.shape[2] == 3:
        return cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Formato no soportado: shape={img_raw.shape}")


def _apply_clahe(img_gray: np.ndarray) -> np.ndarray:
    """
    Aplica ecualización adaptativa de histograma local (CLAHE).

    Mejora el contraste local sin saturar zonas brillantes,
    preservando las diferencias de intensidad diagnósticamente
    relevantes entre tejido pulmonar y consolidaciones.

    Args:
        img_gray: Imagen en escala de grises uint8.

    Returns:
        Imagen con contraste mejorado uint8.
    """
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
    return clahe.apply(img_gray)


def _apply_nlm(img_clahe: np.ndarray) -> np.ndarray:
    """
    Aplica filtro Non-Local Means para reducción de ruido gaussiano.

    Estima la sigma del ruido automáticamente antes de filtrar,
    adaptando la intensidad del filtro a cada imagen individual.

    Args:
        img_clahe: Imagen post-CLAHE uint8.

    Returns:
        Imagen filtrada como array float64 en [0.0, 1.0].
    """
    img_float = img_as_float(img_clahe)
    sigma_est = float(estimate_sigma(img_float, average_sigmas=True))
    return denoise_nl_means(
        img_float,
        h              = H_FACTOR * sigma_est,
        patch_size     = PATCH_SIZE,
        patch_distance = PATCH_DIST,
        fast_mode      = True,
    )


def _binarize(img_nlm: np.ndarray) -> np.ndarray:
    """
    Binariza la imagen usando umbral adaptativo según tipo de fondo.

    Detecta si el fondo es oscuro (CT estándar con FOV negro) o claro
    analizando las esquinas de 20×20 px. Si el fondo es oscuro usa
    umbral fijo; si es claro usa Otsu invertido.

    Args:
        img_nlm: Imagen filtrada float64 en [0.0, 1.0].

    Returns:
        Imagen binaria uint8 donde 1 = tejido, 0 = fondo.
    """
    img_u8 = (img_nlm * 255).astype(np.uint8)
    h, w   = img_u8.shape

    corners = np.concatenate([
        img_u8[:20, :20].ravel(),
        img_u8[:20, w - 20:].ravel(),
        img_u8[h - 20:, :20].ravel(),
        img_u8[h - 20:, w - 20:].ravel(),
    ])

    if corners.mean() < 30:
        return (img_u8 < PX_THRESHOLD).astype(np.uint8)

    thresh = threshold_otsu(img_u8)
    return (img_u8 < thresh).astype(np.uint8)


def _remove_exterior_air(binary: np.ndarray) -> np.ndarray:
    """
    Elimina regiones conectadas al borde (aire exterior y fondo).

    Aplica flood-fill desde todos los píxeles del borde para
    eliminar el aire exterior que rodea al paciente, dejando
    solo las estructuras internas.

    Args:
        binary: Imagen binaria uint8.

    Returns:
        Imagen binaria sin regiones conectadas al borde.
    """
    h, w = binary.shape
    temp = binary.copy().astype(np.uint8)

    for c in range(w):
        if temp[0, c] == 1:
            cv2.floodFill(temp, None, (c, 0), 0)
        if temp[h - 1, c] == 1:
            cv2.floodFill(temp, None, (c, h - 1), 0)
    for r in range(h):
        if temp[r, 0] == 1:
            cv2.floodFill(temp, None, (0, r), 0)
        if temp[r, w - 1] == 1:
            cv2.floodFill(temp, None, (w - 1, r), 0)

    return temp


def _select_lung_components(binary_clean: np.ndarray) -> np.ndarray:
    """
    Conserva las dos componentes conexas más grandes que corresponden a pulmones.

    Filtra por área relativa: entre 0.5% y 35% del total de la imagen,
    para descartar ruido pequeño y estructuras no pulmonares grandes.

    Args:
        binary_clean: Imagen binaria sin aire exterior.

    Returns:
        Máscara binaria con solo los componentes pulmonares.
    """
    labeled   = label(binary_clean)
    regions   = regionprops(labeled)
    h, w      = binary_clean.shape
    img_area  = h * w

    candidates = [
        r for r in regions
        if img_area * 0.005 < r.area < img_area * 0.35
    ]
    candidates = sorted(candidates, key=lambda r: r.area, reverse=True)[:2]

    lung_mask = np.zeros_like(binary_clean)
    for reg in candidates:
        lung_mask[labeled == reg.label] = 1

    return lung_mask


def _apply_morphology(lung_mask: np.ndarray) -> np.ndarray:
    """
    Aplica operaciones morfológicas para refinar la máscara pulmonar.

    Secuencia:
        1. Closing grande → cierra huecos internos (vasos, nódulos, fisuras)
        2. Fill holes → rellena huecos residuales
        3. Erosión suave → recupera tamaño original aproximado
        4. Dilatación leve → suaviza bordes
        5. Fill final → garantiza máscara completamente sólida

    Args:
        lung_mask: Máscara pulmonar binaria sin refinar.

    Returns:
        Máscara pulmonar refinada uint8.
    """
    selem_close  = disk(10)
    lung_closed  = binary_dilation(lung_mask, selem_close).astype(np.uint8)
    lung_closed  = binary_erosion(lung_closed, selem_close).astype(np.uint8)

    lung_filled  = binary_fill_holes(lung_closed).astype(np.uint8)

    selem_e      = disk(EROSION_RADIUS)
    mask_eroded  = binary_erosion(lung_filled, selem_e).astype(np.uint8)

    selem_d      = disk(DILATION_RADIUS)
    mask_dilated = binary_dilation(mask_eroded, selem_d).astype(np.uint8)

    return binary_fill_holes(mask_dilated).astype(np.uint8)


def _apply_mask(img_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Aplica la máscara pulmonar sobre la imagen filtrada.

    Los píxeles fuera de la máscara se ponen a cero (fondo negro),
    dejando solo el tejido pulmonar para el modelo.

    Args:
        img_gray: Imagen filtrada uint8.
        mask: Máscara binaria final uint8.

    Returns:
        Imagen enmascarada como float32 en [0.0, 1.0].
    """
    img_masked              = img_gray.astype(np.float32).copy()
    img_masked[mask == 0]   = 0.0
    return img_masked / 255.0


def _resize_and_pad(img: np.ndarray) -> np.ndarray:
    """
    Redimensiona preservando proporción anatómica mediante padding.

    Calcula el ratio mínimo para que la imagen quepa en target_size
    y rellena con ceros simétricamente para no distorsionar estructuras.

    Args:
        img: Imagen float32 en [0.0, 1.0].

    Returns:
        Imagen redimensionada float32 en [0.0, 1.0] de tamaño target_size.
    """
    h, w             = img.shape[:2]
    target_h, target_w = CONFIG.target_size

    ratio  = min(target_w / w, target_h / h)
    new_w  = int(w * ratio)
    new_h  = int(h * ratio)

    img_u8  = (img * 255).astype(np.uint8)
    resized = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top     = delta_h // 2
    bottom  = delta_h - top
    left    = delta_w // 2
    right   = delta_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0,
    )
    return padded.astype(np.float32) / 255.0


def run_pipeline(
    image_path: Path,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Ejecuta el pipeline completo de preprocesamiento sobre una imagen.

    Etapas:
        1. Carga y conversión a escala de grises
        2. CLAHE — mejora de contraste adaptativo
        3. NLM   — reducción de ruido gaussiano
        4. Binarización adaptativa según tipo de fondo
        5. Flood-fill — eliminación de aire exterior
        6. Selección de componentes pulmonares
        7. Morfología — refinamiento de máscara
        8. Aplicación de máscara sobre imagen filtrada
        9. Resize con padding proporcional

    Args:
        image_path: Ruta al archivo de imagen a procesar.

    Returns:
        Tupla (imagen_procesada, motivo_descarte):
            - Si éxito: (array float32 [0,1] de tamaño target_size, None)
            - Si descarte por máscara vacía: (None, 'mascara')

    Raises:
        FileNotFoundError: Si image_path no existe.
        ValueError: Si el formato de imagen no es soportado.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Ruta no encontrada: {image_path}")

    img_gray  = _load_image(image_path)
    img_clahe = _apply_clahe(img_gray)
    img_nlm   = _apply_nlm(img_clahe)
    binary    = _binarize(img_nlm)

    binary_clean = _remove_exterior_air(binary)
    lung_mask    = _select_lung_components(binary_clean)

    if lung_mask.sum() == 0:
        logger.warning(
            f"Máscara pulmonar vacía, imagen descartada: {image_path.name}"
        )
        return None, DESCARTE_MASCARA

    final_mask = _apply_morphology(lung_mask)
    img_nlm_u8 = (img_nlm * 255).astype(np.uint8)
    img_masked = _apply_mask(img_nlm_u8, final_mask)
    resultado  = _resize_and_pad(img_masked)

    return resultado, None