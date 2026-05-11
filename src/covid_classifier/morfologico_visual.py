"""
Visualización paso a paso del operador morfológico.

Muestra el efecto de cada etapa de segmentación sobre una imagen
aleatoria del dataset para sustentar el proceso ante el profesor.

Uso:
    python -m src.morfologico_visual
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

logger = logging.getLogger(__name__)

RAW_DIR       = Path("./data/raw")
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def visualizar_pasos_morfologicos(clase: str = "Non-COVID-19", seed: int = 100) -> None:
    """
    Visualiza cada paso del operador morfológico sobre una imagen aleatoria.

    Pasos mostrados:
        1. Original
        2. Después de windowing + filtros (entrada al morfológico)
        3. Después de umbralización Otsu inversa
        4. Después de dilatación morfológica
        5. Después de relleno de huecos
        6. Segmentación final (máscara aplicada)

    Args:
        clase: Clase del dataset a usar. Default "Non-COVID-19".
        seed: Semilla para selección aleatoria de imagen.
    """
    # ── Cargar imagen aleatoria ───────────────────────────────────────────
    clase_dir = RAW_DIR / clase
    if not clase_dir.exists():
        raise FileNotFoundError(f"Carpeta no encontrada: {clase_dir}")

    image_files = [
        f for f in clase_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    ]

    rng        = np.random.default_rng(seed=seed)
    indices    = rng.permutation(len(image_files))
    sample     = image_files[indices[700]]
    img_raw    = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)

    if img_raw is None:
        raise ValueError(f"No se pudo leer: {sample}")

    logger.info(f"Imagen seleccionada [{clase}]: {sample.name}")

    # ── Paso 1: Windowing simulado ────────────────────────────────────────
    lower   = 25
    upper   = 175
    img_win = np.clip(img_raw, lower, upper)
    img_win = ((img_win - lower) / (upper - lower) * 255).astype(np.uint8)

    # ── Paso 2: Filtros → entrada al morfológico ──────────────────────────
    f1    = cv2.medianBlur(img_win, 3)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))
    f2    = clahe.apply(f1)
    f3    = cv2.bilateralFilter(f2, 3, 15, 15)
    img_filtrada = cv2.fastNlMeansDenoising(
        f3, None, h=4, templateWindowSize=7, searchWindowSize=21
    )

    # ── Paso 3: Umbralización inversa de Otsu ─────────────────────────────
    _, mascara_otsu = cv2.threshold(
        img_filtrada, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Paso 4: Dilatación morfológica ────────────────────────────────────
    kernel           = np.ones((3, 3), np.uint8)
    mascara_dilatada = cv2.dilate(mascara_otsu, kernel, iterations=3)

    # ── Paso 5: Relleno de huecos ─────────────────────────────────────────
    mascara_llena = ndimage.binary_fill_holes(
        mascara_dilatada
    ).astype(np.uint8) * 255

    # ── Paso 6: Componentes conectados + selección ────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mascara_llena, connectivity=4
    )

    nueva_mascara = np.zeros_like(mascara_llena)
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

    img_segmentada = cv2.bitwise_and(
        img_filtrada, img_filtrada, mask=nueva_mascara
    )

    # ── Verificar cobertura ───────────────────────────────────────────────
    coverage = np.sum(nueva_mascara > 0) / nueva_mascara.size * 100
    logger.info(f"Cobertura pulmonar: {coverage:.1f}%")

    # ── Visualización ─────────────────────────────────────────────────────
    pasos = [
        (img_raw,          "Paso 1\nOriginal",
         "Imagen CT sin procesar.\nRango de intensidades original del escáner."),

        (img_filtrada,     "Paso 2\nFiltrada (entrada al morfológico)",
         "Después de windowing + mediana +\nCLAHE + bilateral + NLM."),

        (mascara_otsu,     "Paso 3\nUmbralización Otsu inversa",
         "Otsu detecta automáticamente. THRESH_BINARY_INV resalta\nlas zonas oscuras (pulmón)."),

        (mascara_dilatada, "Paso 4\nDilatación morfológica (3 iter.)",
         "Expande regiones para conectar"),

        (mascara_llena,    "Paso 5\nRelleno de huecos",
         "binary_fill_holes rellena bronquios\ny vasos que quedaron como agujeros\ndentro del pulmón."),

        (img_segmentada,   "Paso 6\nSegmentación final",
         f"Máscara aplicada sobre imagen filtrada.\nSolo parénquima pulmonar visible.\nCobertura: {coverage:.1f}% de la imagen."),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Operador Morfológico — Pasos de Segmentación Pulmonar\n"
        f"Clase: {clase} | Imagen: {sample.name}",
        fontsize=13, fontweight="bold"
    )

    axes_flat = axes.flatten()

    for i, (img, titulo, descripcion) in enumerate(pasos):
        ax = axes_flat[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(titulo, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(descripcion, fontsize=8, labelpad=6)
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#AAAAAA")
            spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(
        "morfologico_pasos.png",
        dpi=150, bbox_inches="tight"
    )
    plt.show()
    logger.info("Gráfico guardado: morfologico_pasos.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Cambia clase y seed hasta encontrar una imagen
    # donde el pulmón sea claramente visible
    visualizar_pasos_morfologicos(clase="Non-COVID-19", seed=100)