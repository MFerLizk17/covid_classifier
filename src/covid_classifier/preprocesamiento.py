"""
Orquestador principal del pipeline de preprocesamiento CT.

Responsabilidad única: coordinar las etapas del pipeline delegando
cada responsabilidad a su módulo correspondiente. No implementa
lógica de filtrado, detección ni control de calidad.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.config import CONFIG
from src.noise_detector import analyze_noise
from src.quality_control import check_quality
from src.filter_engine import apply_filters

logger = logging.getLogger(__name__)

# Códigos de descarte
DESCARTE_CALIDAD = "calidad"
DESCARTE_MASCARA = "mascara"


def apply_windowing(
    img: np.ndarray,
    center: int = 100,
    width: int = 150
) -> np.ndarray:
    """
    Estandariza el rango dinámico mediante ventana CT simulada.
    """
    if img is None or img.size == 0:
        raise ValueError("La imagen proporcionada está vacía o es None.")

    lower      = center - (width // 2)
    upper      = center + (width // 2)
    img_clipped = np.clip(img, lower, upper)
    return ((img_clipped - lower) / (upper - lower + 1e-8)).astype(np.float32)


def resize_and_pad(img: np.ndarray) -> np.ndarray:
    """
    Redimensiona preservando proporción anatómica mediante padding.
    """
    h, w     = img.shape[:2]
    target_h, target_w = CONFIG.target_size

    ratio  = min(target_w / w, target_h / h)
    new_w  = int(w * ratio)
    new_h  = int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top     = delta_h // 2
    bottom  = delta_h - top
    left    = delta_w // 2
    right   = delta_w - left

    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0
    )


def run_pipeline(
    image_path: Path
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Ejecuta el pipeline completo de preprocesamiento sobre una imagen.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Ruta no encontrada: {image_path}")

    raw_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        logger.warning(f"No se pudo leer la imagen: {image_path.name}")
        return None, DESCARTE_MASCARA

    img_norm      = apply_windowing(raw_img)
    noise_profile = analyze_noise(img_norm)

    if not check_quality(noise_profile, image_path.name):
        return None, DESCARTE_CALIDAD

    img_filtered = apply_filters(img_norm, noise_profile)

    if img_filtered is None:
        logger.warning(
            f"Imagen descartada por máscara pulmonar vacía: {image_path.name}"
        )
        return None, DESCARTE_MASCARA

    return resize_and_pad(img_filtered), None


# --- Funciones de sustentación (Nitidez y Visualización) ---

def calcular_nitidez(img: np.ndarray) -> float:
    return float(cv2.Laplacian(img, cv2.CV_64F).var())

def generar_histograma_laplaciano(img: np.ndarray, nombre: str):
    """
    Genera el histograma del Laplaciano para visualizar la distribución de bordes.
    Un histograma más ancho indica mayor nitidez.
    """
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    plt.figure(figsize=(8, 5))
    plt.hist(laplacian.ravel(), bins=100, color='blue', alpha=0.7)
    plt.title(f"Distribución del Laplaciano: {nombre}")
    plt.xlabel("Valor del gradiente")
    plt.ylabel("Frecuencia")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"hist_{nombre}.png")
    plt.close()

class CTProcessor:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def run_eda(self):
        """Genera métricas de nitidez e histogramas para sustentar el resize."""
        res_sizes = [64,128, 224, 512]
        for clase in ["COVID-19", "Non-COVID-19"]:
            path = self.raw_dir / clase
            if not path.exists(): continue
            
            print(f"\n--- Analizando {clase} ---")
            # Tomamos una muestra
            sample = list(path.iterdir())[:5]
            for img_path in sample:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                for size in res_sizes:
                    img_r = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
                    nitidez = calcular_nitidez(img_r)
                    print(f"Resize {size}x{size} | Nitidez: {nitidez:.2f}")
                    
                    # Con este if, solo generas el histograma para el tamaño que indiques
                    # Al incluir el tamaño en el nombre, garantizas que no se sobrescriba
                    if size == 64: 
                        generar_histograma_laplaciano(img_r, f"{clase}_{img_path.stem}_size{size}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Asegúrate de ajustar esta ruta a tu estructura real
    processor = CTProcessor(Path("./data/raw"))
    processor.run_eda()