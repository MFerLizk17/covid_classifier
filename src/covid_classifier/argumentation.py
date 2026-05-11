"""
Módulo de aumento de datos para imágenes CT.

Solo se aplica durante entrenamiento a la clase minoritaria (Non-COVID-19).
Nunca en validación ni en test para evitar fuga de información.

Justificación de transformaciones seleccionadas:
    - Rotación ±15°: válida clínicamente, los pulmones pueden aparecer
      en posiciones ligeramente distintas entre pacientes.
    - Flip horizontal: anatómicamente válido para pulmones.
    - Zoom ±10%: simula variabilidad en posicionamiento del paciente.
    - Ruido gaussiano suave: simula variabilidad entre equipos CT.

Transformaciones excluidas:
    - Flip vertical: destruye la orientación anatómica superior/inferior.
    - Rotaciones >15°: distorsionan la arquitectura pulmonar aprendida.
    - Cambios de brillo: la normalización min-max ya estandariza intensidades.
"""

import numpy as np
import cv2
from src.config import CONFIG


def random_rotation(img: np.ndarray) -> np.ndarray:
    """
    Aplica rotación aleatoria dentro del rango clínicamente válido.

    Args:
        img: Array float32 (224, 224) normalizado en [0.0, 1.0].

    Returns:
        Array float32 rotado, misma dimensión.
    """
    rng = np.random.default_rng(CONFIG.AUG_SEED)
    angle = rng.uniform(-CONFIG.ROTATION_RANGE, CONFIG.ROTATION_RANGE)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def random_flip(img: np.ndarray) -> np.ndarray:
    """
    Aplica flip horizontal con probabilidad 50%.

    Args:
        img: Array float32 (224, 224) normalizado en [0.0, 1.0].

    Returns:
        Array float32, con o sin flip horizontal.
    """
    rng = np.random.default_rng(CONFIG.AUG_SEED)
    if rng.random() > 0.5:
        return cv2.flip(img, 1)
    return img


def random_zoom(img: np.ndarray) -> np.ndarray:
    """
    Aplica zoom aleatorio recortando y redimensionando de vuelta a 224×224.

    Args:
        img: Array float32 (224, 224) normalizado en [0.0, 1.0].

    Returns:
        Array float32 (224, 224) con zoom aplicado.
    """
    rng = np.random.default_rng(CONFIG.AUG_SEED)
    zoom = rng.uniform(1.0 - CONFIG.ZOOM_RANGE, 1.0 + CONFIG.ZOOM_RANGE)
    h, w = img.shape[:2]
    new_h = int(h * zoom)
    new_w = int(w * zoom)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if zoom > 1.0:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return resized[start_h:start_h + h, start_w:start_w + w]
    else:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        return cv2.copyMakeBorder(
            resized,
            pad_h, h - new_h - pad_h,
            pad_w, w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=0
        )


def add_gaussian_noise(img: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """
    Agrega ruido gaussiano suave para simular variabilidad entre equipos CT.

    Args:
        img: Array float32 (224, 224) normalizado en [0.0, 1.0].
        sigma: Desviación estándar del ruido. Default 0.01 (1% de rango).

    Returns:
        Array float32 con ruido añadido, clippeado a [0.0, 1.0].
    """
    rng = np.random.default_rng(CONFIG.AUG_SEED)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def augment(img: np.ndarray) -> np.ndarray:
    """
    Aplica pipeline completo de augmentation sobre una imagen.

    Solo debe llamarse durante entrenamiento sobre la clase minoritaria.

    Args:
        img: Array float32 (224, 224) normalizado en [0.0, 1.0].

    Returns:
        Array float32 (224, 224) con transformaciones aplicadas.
    """
    img = random_rotation(img)
    img = random_flip(img)
    img = random_zoom(img)
    img = add_gaussian_noise(img)
    return img