"""
Módulo de detección y medición de ruido en imágenes CT.

Responsabilidad única: medir características del ruido y retornar
un perfil tipado. No decide ni filtra — solo mide.
"""

import logging
from typing import Dict, Union

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import welch

from src.config import CONFIG

logger = logging.getLogger(__name__)


def _calculate_sfm(noise_row: np.ndarray) -> float:
    """
    Calcula Spectral Flatness Measure sobre una fila de ruido.

    Compara media geométrica vs aritmética de la PSD estimada por Welch.
    SFM → 1 indica espectro plano (ruido blanco).
    SFM → 0 indica espectro con picos (señal estructurada).

    Args:
        noise_row: Vector 1D con la componente de ruido de la fila central.

    Returns:
        Valor SFM en el rango [0, 1].
    """
    _, psd = welch(noise_row, fs=1.0, nperseg=64)
    psd_pos = psd[psd > 0]
    if psd_pos.size == 0:
        return 0.0
    return float(
        np.exp(np.mean(np.log(psd_pos + 1e-12))) / (np.mean(psd_pos) + 1e-12)
    )


def _calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calcula SNR en decibeles como varianza de señal sobre varianza de ruido.

    Justificación del método:
        La varianza es una medida de energía apropiada para señales de imagen.
        El filtro gaussiano sigma=2.0 separa frecuencias espaciales bajas
        (estructura anatómica) de altas (ruido térmico del detector CT).

    Args:
        signal: Componente de baja frecuencia (señal útil).
        noise: Componente de alta frecuencia (ruido estimado).

    Returns:
        SNR en dB. Valores < 10 dB indican imagen no apta para diagnóstico.
    """
    return float(
        10 * np.log10(np.var(signal) / (np.var(noise) + 1e-12))
    )


def _calculate_sp_density(img_norm: np.ndarray) -> float:
    """
    Calcula la densidad de píxeles outliers extremos (ruido sal y pimienta).

    Args:
        img_norm: Imagen normalizada en [0.0, 1.0].

    Returns:
        Proporción de píxeles en los extremos del rango [0.0, 0.001] ∪ [0.999, 1.0].
    """
    return float(
        np.sum((img_norm < 0.001) | (img_norm > 0.999)) / img_norm.size
    )


def analyze_noise(img_norm: np.ndarray) -> Dict[str, Union[float, bool]]:
    """
    Analiza el perfil de ruido completo de una imagen CT normalizada.

    Combina análisis espectral (SFM) con análisis energético (SNR) y
    detección estadística de outliers (densidad sal y pimienta).

    Args:
        img_norm: Array float32 normalizado en [0.0, 1.0].

    Returns:
        Diccionario con el perfil de ruido:
            - 'snr_db' (float): Relación señal-ruido en dB.
            - 'sfm' (float): Spectral Flatness Measure [0, 1].
            - 'sp_density' (float): Densidad de píxeles outliers.
            - 'is_white' (bool): True si el ruido es predominantemente blanco.
            - 'has_sp' (bool): True si hay ruido impulsivo sal y pimienta.
            - 'low_quality' (bool): True si la imagen debe ser descartada.
    """
    low_pass = gaussian_filter(img_norm, sigma=2.0)
    noise = img_norm - low_pass

    mid_row = img_norm.shape[0] // 2
    sfm = _calculate_sfm(noise[mid_row, :])
    snr_db = _calculate_snr(low_pass, noise)
    sp_density = _calculate_sp_density(img_norm)

    return {
        "snr_db": snr_db,
        "sfm": sfm,
        "sp_density": sp_density,
        "is_white": sfm > CONFIG.SFM_LIMIT,
        "has_sp": sp_density > CONFIG.SP_DENSITY_LIMIT,
        "low_quality": snr_db < CONFIG.SNR_DISCARD,
    }