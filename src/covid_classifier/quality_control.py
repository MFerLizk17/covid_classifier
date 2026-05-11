"""
Módulo de control de calidad para imágenes CT.

Responsabilidad única: recibir un perfil de ruido y decidir
si la imagen es apta para entrenamiento. Separado del detector
para poder cambiar umbrales sin tocar lógica de medición.
"""

import logging
from pathlib import Path
from typing import Dict, Union

from config import CONFIG

logger = logging.getLogger(__name__)


def check_quality(
    noise_profile: Dict[str, Union[float, bool]],
    image_name: str = ""
) -> bool:
    """
    Evalúa si una imagen supera el control de calidad mínimo.

    Criterio de descarte: SNR < SNR_DISCARD (10 dB por defecto).
    Por debajo de este umbral el ruido domina sobre la señal anatómica
    y el modelo aprendería patrones incorrectos.

    Args:
        noise_profile: Diccionario retornado por analyze_noise().
        image_name: Nombre del archivo para logging (opcional).

    Returns:
        True si la imagen es apta, False si debe ser descartada.
    """
    if noise_profile["low_quality"]:
        logger.warning(
            f"Imagen descartada por baja calidad "
            f"(SNR={noise_profile['snr_db']:.2f} dB < "
            f"{CONFIG.SNR_DISCARD} dB): {image_name}"
        )
        return False

    logger.info(
        f"{image_name} | SNR: {noise_profile['snr_db']:.2f} dB "
        f"| SFM: {noise_profile['sfm']:.3f} "
        f"| SP: {noise_profile['has_sp']}"
    )
    return True