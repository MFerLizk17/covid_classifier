"""
Módulo de preprocesamiento para tomografías computarizadas (CT).

Implementa un pipeline inteligente basado en detección de ruidos específicos
identificados en el dataset de Teherán (Kaggle, Aria et al., 2021).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import welch

# --- CONFIGURACIÓN DE REGISTRO ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CT_Preprocess_System")


class CTPreprocessor:
    """
    Preprocesador de Tomografías Computarizadas (CT) para clasificación de COVID-19.

    Implementa un pipeline de 5 etapas:
        1. Normalización de rango dinámico (windowing).
        2. Análisis espectral y estadístico del ruido.
        3. Control de calidad y descarte de imágenes.
        4. Filtrado condicional según perfil de ruido.
        5. Redimensionamiento con preservación de aspecto (padding).

    Attributes:
        target_size (Tuple[int, int]): Dimensión de salida (ancho, alto) en píxeles.
        SNR_LIMIT (float): Umbral mínimo de SNR en dB para aplicar NLM.
        SFM_LIMIT (float): Umbral de planitud espectral para detectar ruido blanco.
        SP_DENSITY_LIMIT (float): Densidad mínima de outliers para detectar sal y pimienta.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Inicializa el preprocesador con los parámetros globales del pipeline.

        Args:
            target_size: Tupla (ancho, alto) para el redimensionamiento final.
                         Por defecto (224, 224), estándar para CNN en imágenes médicas.
        """
        self.target_size = target_size

        # SNR < 20 dB: el ruido comienza a degradar texturas finas (vidrio esmerilado).
        self.SNR_LIMIT: float = 20.0
        # SFM > 0.65: la señal es suficientemente plana para considerarse ruido blanco.
        self.SFM_LIMIT: float = 0.65
        # Densidad de outliers > 0.5%: indica presencia de ruido impulsivo (sal y pimienta).
        self.SP_DENSITY_LIMIT: float = 0.005

    def apply_windowing(self, img: np.ndarray) -> np.ndarray:
        """
        Estandariza el rango dinámico de la imagen (windowing).

        Normaliza los valores de píxel al rango [0.0, 1.0] mediante min-max scaling,
        simulando una ventana pulmonar sobre el dominio de intensidades disponible.

        Args:
            img: Array uint8 con la imagen original en escala de grises.

        Returns:
            Array float32 normalizado en el rango [0.0, 1.0].

        Raises:
            ValueError: Si la imagen de entrada está vacía o es None.
        """
        if img is None or img.size == 0:
            raise ValueError("La imagen proporcionada está vacía o es None.")

        img_float = img.astype(np.float32)
        min_val = np.min(img_float)
        max_val = np.max(img_float)

        # Epsilon (1e-8) evita división por cero en imágenes de intensidad uniforme.
        return (img_float - min_val) / (max_val - min_val + 1e-8)

    def analyze_noise(
        self, img_norm: np.ndarray
    ) -> Dict[str, Union[float, bool]]:
        """
        Analiza la naturaleza del ruido mediante métricas espectrales y estadísticas.

        Métricas calculadas:
            - SNR (dB): Relación señal-ruido. Valores < 10 dB indican imagen no apta.
            - SFM: Spectral Flatness Measure. Compara media geométrica vs aritmética
              de la PSD; valores cercanos a 1 indican ruido blanco (espectro plano).
            - Densidad sal y pimienta: Proporción de píxeles en los extremos del rango.

        Args:
            img_norm: Array float32 normalizado en [0.0, 1.0].

        Returns:
            Diccionario con las siguientes claves:
                - 'snr_db' (float): Relación señal-ruido en decibeles.
                - 'sfm' (float): Spectral Flatness Measure [0, 1].
                - 'is_white' (bool): True si el ruido es predominantemente blanco.
                - 'has_sp' (bool): True si hay ruido impulsivo (sal y pimienta).
                - 'low_quality' (bool): True si la imagen debe ser descartada.
        """
        # Separar señal base (paso bajo) del componente de ruido (alta frecuencia).
        low_pass = gaussian_filter(img_norm, sigma=2.0)
        noise = img_norm - low_pass

        # 1. SFM vía método de Welch sobre la fila central de la imagen.
        mid_row = img_norm.shape[0] // 2
        _, psd = welch(noise[mid_row, :], fs=1.0, nperseg=64)
        psd_pos = psd[psd > 0]
        sfm = float(
            np.exp(np.mean(np.log(psd_pos + 1e-12))) / (np.mean(psd_pos) + 1e-12)
        )

        # 2. SNR en dB: varianza de señal sobre varianza de ruido estimado.
        snr_db = float(
            10 * np.log10(np.var(low_pass) / (np.var(noise) + 1e-12))
        )

        # 3. Densidad de píxeles outliers extremos (sal y pimienta).
        sp_density = float(
            np.sum((img_norm < 0.001) | (img_norm > 0.999)) / img_norm.size
        )

        return {
            "snr_db": snr_db,
            "sfm": sfm,
            "is_white": sfm > self.SFM_LIMIT,
            "has_sp": sp_density > self.SP_DENSITY_LIMIT,
            # SNR < 10 dB: umbral de descarte crítico, imagen no apta para diagnóstico.
            "low_quality": snr_db < 10.0,
        }

    def apply_filters(
        self, img_norm: np.ndarray, flags: Dict[str, Union[float, bool]]
    ) -> np.ndarray:
        """
        Aplica filtros condicionales según el perfil de ruido detectado.

        Lógica de filtrado:
            - Filtro de mediana: elimina outliers (sal y pimienta) sin difuminar bordes.
            - Non-Local Means (NLM): promedia parches similares, ideal para ruido
              blanco térmico con SNR degradado.
            - CLAHE: estandariza el contraste local con clipLimit=2.0 para evitar
              que el ruido de fondo se amplifique como artefactos.

        Args:
            img_norm: Array float32 normalizado en [0.0, 1.0].
            flags: Diccionario generado por analyze_noise con claves
                   'has_sp', 'is_white' y 'snr_db'.

        Returns:
            Array float32 filtrado y realzado, normalizado en [0.0, 1.0].
        """
        output = img_norm.copy()

        # Paso 1: Filtro de mediana para ruido impulsivo (kernel 3×3).
        if flags["has_sp"]:
            output_u8 = (output * 255).astype(np.uint8)
            output = cv2.medianBlur(output_u8, 3).astype(np.float32) / 255.0

        # Paso 2: NLM para ruido blanco con SNR degradado.
        # h=6: equilibrio entre eliminación de grano y preservación de vasos sanguíneos.
        if flags["is_white"] and flags["snr_db"] < self.SNR_LIMIT:
            output_u8 = (output * 255).astype(np.uint8)
            output = cv2.fastNlMeansDenoising(
                output_u8, None, h=6, templateWindowSize=7, searchWindowSize=21
            ).astype(np.float32) / 255.0

        # Paso 3: CLAHE siempre aplicado para estandarizar contraste local.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        output_u8 = (output * 255).astype(np.uint8)
        output = clahe.apply(output_u8).astype(np.float32) / 255.0

        return output

    def resize_and_pad(self, img: np.ndarray) -> np.ndarray:
        """
        Redimensiona la imagen preservando la proporción original mediante padding.

        Calcula el factor de escala para que la imagen quepa en target_size sin
        distorsión anatómica, y rellena con ceros (negro) el espacio restante.

        Args:
            img: Array float32 de cualquier dimensión válida.

        Returns:
            Array float32 de dimensión exacta target_size (alto, ancho).
        """
        h, w = img.shape[:2]
        target_h, target_w = self.target_size

        ratio = min(target_w / w, target_h / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        resized = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        return cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )

    def run_pipeline(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Ejecuta el pipeline completo de preprocesamiento sobre una imagen.

        Etapas:
            1. Lectura en escala de grises.
            2. Normalización de rango dinámico (windowing).
            3. Análisis del perfil de ruido.
            4. Control de calidad (descarte si SNR < 10 dB).
            5. Filtrado y realce condicional.
            6. Redimensionamiento con padding a target_size.

        Args:
            image_path: Objeto Path apuntando al archivo de imagen.

        Returns:
            Array float32 de dimensión target_size listo para la CNN,
            o None si la imagen no supera el control de calidad.

        Raises:
            FileNotFoundError: Si la ruta especificada no existe.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Ruta no encontrada: {image_path}")

        raw_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            logger.warning(f"No se pudo leer la imagen: {image_path.name}")
            return None

        # Etapa 1: Normalización.
        img_norm = self.apply_windowing(raw_img)

        # Etapa 2: Análisis de ruido.
        noise_profile = self.analyze_noise(img_norm)
        logger.info(
            f"{image_path.name} | SNR: {noise_profile['snr_db']:.2f} dB "
            f"| SFM: {noise_profile['sfm']:.3f} "
            f"| SP: {noise_profile['has_sp']}"
        )

        # Etapa 3: Control de calidad.
        if noise_profile["low_quality"]:
            logger.warning(
                f"Imagen descartada por baja calidad "
                f"(SNR={noise_profile['snr_db']:.2f} dB): {image_path.name}"
            )
            return None

        # Etapa 4: Filtrado condicional.
        img_filtered = self.apply_filters(img_norm, noise_profile)

        # Etapa 5: Geometría final.
        return self.resize_and_pad(img_filtered)
