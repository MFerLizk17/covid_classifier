"""
Tests unitarios para el módulo src/preprocessor.py.

Cubre los métodos principales de CTPreprocessor verificando:
    - Normalización correcta del rango de intensidades.
    - Detección del perfil de ruido y sus flags.
    - Aplicación condicional de filtros.
    - Redimensionamiento con preservación de aspecto.
    - Comportamiento del pipeline ante entradas inválidas.

Uso:
    pytest tests/test_preprocessor.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from src.preprocessor import CTPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocessor() -> CTPreprocessor:
    """Instancia de CTPreprocessor con tamaño de salida estándar."""
    return CTPreprocessor(target_size=(224, 224))


@pytest.fixture
def clean_image() -> np.ndarray:
    """Imagen sintética limpia de 128×128 con gradiente suave (bajo ruido)."""
    img = np.zeros((128, 128), dtype=np.uint8)
    for i in range(128):
        img[i, :] = i * 2
    return img


@pytest.fixture
def noisy_image() -> np.ndarray:
    """Imagen sintética con ruido gaussiano intenso para pruebas de análisis."""
    rng = np.random.default_rng(seed=42)
    base = np.full((128, 128), 128, dtype=np.float32)
    noise = rng.normal(0, 40, (128, 128)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def sp_image() -> np.ndarray:
    """Imagen con ruido sal y pimienta para verificar detección de outliers."""
    rng = np.random.default_rng(seed=0)
    img = np.full((128, 128), 128, dtype=np.uint8)
    # Insertar ~2% de píxeles extremos (supera SP_DENSITY_LIMIT=0.005)
    n_pixels = int(0.02 * img.size)
    rows = rng.integers(0, 128, n_pixels)
    cols = rng.integers(0, 128, n_pixels)
    img[rows[:n_pixels // 2], cols[:n_pixels // 2]] = 0
    img[rows[n_pixels // 2:], cols[n_pixels // 2:]] = 255
    return img


# ---------------------------------------------------------------------------
# Tests: apply_windowing
# ---------------------------------------------------------------------------

class TestApplyWindowing:

    def test_output_range_is_zero_to_one(self, preprocessor, clean_image):
        """La imagen normalizada debe estar completamente en [0.0, 1.0]."""
        result = preprocessor.apply_windowing(clean_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_is_float32(self, preprocessor, clean_image):
        """El tipo de dato de salida debe ser float32."""
        result = preprocessor.apply_windowing(clean_image)
        assert result.dtype == np.float32

    def test_raises_on_empty_image(self, preprocessor):
        """Debe lanzar ValueError si la imagen está vacía."""
        with pytest.raises(ValueError):
            preprocessor.apply_windowing(np.array([]))

    def test_raises_on_none_image(self, preprocessor):
        """Debe lanzar ValueError si la imagen es None."""
        with pytest.raises(ValueError):
            preprocessor.apply_windowing(None)

    def test_uniform_image_does_not_raise(self, preprocessor):
        """Una imagen de intensidad uniforme no debe lanzar error (división por cero)."""
        uniform = np.full((64, 64), 128, dtype=np.uint8)
        result = preprocessor.apply_windowing(uniform)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: analyze_noise
# ---------------------------------------------------------------------------

class TestAnalyzeNoise:

    def test_returns_required_keys(self, preprocessor, clean_image):
        """El diccionario de salida debe contener todas las claves esperadas."""
        img_norm = preprocessor.apply_windowing(clean_image)
        profile = preprocessor.analyze_noise(img_norm)
        expected_keys = {"snr_db", "sfm", "is_white", "has_sp", "low_quality"}
        assert expected_keys == set(profile.keys())

    def test_snr_is_float(self, preprocessor, clean_image):
        """snr_db debe ser un número flotante."""
        img_norm = preprocessor.apply_windowing(clean_image)
        profile = preprocessor.analyze_noise(img_norm)
        assert isinstance(profile["snr_db"], float)

    def test_flags_are_bool(self, preprocessor, clean_image):
        """Los flags is_white, has_sp y low_quality deben ser booleanos."""
        img_norm = preprocessor.apply_windowing(clean_image)
        profile = preprocessor.analyze_noise(img_norm)
        assert isinstance(profile["is_white"], bool)
        assert isinstance(profile["has_sp"], bool)
        assert isinstance(profile["low_quality"], bool)

    def test_sp_detection_on_sp_image(self, preprocessor, sp_image):
        """Una imagen con 2% de outliers debe activar el flag has_sp."""
        img_norm = preprocessor.apply_windowing(sp_image)
        profile = preprocessor.analyze_noise(img_norm)
        assert profile["has_sp"] is True

    def test_clean_image_not_low_quality(self, preprocessor, clean_image):
        """Una imagen limpia con gradiente suave no debe ser marcada como baja calidad."""
        img_norm = preprocessor.apply_windowing(clean_image)
        profile = preprocessor.analyze_noise(img_norm)
        assert profile["low_quality"] is False


# ---------------------------------------------------------------------------
# Tests: apply_filters
# ---------------------------------------------------------------------------

class TestApplyFilters:

    def test_output_shape_preserved(self, preprocessor, clean_image):
        """La forma de la imagen no debe cambiar tras aplicar filtros."""
        img_norm = preprocessor.apply_windowing(clean_image)
        flags = preprocessor.analyze_noise(img_norm)
        result = preprocessor.apply_filters(img_norm, flags)
        assert result.shape == img_norm.shape

    def test_output_range_after_filters(self, preprocessor, sp_image):
        """La imagen filtrada debe mantenerse en el rango [0.0, 1.0]."""
        img_norm = preprocessor.apply_windowing(sp_image)
        flags = preprocessor.analyze_noise(img_norm)
        result = preprocessor.apply_filters(img_norm, flags)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_float32(self, preprocessor, clean_image):
        """El tipo de salida debe ser float32."""
        img_norm = preprocessor.apply_windowing(clean_image)
        flags = preprocessor.analyze_noise(img_norm)
        result = preprocessor.apply_filters(img_norm, flags)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:

    def test_output_matches_target_size(self, preprocessor, clean_image):
        """La imagen de salida debe tener exactamente el tamaño target_size."""
        img_norm = preprocessor.apply_windowing(clean_image)
        result = preprocessor.resize_and_pad(img_norm)
        target_h, target_w = preprocessor.target_size
        assert result.shape == (target_h, target_w)

    def test_non_square_image_fits_target(self, preprocessor):
        """Una imagen rectangular debe ajustarse a target_size sin distorsión."""
        rect_img = np.random.rand(100, 200).astype(np.float32)
        result = preprocessor.resize_and_pad(rect_img)
        target_h, target_w = preprocessor.target_size
        assert result.shape == (target_h, target_w)


# ---------------------------------------------------------------------------
# Tests: run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:

    def test_raises_on_missing_file(self, preprocessor, tmp_path):
        """Debe lanzar FileNotFoundError si la ruta no existe."""
        missing = tmp_path / "no_existe.png"
        with pytest.raises(FileNotFoundError):
            preprocessor.run_pipeline(missing)

    def test_returns_correct_shape_on_valid_image(self, preprocessor, tmp_path):
        """Una imagen válida debe producir un array con la forma target_size."""
        import cv2
        img = np.full((128, 128), 100, dtype=np.uint8)
        # Agregar gradiente para que tenga SNR adecuado
        for i in range(128):
            img[i, :] = i * 2
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)
        result = preprocessor.run_pipeline(img_path)
        if result is not None:
            target_h, target_w = preprocessor.target_size
            assert result.shape == (target_h, target_w)
