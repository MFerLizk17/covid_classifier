"""
run_pipeline.py
===============
Ejemplo completo de uso del pipeline de preprocesamiento CT pulmón COVID-19.

Flujo:
    data/raw/  →  ImageQualityFilter  →  data/valid/
    data/valid/ → CTPreprocessor      →  data/processed/
    data/processed/ → build_tf_dataset → tf.data.Dataset → CNN

Ejecutar:
    python run_pipeline.py
"""

from pathlib import Path

from image_quality_filter import ImageQualityFilter, DEFAULT_THRESHOLDS
from ct_preprocessor import CTPreprocessor, PreprocessConfig, build_tf_dataset


def main():
    # ---------------------------------------------------------------
    # 1. Configurar rutas
    # ---------------------------------------------------------------
    RAW_DIR       = Path("data/raw")
    VALID_DIR     = Path("data/filtered/valid")
    PROCESSED_DIR = Path("data/processed")
    LOG_PATH      = Path("logs/quality_control.csv")

    # ---------------------------------------------------------------
    # 2. Filtrado de calidad (ImageQualityFilter)
    # ---------------------------------------------------------------
    # Umbrales personalizados: más estricto en SNR para dataset ruidoso
    custom_thresholds = {
        **DEFAULT_THRESHOLDS,
        "snr_min": 18.0,          # Más exigente que el default (15 dB)
        "ring_score_max": 0.25,   # Menos tolerante a artefactos de anillo
    }

    qf = ImageQualityFilter(
        thresholds=custom_thresholds,
        patch_size=32,
        ring_n_harmonics=5,
        verbose=True,
    )

    print("=" * 60)
    print("ETAPA 1: Filtrado de calidad automático")
    print("=" * 60)

    summary = qf.batch_filter(
        folder_path=RAW_DIR,
        output_folder=Path("data/filtered"),
        log_path=LOG_PATH,
    )
    print(f"\nResumen QC: {summary}")

    # ---------------------------------------------------------------
    # 3. Preprocesamiento (CTPreprocessor)
    # ---------------------------------------------------------------
    # Configuración para arquitectura con preentrenamiento ImageNet
    config = PreprocessConfig(
        hu_min=-1500.0,
        hu_max=0.0,           # Extendido para capturar consolidaciones COVID
        bilateral_d=9,
        bilateral_sigma_color=12.0,
        bilateral_sigma_space=4.0,
        clahe_clip_limit=2.0,
        clahe_tile_grid=(8, 8),
        target_size=(224, 224),
        output_channels=3,    # Para ResNet50 / EfficientNet preentrenadas
        pad_value=0.0,
    )

    preprocessor = CTPreprocessor(config)

    print("\n" + "=" * 60)
    print("ETAPA 2: Estandarización y preprocesamiento")
    print("=" * 60)

    n_processed = preprocessor.process_folder(
        input_folder=VALID_DIR,
        output_folder=PROCESSED_DIR,
    )
    print(f"Tensores generados: {n_processed}")

    # ---------------------------------------------------------------
    # 4. Construir tf.data.Dataset (ejemplo con etiquetas dummy)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ETAPA 3: Construcción del Dataset TensorFlow")
    print("=" * 60)

    # En producción, labels viene de un CSV de anotaciones clínicas
    # Ejemplo: {nombre_archivo_sin_ext: 0=negativo, 1=positivo}
    labels = {
        p.stem: 1  # Placeholder: reemplazar con labels reales
        for p in PROCESSED_DIR.glob("*.npy")
    }

    ds = build_tf_dataset(
        npy_folder=PROCESSED_DIR,
        labels=labels,
        batch_size=32,
        shuffle=True,
        seed=42,
    )

    print(f"Dataset listo. Batches: {len(list(ds))}")
    print("Pipeline completado. Listo para entrenamiento CNN.\n")


# ---------------------------------------------------------------
# Casos borde y cómo manejarlos
# ---------------------------------------------------------------
EDGE_CASES = """
CASOS BORDE Y ESTRATEGIAS DE MANEJO
=====================================

1. DICOM sin RescaleSlope/Intercept
   → ImageQualityFilter.load_image() usa valores default (slope=1, intercept=0).
   → Riesgo: valores de pixel raw ≠ HU. Verificar con ds.Modality == "CT".
   → Fix: agregar comprobación de modalidad y loguear imágenes sin metadatos HU.

2. Imagen completamente negra (pulmón no visible, FOV incorrecto)
   → compute_snr() devuelve SNR muy bajo (< 5 dB).
   → local_variance() < 10 → flag "low_variance_flat_image".
   → La imagen se rechaza automáticamente.

3. Slice de CT fuera del campo pulmonar (mediastino puro, abdomen)
   → El histograma en ventana pulmonar estará completamente en 0 tras windowing.
   → Mismo flujo que imagen negra → rechazo por varianza.
   → Mejora opcional: segmentar pulmón con SimpleITK antes del filtrado QC.

4. Imagen con > 1 artefacto simultáneo (metal + movimiento)
   → detect_artifacts() retorna múltiples flags en "flags" field.
   → is_valid() evalúa en orden de severidad; el primer flag crítico determina razón.
   → El CSV de log guarda todos los flags para análisis post-hoc.

5. DICOM multiframe (CT volumétrico completo en un solo archivo)
   → load_image() toma el primer slice [..0].
   → Para volumétricos: iterar sobre pixel_array con bucle externo antes de filtrar.
   → Considerar samplear cada N slices para evitar correlación entre slices.

6. PNG/JPG de 16 bits (TIFF médico exportado)
   → cv2.IMREAD_GRAYSCALE colapsa a 8 bits → pérdida de información HU.
   → Fix: usar cv2.IMREAD_ANYDEPTH y mapear al rango correcto explícitamente.

7. GPU 4 GB vRAM (desarrollo en GTX 1650)
   → batch_size=8 máximo para CNN 224×224 con backbone ResNet50.
   → Usar tf.config.experimental.set_memory_growth(True) para evitar OOM.
   → En producción (8 GB vRAM): batch_size=32 seguro.

8. Imágenes de distintos centros con bit-depth diferente (8 vs 12 vs 16 bits)
   → HU windowing normaliza a [0,1] independientemente del bit-depth.
   → CLAHE opera en uint8 → la conversión es robusta al bit-depth original.
   → Verificar que el rescale HU sea correcto antes de comparar centros.

9. Ruido de anillo severo pero SNR aceptable
   → ring_score > 0.30 causa rechazo independientemente del SNR.
   → Si el dataset tiene muchas imágenes con anillos leves, calibrar
     ring_score_max a 0.40 y añadir corrección de anillo (filtro polar FFT).

10. Imágenes con lesiones muy pequeñas (< 3 mm) en vidrio esmerilado
    → bilateral con sigma_color > 15 puede borrar estas lesiones.
    → Reducir bilateral_sigma_color a 8 y aumentar CLAHE clip_limit a 3.0
      para compensar la pérdida de contraste en regiones finas.
"""

if __name__ == "__main__":
    print(EDGE_CASES)
    # Descomentar para ejecutar el pipeline completo:
    # main()
