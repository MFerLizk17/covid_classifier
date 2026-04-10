
import logging
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from src.preprocessor import CTPreprocessor

# --- CONFIGURACIÓN ---
RAW_DIR            = Path("./data/raw")
PROCESSED_DIR      = Path("./data/processed")
SUPPORTED_EXT      = {".png", ".jpg", ".jpeg", ".bmp"}
CLASES             = ["COVID-19", "Non-COVID-19"]
META_TASA_EXITO    = 95.0   # % mínimo aceptable de procesamiento exitoso

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CT_Main")


def save_image(img: np.ndarray, output_path: Path) -> None:
    """
    Guarda una imagen procesada en disco en formato PNG.

    Args:
        img: Array float32 normalizado en [0.0, 1.0].
        output_path: Ruta completa de destino incluyendo nombre de archivo.
    """
    img_u8 = (img * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), img_u8)


def run(raw_dir: Path, processed_dir: Path) -> Dict:
    """
    Ejecuta el pipeline sobre las imágenes de ambas clases del dataset.

    Recorre raw_dir/COVID-19/ y raw_dir/Non-COVID-19/, aplica el pipeline
    de preprocesamiento a cada imagen y guarda los resultados manteniendo
    la estructura de carpetas por clase en processed_dir.

    Args:
        raw_dir: Directorio raíz con subcarpetas por clase (COVID-19, Non-COVID-19).
        processed_dir: Directorio raíz de destino para imágenes procesadas.

    Returns:
        Diccionario con estadísticas del procesamiento:
            - total (int): imágenes encontradas en ambas clases.
            - procesadas (int): imágenes procesadas exitosamente.
            - descartadas_calidad (int): rechazadas por SNR < 10 dB.
            - errores_lectura (int): archivos corruptos o ilegibles.
            - tiempo_total_s (float): tiempo total de ejecución.
            - tiempo_promedio_s (float): tiempo promedio por imagen.
            - balance (dict): imágenes procesadas por clase {clase: int}.
            - balance_original (dict): imágenes totales por clase antes del descarte.
    """
    preprocessor = CTPreprocessor()
    tiempos = []

    stats: Dict = {
        "total": 0,
        "procesadas": 0,
        "descartadas_calidad": 0,
        "errores_lectura": 0,
        "tiempo_total_s": 0.0,
        "tiempo_promedio_s": 0.0,
        "balance": {clase: 0 for clase in CLASES},
        "balance_original": {clase: 0 for clase in CLASES},
    }

    for clase in CLASES:
        clase_raw_dir       = raw_dir / clase
        clase_processed_dir = processed_dir / clase

        if not clase_raw_dir.exists():
            logger.warning(f"Carpeta no encontrada, se omite: {clase_raw_dir}")
            continue

        clase_processed_dir.mkdir(parents=True, exist_ok=True)

        image_files = [
            f for f in clase_raw_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXT
        ]

        stats["balance_original"][clase] = len(image_files)
        stats["total"] += len(image_files)

        for img_file in image_files:
            start = time.perf_counter()
            try:
                result = preprocessor.run_pipeline(img_file)
                elapsed = time.perf_counter() - start
                tiempos.append(elapsed)

                if result is not None:
                    save_image(result, clase_processed_dir / img_file.name)
                    stats["procesadas"] += 1
                    stats["balance"][clase] += 1
                    logger.info(f"✓ [{clase}] {img_file.name} en {elapsed:.3f}s")
                else:
                    stats["descartadas_calidad"] += 1
                    logger.warning(f"✗ [{clase}] {img_file.name} descartada por baja calidad")

            except Exception as e:
                stats["errores_lectura"] += 1
                logger.error(f"Error en [{clase}] {img_file.name}: {e}")

    stats["tiempo_total_s"]   = round(sum(tiempos), 3)
    stats["tiempo_promedio_s"] = round(
        sum(tiempos) / len(tiempos) if tiempos else 0.0, 3
    )

    return stats


def print_report(stats: Dict) -> None:
    """
    Imprime el reporte final de indicadores del Sprint 1.

    Muestra tasa de éxito vs meta, desglose de descartes, balance de clases
    antes y después del filtro de calidad, y tiempos de procesamiento.

    Args:
        stats: Diccionario retornado por run().
    """
    total = stats["total"]

    tasa_exito    = (stats["procesadas"] / total * 100) if total > 0 else 0.0
    tasa_descarte = (stats["descartadas_calidad"] / total * 100) if total > 0 else 0.0
    cumple_meta   = "✓ CUMPLE" if tasa_exito >= META_TASA_EXITO else "✗ NO CUMPLE"

    # Balance de clases post-limpieza
    total_procesadas = stats["procesadas"]
    balance_post = {}
    for clase in CLASES:
        n = stats["balance"][clase]
        pct = (n / total_procesadas * 100) if total_procesadas > 0 else 0.0
        balance_post[clase] = (n, pct)

    # Balance original
    balance_orig = {}
    for clase in CLASES:
        n = stats["balance_original"][clase]
        pct = (n / total * 100) if total > 0 else 0.0
        balance_orig[clase] = (n, pct)

    sep = "=" * 55
    print(f"\n{sep}")
    print("        REPORTE DE PREPROCESAMIENTO CT – SPRINT 1")
    print(sep)

    print("\n  [TASA DE PROCESAMIENTO EXITOSO]")
    print(f"  Total imágenes encontradas  : {total}")
    print(f"  Procesadas exitosamente     : {stats['procesadas']} ({tasa_exito:.1f}%)")
    print(f"  Meta (≥ {META_TASA_EXITO:.0f}%)               : {cumple_meta}")

    print("\n  [TASA DE DESCARTE POR CALIDAD]")
    print(f"  Descartadas (SNR < 10 dB)   : {stats['descartadas_calidad']} ({tasa_descarte:.1f}%)")
    print(f"  Errores de lectura          : {stats['errores_lectura']}")

    print("\n  [BALANCE DE CLASES POST-LIMPIEZA]")
    print(f"  {'Clase':<20} {'Original':>12} {'Post-limpieza':>15}")
    print(f"  {'-'*48}")
    for clase in CLASES:
        n_orig, pct_orig = balance_orig[clase]
        n_post, pct_post = balance_post[clase]
        print(f"  {clase:<20} {n_orig:>6} ({pct_orig:4.1f}%)  {n_post:>6} ({pct_post:4.1f}%)")

    print("\n  [TIEMPOS DE PROCESAMIENTO]")
    print(f"  Tiempo total                : {stats['tiempo_total_s']} s")
    print(f"  Tiempo promedio por imagen  : {stats['tiempo_promedio_s']} s")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    logger.info("Iniciando pipeline de preprocesamiento...")
    stats = run(RAW_DIR, PROCESSED_DIR)
    print_report(stats)