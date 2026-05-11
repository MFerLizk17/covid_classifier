"""
Punto de entrada principal del proyecto CT COVID-19.

Modos de ejecución:
    python main.py --mode preprocess   → pipeline de preprocesamiento (Sprint 1)
    python main.py --mode train        → entrena el Autoencoder UNET (Sprint 2)
    python main.py --mode evaluate     → evalúa modelo guardado sobre test set
    python main.py --mode predict --image ruta/imagen.png  → clasifica una imagen

Uso típico (flujo completo):
    python main.py --mode preprocess
    python main.py --mode train
    python main.py --mode evaluate
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from preprocesamiento import run_pipeline
from config import CONFIG, UNET_CONFIG

RAW_DIR       = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
CLASES        = ["COVID-19", "Non-COVID-19"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CT_Main")


# ─────────────────────────────────────────────
# MODO 1: PREPROCESAMIENTO (Sprint 1)
# ─────────────────────────────────────────────

def save_image(img: np.ndarray, output_path: Path) -> None:
    """
    Guarda imagen procesada en disco como PNG.

    Args:
        img: Array float32 normalizado en [0.0, 1.0].
        output_path: Ruta de destino.
    """
    img_u8 = (img * 255).astype(np.uint8)
    cv2.imwrite(str(output_path), img_u8)


def run_preprocessing(raw_dir: Path, processed_dir: Path) -> Dict:
    """
    Ejecuta el pipeline de preprocesamiento sobre todas las imágenes del dataset.

    Args:
        raw_dir: Directorio raíz con subcarpetas por clase.
        processed_dir: Directorio de salida para imágenes procesadas.

    Returns:
        Diccionario con estadísticas del proceso.
    """
    tiempos = []
    stats: Dict = {
        "total":                 0,
        "procesadas":            0,
        "descartadas_calidad":   0,
        "descartadas_mascara":   0,
        "errores_lectura":       0,
        "tiempo_total_s":        0.0,
        "tiempo_promedio_s":     0.0,
        "balance":               {clase: 0 for clase in CLASES},
        "balance_original":      {clase: 0 for clase in CLASES},
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
                result, motivo = run_pipeline(img_file)
                elapsed        = time.perf_counter() - start
                tiempos.append(elapsed)

                if result is not None:
                    save_image(result, clase_processed_dir / img_file.name)
                    stats["procesadas"]     += 1
                    stats["balance"][clase] += 1
                    logger.info(f"✓ [{clase}] {img_file.name} en {elapsed:.3f}s")

                elif motivo == "calidad":
                    stats["descartadas_calidad"] += 1
                    logger.warning(
                        f"✗ [{clase}] {img_file.name} descartada por SNR"
                    )
                elif motivo == "mascara":
                    stats["descartadas_mascara"] += 1
                    logger.warning(
                        f"✗ [{clase}] {img_file.name} descartada por máscara vacía"
                    )

            except Exception as e:
                stats["errores_lectura"] += 1
                logger.error(f"Error en [{clase}] {img_file.name}: {e}")

    stats["tiempo_total_s"]    = round(sum(tiempos), 3)
    stats["tiempo_promedio_s"] = round(
        sum(tiempos) / len(tiempos) if tiempos else 0.0, 3
    )
    return stats


def print_report(stats: Dict) -> None:
    """
    Imprime reporte final de indicadores del preprocesamiento.

    Args:
        stats: Diccionario retornado por run_preprocessing.
    """
    total             = stats["total"]
    total_descartadas = stats["descartadas_calidad"] + stats["descartadas_mascara"]
    tasa_exito        = (stats["procesadas"] / total * 100) if total > 0 else 0.0
    tasa_descarte     = (total_descartadas / total * 100) if total > 0 else 0.0
    cumple_meta       = (
        "✓ CUMPLE" if tasa_exito >= CONFIG.META_TASA_EXITO else "✗ NO CUMPLE"
    )

    total_proc   = stats["procesadas"]
    balance_post = {}
    balance_orig = {}

    for clase in CLASES:
        n_post = stats["balance"][clase]
        n_orig = stats["balance_original"][clase]
        balance_post[clase] = (
            n_post, (n_post / total_proc * 100) if total_proc > 0 else 0.0
        )
        balance_orig[clase] = (
            n_orig, (n_orig / total * 100) if total > 0 else 0.0
        )

    sep = "=" * 55
    print(f"\n{sep}\n REPORTE DE PREPROCESAMIENTO CT – SPRINT 1\n{sep}")
    print(f" Total imágenes encontradas  : {total}")
    print(f" Procesadas exitosamente     : {stats['procesadas']} ({tasa_exito:.1f}%)")
    print(f" Meta (≥ {CONFIG.META_TASA_EXITO:.0f}%)             : {cumple_meta}")
    print(f"\n Total descartadas           : {total_descartadas} ({tasa_descarte:.1f}%)")
    print(f" Errores de lectura          : {stats['errores_lectura']}")
    print(
        f"\n[TIEMPOS DE PROCESAMIENTO]\n"
        f" Tiempo total                : {stats['tiempo_total_s']} s\n"
        f" Tiempo promedio por imagen  : {stats['tiempo_promedio_s']} s\n{sep}\n"
    )


# ─────────────────────────────────────────────
# MODO 2: ENTRENAMIENTO UNET (Sprint 2)
# ─────────────────────────────────────────────

def run_training(processed_dir: Path) -> None:
    """
    Entrena el Autoencoder sobre las imágenes ya preprocesadas.

    Importa los módulos de la UNET solo cuando se necesitan para no
    requerir torch en el modo preprocess.

    Args:
        processed_dir: Directorio con subcarpetas COVID-19/ y Non-COVID-19/.
    """
    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    from dataset   import TomografiaDataset
    from model     import Autoencoder
    from trainer   import train_one_epoch, compute_centroids
    from evaluator import evaluate_test_set
    from visualizer import plot_losses

    cfg    = UNET_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo: {device}")

    # ── Dataset completo ────────────────────────────────────────────
    dataset = TomografiaDataset(
        ruta_covid    = processed_dir / "covid",
        ruta_no_covid = processed_dir / "non_covid",
        img_size      = cfg.IMG_SIZE,
    )
    logger.info(f"Dataset cargado: {len(dataset)} imágenes")

    # ── Split estratificado 80/20 ───────────────────────────────────
    indices   = list(range(len(dataset)))
    etiquetas = [dataset.datos[i][1] for i in indices]

    idx_train, idx_test = train_test_split(
        indices,
        test_size=0.2,
        stratify=etiquetas,
        random_state=42,
    )

    dataset_train = Subset(dataset, idx_train)
    dataset_test  = Subset(dataset, idx_test)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    logger.info(
        f"Train: {len(dataset_train)} | Test: {len(dataset_test)}"
    )

    # ── Modelo ─────────────────────────────────────────────────────
    modelo = Autoencoder(cfg.LATENT_DIM).to(device)
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logger.info(f"Parámetros entrenables: {total_params:,}")

    # ── Entrenamiento ───────────────────────────────────────────────
    historial = train_one_epoch(
        modelo     = modelo,
        dataloader = dataloader_train,
        cfg        = cfg,
        device     = device,
    )

    plot_losses(historial)

    # ── Centroides ─────────────────────────────────────────────────
    centroide_covid, centroide_no_covid = compute_centroids(
        modelo  = modelo,
        dataset = dataset,
        device  = device,
    )

    # ── Evaluación en test ─────────────────────────────────────────
    evaluate_test_set(
        modelo             = modelo,
        dataset_test       = dataset_test,
        centroide_covid    = centroide_covid,
        centroide_no_covid = centroide_no_covid,
        device             = device,
        cfg                = cfg,
    )

    # ── Persistencia ───────────────────────────────────────────────
    _save_model(modelo, centroide_covid, centroide_no_covid, cfg)


def _save_model(modelo, centroide_covid, centroide_no_covid, cfg) -> None:
    """
    Guarda los pesos del modelo y los centroides en disco.

    Args:
        modelo: Autoencoder entrenado.
        centroide_covid: Vector numpy del centroide COVID.
        centroide_no_covid: Vector numpy del centroide No-COVID.
        cfg: Instancia de UNetConfig.
    """
    import torch
    import pickle

    torch.save(modelo.state_dict(), cfg.MODEL_PATH)

    config_persistencia = {
        "centroide_covid":    centroide_covid,
        "centroide_no_covid": centroide_no_covid,
        "latent_dim":         cfg.LATENT_DIM,
        "img_size":           cfg.IMG_SIZE,
    }
    with open(cfg.CENTROIDS_PATH, "wb") as f:
        pickle.dump(config_persistencia, f)

    logger.info(f"Modelo guardado en:     {cfg.MODEL_PATH}")
    logger.info(f"Centroides guardados en: {cfg.CENTROIDS_PATH}")


# ─────────────────────────────────────────────
# MODO 3: EVALUACIÓN DE MODELO GUARDADO
# ─────────────────────────────────────────────

def run_evaluation(processed_dir: Path) -> None:
    """
    Carga un modelo guardado y lo evalúa sobre el test set.

    Args:
        processed_dir: Directorio con imágenes preprocesadas.
    """
    import torch
    import pickle
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    from dataset   import TomografiaDataset
    from model     import Autoencoder
    from evaluator import evaluate_test_set

    cfg    = UNET_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    modelo = Autoencoder(cfg.LATENT_DIM).to(device)
    modelo.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    logger.info(f"Modelo cargado desde: {cfg.MODEL_PATH}")

    with open(cfg.CENTROIDS_PATH, "rb") as f:
        saved = pickle.load(f)

    centroide_covid    = saved["centroide_covid"]
    centroide_no_covid = saved["centroide_no_covid"]

    dataset = TomografiaDataset(
        ruta_covid    = processed_dir / "covid",
        ruta_no_covid = processed_dir / "non_covid",
        img_size      = cfg.IMG_SIZE,
    )

    indices   = list(range(len(dataset)))
    etiquetas = [dataset.datos[i][1] for i in indices]
    _, idx_test = train_test_split(
        indices, test_size=0.2, stratify=etiquetas, random_state=42
    )
    dataset_test = Subset(dataset, idx_test)

    evaluate_test_set(
        modelo             = modelo,
        dataset_test       = dataset_test,
        centroide_covid    = centroide_covid,
        centroide_no_covid = centroide_no_covid,
        device             = device,
        cfg                = cfg,
    )


# ─────────────────────────────────────────────
# MODO 4: PREDICCIÓN DE UNA IMAGEN
# ─────────────────────────────────────────────

def run_predict(image_path: str) -> None:
    """
    Clasifica una sola imagen TAC usando el modelo guardado.

    Args:
        image_path: Ruta a la imagen a clasificar.
    """
    import torch
    import pickle

    from model     import Autoencoder
    from evaluator import classify_image

    cfg    = UNET_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    modelo = Autoencoder(cfg.LATENT_DIM).to(device)
    modelo.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))

    with open(cfg.CENTROIDS_PATH, "rb") as f:
        saved = pickle.load(f)

    resultado = classify_image(
        ruta_imagen        = image_path,
        modelo             = modelo,
        centroide_covid    = saved["centroide_covid"],
        centroide_no_covid = saved["centroide_no_covid"],
        img_size           = cfg.IMG_SIZE,
        device             = device,
        mostrar            = True,
    )

    print(f"\n Resultado para {Path(image_path).name}:")
    print(f"   Clase      : {resultado['clase']}")
    print(f"   Confianza  : {resultado['confianza']:.1f}%")
    print(f"   Dist COVID : {resultado['dist_covid']:.4f}")
    print(f"   Dist No-COVID: {resultado['dist_no_covid']:.4f}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline CT COVID-19: preprocesamiento y clasificación UNET"
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "evaluate", "predict"],
        default="preprocess",
        help="Modo de ejecución (default: preprocess)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a la imagen a clasificar (solo para --mode predict)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "preprocess":
        logger.info("Modo: preprocesamiento")
        stats = run_preprocessing(RAW_DIR, PROCESSED_DIR)
        print_report(stats)

    elif args.mode == "train":
        logger.info("Modo: entrenamiento UNET")
        run_training(PROCESSED_DIR)

    elif args.mode == "evaluate":
        logger.info("Modo: evaluación de modelo guardado")
        run_evaluation(PROCESSED_DIR)

    elif args.mode == "predict":
        if not args.image:
            logger.error("--mode predict requiere --image <ruta>")
        else:
            run_predict(args.image)