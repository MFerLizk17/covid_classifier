"""
Punto de entrada principal del proyecto CT COVID-19.

Modos de ejecución:
    python main.py --mode preprocess   → pipeline de preprocesamiento (Sprint 1)
    python main.py --mode train        → entrena el Autoencoder UNET (Sprint 2)
    python main.py --mode evaluate     → evalúa modelo guardado sobre test set
    python main.py --mode predict      → clasificación interactiva de una imagen

Uso típico (flujo completo):
    python main.py --mode preprocess
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode predict
    python main.py --mode predict --image data/processed/covid/imagen.png
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from preprocesamiento import run_pipeline
from config import CONFIG, UNET_CONFIG

RAW_DIR       = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
CLASES        = ["covid", "non-covid"]

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
        processed_dir: Directorio con subcarpetas covid/ y non_covid/.
    """
    import torch
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split

    from dataset    import TomografiaDataset
    from model      import Autoencoder
    from trainer    import train_one_epoch, compute_centroids
    from evaluator  import evaluate_test_set
    from visualizer import plot_losses

    cfg    = UNET_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo: {device}")

    # ── Dataset completo ────────────────────────────────────────
    dataset = TomografiaDataset(
        ruta_covid    = processed_dir / "covid",
        ruta_no_covid = processed_dir / "non_covid",
        img_size      = cfg.IMG_SIZE,
    )
    logger.info(f"Dataset cargado: {len(dataset)} imágenes")

    # ── Split estratificado 80/20 ───────────────────────────────
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

    logger.info(f"Train: {len(dataset_train)} | Test: {len(dataset_test)}")

    # ── Modelo ──────────────────────────────────────────────────
    modelo = Autoencoder(cfg.LATENT_DIM).to(device)
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logger.info(f"Parámetros entrenables: {total_params:,}")

    # ── Entrenamiento ───────────────────────────────────────────
    historial = train_one_epoch(
        modelo     = modelo,
        dataloader = dataloader_train,
        cfg        = cfg,
        device     = device,
    )

    plot_losses(historial)

    # ── Centroides ──────────────────────────────────────────────
    centroide_covid, centroide_no_covid = compute_centroids(
        modelo  = modelo,
        dataset = dataset,
        device  = device,
    )

    # ── Evaluación en test ──────────────────────────────────────
    evaluate_test_set(
        modelo             = modelo,
        dataset_test       = dataset_test,
        centroide_covid    = centroide_covid,
        centroide_no_covid = centroide_no_covid,
        device             = device,
        cfg                = cfg,
    )

    # ── Persistencia ────────────────────────────────────────────
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

    logger.info(f"Modelo guardado en:      {cfg.MODEL_PATH}")
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
    from torch.utils.data import Subset
    from sklearn.model_selection import train_test_split

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
# MODO 4: PREDICCIÓN INTERACTIVA CLI
# ─────────────────────────────────────────────

def _listar_imagenes_disponibles(processed_dir: Path) -> List[Path]:
    """
    Recolecta todas las imágenes disponibles en el directorio procesado.

    Busca en las subcarpetas covid/ y non_covid/ del directorio
    de imágenes procesadas.

    Args:
        processed_dir: Directorio raíz de imágenes procesadas.

    Returns:
        Lista de rutas Path a las imágenes encontradas.
    """
    imagenes: List[Path] = []
    for carpeta in ["covid", "non_covid"]:
        ruta = processed_dir / carpeta
        if ruta.exists():
            for f in sorted(ruta.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXT:
                    imagenes.append(f)
    return imagenes


def _mostrar_menu_seleccion(imagenes: List[Path]) -> None:
    """
    Imprime el menú numerado de imágenes disponibles en terminal.

    Args:
        imagenes: Lista de rutas a imágenes disponibles.
    """
    sep = "=" * 55
    print(f"\n{sep}")
    print("  CLASIFICADOR COVID-19 — SELECCIÓN DE IMAGEN")
    print(f"{sep}")
    print(f"\n  Imágenes disponibles en data/processed/:\n")

    for i, img in enumerate(imagenes, 1):
        clase_carpeta = img.parent.name
        print(f"  [{i:>3}] {clase_carpeta}/{img.name}")

    print(f"\n  [  0] Ingresar ruta manual")
    print("-" * 55)


def _solicitar_ruta_manual() -> str:
    """
    Solicita al usuario ingresar una ruta de imagen manualmente.

    Valida que el archivo exista antes de retornar.

    Returns:
        Ruta válida a la imagen como string.
    """
    print("\n  Ingresa la ruta completa a la imagen TAC:")
    print("  Ejemplo: data/processed/covid/imagen.png")
    print("-" * 55)

    while True:
        ruta = input("\n  Ruta: ").strip()
        if Path(ruta).exists():
            return ruta
        print(f"  ⚠  No se encontró el archivo: {ruta}")


def _seleccionar_imagen(processed_dir: Path) -> str:
    """
    Flujo completo de selección de imagen por CLI.

    Muestra la lista de imágenes disponibles en data/processed/ y
    permite al usuario elegir por número o ingresar una ruta manual.

    Args:
        processed_dir: Directorio raíz de imágenes procesadas.

    Returns:
        Ruta absoluta a la imagen seleccionada como string.
    """
    imagenes = _listar_imagenes_disponibles(processed_dir)

    if not imagenes:
        logger.warning(
            "No se encontraron imágenes en data/processed/. "
            "Ejecuta primero: python main.py --mode preprocess"
        )
        return _solicitar_ruta_manual()

    _mostrar_menu_seleccion(imagenes)

    while True:
        try:
            opcion = int(
                input("\n  Selecciona el número de la imagen (0 para ruta manual): ")
                .strip()
            )
            if opcion == 0:
                return _solicitar_ruta_manual()
            elif 1 <= opcion <= len(imagenes):
                return str(imagenes[opcion - 1])
            else:
                print(f"  ⚠  Ingresa un número entre 0 y {len(imagenes)}")
        except ValueError:
            print("  ⚠  Ingresa un número válido")


def _previsualizar_imagen(img_np: np.ndarray, nombre: str) -> None:
    """
    Muestra la imagen seleccionada antes del análisis.

    El flujo se bloquea hasta que el usuario cierra la ventana,
    momento en el que se inicia el análisis del modelo.

    Args:
        img_np: Array numpy de la imagen normalizada en [0, 1].
        nombre: Nombre del archivo para mostrar en el título.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_np, cmap="gray")
    ax.set_title(
        f"{nombre}\n⬇ Analisis en progreso...",
        fontsize=11, color="#1565C0",
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("data/preview.png", dpi = 150, bbox_inches ="tight")
    plt.close(fig)
    logger.info("Vista previa guardada en: preview.png")


def _calcular_heatmap(enc4_output, target_size: int) -> np.ndarray:
    """
    Genera heatmap normalizado desde las activaciones de enc4.

    Promedia los 256 canales del feature map para obtener un mapa
    de importancia 2D y lo redimensiona al tamaño de la imagen.

    Args:
        enc4_output: Tensor de activaciones (1, 256, H, W).
        target_size: Tamaño final del heatmap en píxeles (cuadrado).

    Returns:
        Array numpy normalizado en [0, 1] de forma (target_size, target_size).
    """
    import torch
    from PIL import Image as PILImage

    act     = enc4_output.squeeze().cpu().numpy()   # (256, H, W)
    heatmap = act.mean(axis=0)                       # (H, W)
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    hmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
    return np.array(
        hmap_pil.resize((target_size, target_size), PILImage.BILINEAR)
    ) / 255.0


def _calcular_probabilidades(
    z_np: np.ndarray,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
) -> Tuple[float, float, str]:
    """
    Calcula probabilidades COVID/No-COVID por distancia euclidiana al centroide.

    La probabilidad de cada clase es proporcional a la distancia al
    centroide contrario: más lejos del centroide No-COVID → más probable COVID.

    Args:
        z_np: Vector latente de la imagen, forma (latent_dim,).
        centroide_covid: Centroide COVID en espacio latente.
        centroide_no_covid: Centroide No-COVID en espacio latente.

    Returns:
        Tupla (prob_covid, prob_no_covid, clase_predicha).
    """
    dist_covid    = float(np.linalg.norm(z_np - centroide_covid))
    dist_no_covid = float(np.linalg.norm(z_np - centroide_no_covid))
    dist_total    = dist_covid + dist_no_covid

    prob_covid    = (dist_no_covid / dist_total) * 100.0
    prob_no_covid = (dist_covid    / dist_total) * 100.0
    clase         = "COVID" if dist_covid < dist_no_covid else "No-COVID"

    return prob_covid, prob_no_covid, clase


def _imprimir_resultado(
    nombre_imagen: str,
    prob_covid: float,
    prob_no_covid: float,
    clase: str,
    porcentaje_zona: float,
) -> None:
    """
    Imprime el reporte de clasificación en la terminal.

    Incluye barras de texto proporcionales a las probabilidades
    para facilitar la lectura visual en terminal.

    Args:
        nombre_imagen: Nombre del archivo analizado.
        prob_covid: Probabilidad COVID en porcentaje.
        prob_no_covid: Probabilidad No-COVID en porcentaje.
        clase: Clase predicha ('COVID' o 'No-COVID').
        porcentaje_zona: Porcentaje del área pulmonar analizada.
    """
    sep    = "=" * 55
    titulo = "COVID-19 POSITIVO ⚠" if clase == "COVID" else "No-COVID NEGATIVO ✓"

    barra_covid    = "▓" * int(prob_covid    / 2)
    barra_no_covid = "▓" * int(prob_no_covid / 2)

    print(f"\n{sep}")
    print(f"  RESULTADO: {titulo}")
    print(f"{sep}")
    print(f"  Imagen analizada : {nombre_imagen}")
    print(f"\n  PROBABILIDADES:")
    print(f"  {barra_covid:<25} COVID    : {prob_covid:.1f}%")
    print(f"  {barra_no_covid:<25} No-COVID : {prob_no_covid:.1f}%")
    print(f"\n  ZONA DIAGNÓSTICA:")
    print(f"  El modelo focalizó su atención en el {porcentaje_zona:.1f}%")
    print(f"  del área pulmonar para tomar esta decisión.")
    print(f"  Las zonas en rojo en la imagen son las de")
    print(f"  mayor peso diagnóstico.")
    print(f"{sep}\n")


def _mostrar_resultado_visual(
    img_np: np.ndarray,
    recon_np: np.ndarray,
    heatmap: np.ndarray,
    prob_covid: float,
    prob_no_covid: float,
    clase: str,
) -> None:
    """
    Muestra ventana con 4 paneles de análisis del resultado.

    Paneles:
        1. Imagen TAC original seleccionada
        2. Heatmap de zona diagnóstica superpuesto
        3. Reconstrucción interna del modelo
        4. Barras horizontales de probabilidad

    Args:
        img_np: Imagen original como array numpy (H, W).
        recon_np: Imagen reconstruida por el Decoder (H, W).
        heatmap: Mapa de importancia normalizado (H, W).
        prob_covid: Probabilidad COVID en porcentaje.
        prob_no_covid: Probabilidad No-COVID en porcentaje.
        clase: Clase predicha ('COVID' o 'No-COVID').
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    color_titulo = "#C62828" if clase == "COVID" else "#1565C0"
    titulo       = (
        f"Resultado: {'COVID-19 POSITIVO ⚠' if clase == 'COVID' else 'No-COVID NEGATIVO ✓'}"
        f"  |  COVID: {prob_covid:.1f}%  |  No-COVID: {prob_no_covid:.1f}%"
    )

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(titulo, fontsize=13, fontweight="bold", color=color_titulo)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # Panel 1 — Imagen original
    ax1.imshow(img_np, cmap="gray")
    ax1.set_title("Imagen TAC\noriginal", fontsize=10)
    ax1.axis("off")

    # Panel 2 — Zona diagnóstica
    ax2.imshow(img_np, cmap="gray")
    hmap_plot = ax2.imshow(heatmap, cmap="jet", alpha=0.5)
    ax2.set_title("Zona diagnóstica\n(rojo = mayor peso)", fontsize=10)
    ax2.axis("off")
    plt.colorbar(hmap_plot, ax=ax2, fraction=0.046)

    # Panel 3 — Reconstrucción interna
    ax3.imshow(recon_np, cmap="gray")
    ax3.set_title("Reconstrucción\ndel modelo", fontsize=10)
    ax3.axis("off")

    # Panel 4 — Barras de probabilidad
    etiquetas = ["COVID", "No-COVID"]
    valores   = [prob_covid, prob_no_covid]
    colores   = ["#E53935", "#1E88E5"]

    bars = ax4.barh(etiquetas, valores, color=colores, alpha=0.85, height=0.4)
    ax4.set_xlim(0, 100)
    ax4.set_xlabel("Probabilidad (%)")
    ax4.set_title("Probabilidades\nde clasificación", fontsize=10)
    ax4.axvline(x=50, color="gray", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, valores):
        ax4.text(
            val + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", fontsize=11, fontweight="bold",
        )
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig("data/resultado_clasificacion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Resultado guardado en: resultado_clasificacion.png")


def run_predict(image_path: Optional[str] = None) -> None:
    """
    Modo interactivo de clasificación de una imagen TAC individual.

    Flujo completo:
        1. Si no se pasa image_path, abre el selector CLI numerado
        2. Muestra la imagen seleccionada → usuario confirma cerrando ventana
        3. El modelo analiza la imagen
        4. Imprime probabilidades y zona diagnóstica en terminal
        5. Abre ventana con 4 paneles: original, heatmap, reconstrucción, barras

    Args:
        image_path: Ruta opcional a la imagen. Si es None activa el selector CLI.
    """
    import torch
    import pickle
    from PIL import Image as PILImage
    from torchvision import transforms

    from model import Autoencoder

    cfg    = UNET_CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Verificar modelo disponible ─────────────────────────────
    if not Path(cfg.MODEL_PATH).exists():
        logger.error(
            f"No se encontró el modelo en '{cfg.MODEL_PATH}'.\n"
            "Ejecuta primero: python main.py --mode train"
        )
        return

    # ── Selección de imagen ─────────────────────────────────────
    if image_path is None:
        image_path = _seleccionar_imagen(PROCESSED_DIR)

    # ── Transformación ──────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    img_pil    = PILImage.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    img_np     = img_tensor.cpu().squeeze().numpy()

    # ── Previsualización antes del análisis ─────────────────────
    _previsualizar_imagen(img_np, Path(image_path).name)

    # ── Cargar modelo y centroides ──────────────────────────────
    modelo = Autoencoder(cfg.LATENT_DIM).to(device)
    modelo.load_state_dict(torch.load(cfg.MODEL_PATH, map_location=device))
    modelo.eval()

    with open(cfg.CENTROIDS_PATH, "rb") as f:
        saved              = pickle.load(f)
    centroide_covid    = saved["centroide_covid"]
    centroide_no_covid = saved["centroide_no_covid"]

    # ── Hook para capturar activaciones de enc4 ─────────────────
    activacion: Dict = {}

    def _hook_enc4(module, input, output) -> None:  # noqa: ANN001
        activacion["enc4"] = output.detach()

    handle = modelo.encoder.enc4.register_forward_hook(_hook_enc4)

    with torch.no_grad():
        recon, z = modelo(img_tensor)

    handle.remove()

    # ── Vectores y métricas ─────────────────────────────────────
    z_np             = z.cpu().numpy().squeeze()
    recon_np         = recon.cpu().squeeze().numpy()
    heatmap          = _calcular_heatmap(activacion["enc4"], cfg.IMG_SIZE)
    prob_covid, prob_no_covid, clase = _calcular_probabilidades(
        z_np, centroide_covid, centroide_no_covid
    )
    porcentaje_zona  = float((heatmap >= 0.75).mean() * 100)

    # ── Reporte en terminal ─────────────────────────────────────
    _imprimir_resultado(
        nombre_imagen   = Path(image_path).name,
        prob_covid      = prob_covid,
        prob_no_covid   = prob_no_covid,
        clase           = clase,
        porcentaje_zona = porcentaje_zona,
    )

    # ── Visualización final ─────────────────────────────────────
    _mostrar_resultado_visual(
        img_np       = img_np,
        recon_np     = recon_np,
        heatmap      = heatmap,
        prob_covid   = prob_covid,
        prob_no_covid= prob_no_covid,
        clase        = clase,
    )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline CT COVID-19: preprocesamiento y clasificación UNET",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "evaluate", "predict"],
        default="preprocess",
        help=(
            "Modo de ejecución:\n"
            "  preprocess → pipeline de preprocesamiento\n"
            "  train      → entrena el Autoencoder\n"
            "  evaluate   → evalúa modelo guardado\n"
            "  predict    → clasifica una imagen (interactivo)"
        ),
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a la imagen a clasificar (opcional para --mode predict)",
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
        logger.info("Modo: predicción interactiva")
        run_predict(args.image)