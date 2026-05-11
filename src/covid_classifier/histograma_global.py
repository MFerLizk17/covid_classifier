"""
Histogramas globales de distribución de intensidades por clase.

Acumula los histogramas de todas las imágenes de cada clase
y los visualiza comparativamente.

Uso:
    python -m src.histograma_global
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

RAW_DIR       = Path("./data/raw")
CLASES        = ["COVID-19", "Non-COVID-19"]
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
COLORES       = {"COVID-19": "#E74C3C", "Non-COVID-19": "#2980B9"}


def acumular_histograma(clase_dir: Path) -> np.ndarray:
    """
    Acumula el histograma de todas las imágenes de una clase.

    Ignora el valor 0 (fondo negro del escáner) para que la
    distribución refleje únicamente los tejidos reales.

    Args:
        clase_dir: Directorio con las imágenes de la clase.

    Returns:
        Array de 256 valores con la frecuencia acumulada por bin.
    """
    hist_total = np.zeros(256, dtype=np.float64)
    image_files = [
        f for f in clase_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    ]

    logger.info(f"Procesando {len(image_files)} imágenes de {clase_dir.name}...")

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 255))
        hist_total += hist

    # Excluir bin 0 (fondo negro del escáner CT)
    hist_total[0] = 0

    return hist_total


def plot_histogramas_globales() -> None:
    """
    Genera y guarda los histogramas globales de ambas clases.

    Produce tres visualizaciones:
        1. Histograma individual COVID-19
        2. Histograma individual Non-COVID-19
        3. Superposición de ambas clases para comparación directa
    """
    histogramas = {}
    stats        = {}

    for clase in CLASES:
        clase_dir = RAW_DIR / clase
        if not clase_dir.exists():
            logger.warning(f"Carpeta no encontrada: {clase_dir}")
            continue

        hist = acumular_histograma(clase_dir)
        histogramas[clase] = hist

        # Estadísticas ponderadas por frecuencia
        bins       = np.arange(256)
        total      = hist.sum()
        media      = float(np.average(bins, weights=hist))
        varianza   = float(np.average((bins - media) ** 2, weights=hist))
        std        = float(np.sqrt(varianza))
        moda       = int(np.argmax(hist))
        stats[clase] = {
            "total_pixeles": int(total),
            "media":         round(media, 1),
            "std":           round(std, 1),
            "moda":          moda,
        }

    if not histogramas:
        logger.error("No se encontraron imágenes.")
        return

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#F8F9FA")

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.45,
        wspace=0.30
    )

    ax1 = fig.add_subplot(gs[0, 0])   # COVID-19
    ax2 = fig.add_subplot(gs[0, 1])   # Non-COVID-19
    ax3 = fig.add_subplot(gs[1, :])   # Superposición

    bins = np.arange(256)

    def estilizar_ax(ax, titulo, color_fondo):
        ax.set_facecolor(color_fondo)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(colors="#555555", labelsize=10)
        ax.set_title(titulo, fontsize=13, fontweight="bold",
                     color="#2C2C2C", pad=12)
        ax.set_xlabel("Valor de intensidad (0 – 255)",
                      fontsize=10, color="#555555")
        ax.set_ylabel("Frecuencia acumulada",
                      fontsize=10, color="#555555")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
                              else f"{x/1e3:.0f}K")
        )

    # ── Histograma COVID-19 ───────────────────────────────────────────────
    if "COVID-19" in histogramas:
        h   = histogramas["COVID-19"]
        s   = stats["COVID-19"]
        col = COLORES["COVID-19"]

        ax1.fill_between(bins, h, alpha=0.55, color=col)
        ax1.plot(bins, h, color=col, linewidth=1.5)
        ax1.axvline(s["media"], color="#8B0000", linewidth=1.5,
                    linestyle="--", label=f"Media: {s['media']}")
        ax1.axvline(s["moda"], color="#CC6600", linewidth=1.2,
                    linestyle=":", label=f"Moda: {s['moda']}")

        estilizar_ax(ax1, "COVID-19 — distribución de intensidades", "#FFF5F5")
        ax1.legend(fontsize=9, framealpha=0.8)

        ax1.text(0.97, 0.92,
                 f"μ = {s['media']}  σ = {s['std']}\n"
                 f"Píxeles totales: {s['total_pixeles']:,}",
                 transform=ax1.transAxes,
                 fontsize=9, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="white", alpha=0.8,
                           edgecolor="#DDDDDD"))

    # ── Histograma Non-COVID-19 ───────────────────────────────────────────
    if "Non-COVID-19" in histogramas:
        h   = histogramas["Non-COVID-19"]
        s   = stats["Non-COVID-19"]
        col = COLORES["Non-COVID-19"]

        ax2.fill_between(bins, h, alpha=0.55, color=col)
        ax2.plot(bins, h, color=col, linewidth=1.5)
        ax2.axvline(s["media"], color="#003366", linewidth=1.5,
                    linestyle="--", label=f"Media: {s['media']}")
        ax2.axvline(s["moda"], color="#006666", linewidth=1.2,
                    linestyle=":", label=f"Moda: {s['moda']}")

        estilizar_ax(ax2, "Non-COVID-19 — distribución de intensidades", "#F0F5FF")
        ax2.legend(fontsize=9, framealpha=0.8)

        ax2.text(0.97, 0.92,
                 f"μ = {s['media']}  σ = {s['std']}\n"
                 f"Píxeles totales: {s['total_pixeles']:,}",
                 transform=ax2.transAxes,
                 fontsize=9, ha="right", va="top",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="white", alpha=0.8,
                           edgecolor="#DDDDDD"))

    # ── Superposición comparativa ─────────────────────────────────────────
    for clase in CLASES:
        if clase not in histogramas:
            continue
        h   = histogramas[clase]
        col = COLORES[clase]
        s   = stats[clase]

        # Normalizar para comparación justa (densidad de probabilidad)
        h_norm = h / h.sum()

        ax3.fill_between(bins, h_norm, alpha=0.35, color=col, label=clase)
        ax3.plot(bins, h_norm, color=col, linewidth=2)
        ax3.axvline(s["media"], color=col, linewidth=1.5,
                    linestyle="--", alpha=0.8)

    estilizar_ax(ax3,
                 "Comparación normalizada — COVID-19 vs Non-COVID-19",
                 "#F9F9F9")
    ax3.set_ylabel("Densidad de probabilidad", fontsize=10, color="#555555")
    ax3.legend(fontsize=11, framealpha=0.9,
               loc="upper right",
               facecolor="white", edgecolor="#DDDDDD")

    # Zonas de interés diagnóstico
    ax3.axvspan(20, 80, alpha=0.07, color="#888888",
                label="_Tejido pulmonar")
    ax3.axvspan(180, 255, alpha=0.07, color="#AAAAAA",
                label="_Tejido denso")
    ax3.text(50,  ax3.get_ylim()[1] * 0.85,
             "Tejido\npulmonar", fontsize=8,
             ha="center", color="#777777")
    ax3.text(220, ax3.get_ylim()[1] * 0.85,
             "Tejido\ndenso", fontsize=8,
             ha="center", color="#777777")

    # ── Título global ─────────────────────────────────────────────────────
    fig.suptitle(
        "Análisis exploratorio — Distribución global de intensidades por clase\n"
        "Dataset CT Teherán · Imágenes crudas (sin preprocesar) · Valor 0 excluido (fondo escáner)",
        fontsize=14, fontweight="bold", color="#1A1A1A", y=0.98
    )

    plt.savefig("eda_histogramas_globales.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    logger.info("Gráfico guardado: eda_histogramas_globales.png")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    plot_histogramas_globales()