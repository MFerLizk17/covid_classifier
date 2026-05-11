"""
Visualizaciones del entrenamiento y análisis del espacio latente.

Responsabilidades:
    - Gráficas de pérdidas durante el entrenamiento
    - Proyección PCA 2D del espacio latente
    - Heatmaps de atención del Encoder
    - Mapa de zonas de confianza por parche
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from src.model import Autoencoder


# ─────────────────────────────────────────────
# PÉRDIDAS DE ENTRENAMIENTO
# ─────────────────────────────────────────────

def plot_losses(historial: Dict[str, List[float]]) -> None:
    """
    Grafica la evolución de las tres pérdidas durante el entrenamiento.

    Args:
        historial: Diccionario con listas 'total', 'recon' y 'contrast',
                   retornado por trainer.train_one_epoch.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    _plot_single_loss(axes[0], historial["total"],    "#9C27B0", "Loss Total")
    _plot_single_loss(axes[1], historial["recon"],    "#2196F3", "Loss Reconstrucción")
    _plot_single_loss(axes[2], historial["contrast"], "#F44336", "Loss Contrastiva")

    plt.suptitle("Evolución del entrenamiento", fontsize=13)
    plt.tight_layout()
    plt.show()


def _plot_single_loss(
    ax: plt.Axes,
    valores: List[float],
    color: str,
    titulo: str,
) -> None:
    """
    Dibuja una curva de pérdida individual en un eje dado.

    Args:
        ax: Eje de matplotlib donde dibujar.
        valores: Lista de pérdidas por época.
        color: Color hex de la línea.
        titulo: Título del subplot.
    """
    ax.plot(valores, color=color, linewidth=2)
    ax.set_title(titulo)
    ax.set_xlabel("Época")
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────
# PROYECCIÓN PCA DEL ESPACIO LATENTE
# ─────────────────────────────────────────────

def plot_pca(
    todos_z: np.ndarray,
    todos_labels: np.ndarray,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
) -> None:
    """
    Proyecta el espacio latente a 2D con PCA y grafica la separación de clases.

    Transforma también los centroides con el mismo PCA para mostrar
    su posición relativa en la proyección.

    Args:
        todos_z: Todos los vectores Z, forma (N, latent_dim).
        todos_labels: Etiquetas reales, forma (N,).
        centroide_covid: Centroide COVID, forma (latent_dim,).
        centroide_no_covid: Centroide No-COVID, forma (latent_dim,).
    """
    pca  = PCA(n_components=2)
    z_2d = pca.fit_transform(todos_z)

    varianza      = pca.explained_variance_ratio_
    c_covid_2d    = pca.transform(centroide_covid.reshape(1, -1))[0]
    c_nocovid_2d  = pca.transform(centroide_no_covid.reshape(1, -1))[0]

    n_covid    = int((todos_labels == 1).sum())
    n_no_covid = int((todos_labels == 0).sum())

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(
        z_2d[todos_labels == 1, 0], z_2d[todos_labels == 1, 1],
        c="#F44336", alpha=0.5, s=20, label=f"COVID ({n_covid})",
    )
    ax.scatter(
        z_2d[todos_labels == 0, 0], z_2d[todos_labels == 0, 1],
        c="#2196F3", alpha=0.9, s=80, label=f"No-COVID ({n_no_covid})",
    )
    ax.scatter(
        *c_covid_2d, c="#B71C1C", s=400, marker="X",
        label="Centroide COVID", zorder=5,
        edgecolors="white", linewidths=2,
    )
    ax.scatter(
        *c_nocovid_2d, c="#0D47A1", s=400, marker="o",
        label="Centroide No-COVID", zorder=5,
        edgecolors="white", linewidths=2,
    )
    ax.plot(
        [c_covid_2d[0], c_nocovid_2d[0]],
        [c_covid_2d[1], c_nocovid_2d[1]],
        "k--", alpha=0.4, linewidth=1.5, label="Distancia entre centroides",
    )

    ax.set_title(
        f"PCA 2D — Separación de clases\n"
        f"Varianza explicada: {sum(varianza) * 100:.1f}%",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    print(
        f"Varianza explicada: "
        f"{varianza[0]*100:.1f}% + {varianza[1]*100:.1f}% "
        f"= {sum(varianza)*100:.1f}%"
    )


# ─────────────────────────────────────────────
# HEATMAP DE ATENCIÓN DEL ENCODER
# ─────────────────────────────────────────────

def plot_attention_heatmap(
    modelo: Autoencoder,
    dataset_test: Dataset,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    device: str,
    n_por_caso: int = 3,
) -> None:
    """
    Muestra análisis completo por categoría con heatmap de activaciones.

    Para cada imagen muestra 4 paneles:
        1. Imagen original
        2. Heatmap de activaciones del enc4 sobre la imagen
        3. Imagen reconstruida desde Z
        4. Barras de distancia a cada centroide

    Categorías mostradas:
        - COVID clasificado correctamente
        - COVID mal clasificado
        - No-COVID clasificado correctamente
        - No-COVID mal clasificado

    Args:
        modelo: Autoencoder entrenado.
        dataset_test: Subset de test con imágenes y etiquetas.
        centroide_covid: Centroide COVID en espacio latente.
        centroide_no_covid: Centroide No-COVID en espacio latente.
        device: 'cuda' o 'cpu'.
        n_por_caso: Número de ejemplos a mostrar por categoría.
    """
    categorias = _classify_test_cases(
        modelo, dataset_test, centroide_covid, centroide_no_covid, device
    )

    _print_category_summary(categorias)

    grupos = [
        ("COVID → CORRECTO",          "covid_ok",      "#1565C0"),
        ("COVID → MAL CLASIFICADO ⚠️", "covid_error",   "#B71C1C"),
        ("No-COVID → CORRECTO",        "nocovid_ok",    "#1565C0"),
        ("No-COVID → MAL CLASIFICADO ⚠️","nocovid_error","#E65100"),
    ]

    for titulo, clave, color in grupos:
        print(f"\n{'═'*60}\n{titulo}\n{'═'*60}")
        casos = categorias[clave]
        if not casos:
            print("  ✅ Ningún caso en esta categoría")
            continue
        for info in casos[:n_por_caso]:
            _plot_case_analysis(
                info, modelo, dataset_test,
                centroide_covid, centroide_no_covid,
                device, color,
            )


def _classify_test_cases(
    modelo: Autoencoder,
    dataset_test: Dataset,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    device: str,
) -> Dict[str, List[Dict]]:
    """
    Clasifica todos los casos del test y los organiza en 4 categorías.

    Args:
        modelo: Autoencoder entrenado.
        dataset_test: Subset de test.
        centroide_covid: Centroide COVID.
        centroide_no_covid: Centroide No-COVID.
        device: 'cuda' o 'cpu'.

    Returns:
        Diccionario con listas de info por categoría:
        'covid_ok', 'covid_error', 'nocovid_ok', 'nocovid_error'.
    """
    categorias: Dict[str, List[Dict]] = {
        "covid_ok":      [],
        "covid_error":   [],
        "nocovid_ok":    [],
        "nocovid_error": [],
    }

    modelo.eval()
    for idx in range(len(dataset_test)):
        img, label = dataset_test[idx]
        img_t = img.unsqueeze(0).to(device)

        with torch.no_grad():
            _, z = modelo(img_t)

        z_np      = z.cpu().numpy().squeeze()
        d_covid   = float(np.linalg.norm(z_np - centroide_covid))
        d_nocovid = float(np.linalg.norm(z_np - centroide_no_covid))
        pred      = 1 if d_covid < d_nocovid else 0
        dist_total = d_covid + d_nocovid
        confianza  = (
            (d_nocovid / dist_total * 100) if pred == 1
            else (d_covid / dist_total * 100)
        )

        info = {
            "idx": idx, "label": label, "pred": pred,
            "confianza": confianza,
            "d_covid": d_covid, "d_nocovid": d_nocovid,
        }

        if label == 1 and pred == 1:
            categorias["covid_ok"].append(info)
        elif label == 1 and pred == 0:
            categorias["covid_error"].append(info)
        elif label == 0 and pred == 0:
            categorias["nocovid_ok"].append(info)
        else:
            categorias["nocovid_error"].append(info)

    return categorias


def _print_category_summary(categorias: Dict[str, List[Dict]]) -> None:
    """
    Imprime resumen de casos por categoría.

    Args:
        categorias: Diccionario retornado por _classify_test_cases.
    """
    print("═" * 60)
    print("RESUMEN POR CATEGORÍA")
    print("═" * 60)
    print(f"  COVID bien clasificado    : {len(categorias['covid_ok']):>4} casos")
    print(f"  COVID mal clasificado     : {len(categorias['covid_error']):>4} casos")
    print(f"  No-COVID bien clasificado : {len(categorias['nocovid_ok']):>4} casos")
    print(f"  No-COVID mal clasificado  : {len(categorias['nocovid_error']):>4} casos")


def _plot_case_analysis(
    info: Dict,
    modelo: Autoencoder,
    dataset_test: Dataset,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    device: str,
    titulo_color: str,
) -> None:
    """
    Dibuja los 4 paneles de análisis para un caso individual.

    Paneles:
        1. Imagen original TAC
        2. Heatmap de activaciones enc4 superpuesto
        3. Imagen reconstruida desde Z
        4. Barras de distancia a centroides

    Args:
        info: Diccionario con idx, label, pred, confianza, d_covid, d_nocovid.
        modelo: Autoencoder entrenado.
        dataset_test: Subset de test.
        centroide_covid: Centroide COVID.
        centroide_no_covid: Centroide No-COVID.
        device: 'cuda' o 'cpu'.
        titulo_color: Color hex para el título del plot.
    """
    img, label = dataset_test[info["idx"]]
    img_t      = img.unsqueeze(0).to(device)

    # Hook para capturar activaciones de enc4
    activacion: Dict[str, torch.Tensor] = {}

    def hook_enc4(module, input, output):  # noqa: ANN001
        activacion["enc4"] = output.detach()

    handle = modelo.encoder.enc4.register_forward_hook(hook_enc4)

    with torch.no_grad():
        recon, _ = modelo(img_t)
    handle.remove()

    heatmap = _compute_heatmap(activacion["enc4"], target_size=128)

    nombres  = {0: "No-COVID", 1: "COVID"}
    estado   = "✅ CORRECTO" if info["pred"] == label else "❌ ERROR"
    img_np   = img.squeeze().cpu().numpy()
    recon_np = recon.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        f"Real: {nombres[label]} | Pred: {nombres[info['pred']]} | "
        f"Confianza: {info['confianza']:.1f}% | {estado}",
        fontsize=12, color=titulo_color, fontweight="bold",
    )

    # Panel 1 — Original
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Imagen original\n(TAC de entrada)", fontsize=10)
    axes[0].axis("off")

    # Panel 2 — Heatmap
    axes[1].imshow(img_np, cmap="gray")
    hmap_plot = axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[1].set_title(
        "Zonas de atención del Encoder\n(rojo=alta, azul=baja)", fontsize=10
    )
    axes[1].axis("off")
    plt.colorbar(hmap_plot, ax=axes[1], fraction=0.046)

    # Panel 3 — Reconstrucción
    axes[2].imshow(recon_np, cmap="gray")
    axes[2].set_title(
        "Reconstrucción desde Z\n(qué conservó el Encoder)", fontsize=10
    )
    axes[2].axis("off")

    # Panel 4 — Barras de distancia
    _plot_distance_bars(
        axes[3], info["d_covid"], info["d_nocovid"]
    )

    plt.tight_layout()
    plt.show()


def _compute_heatmap(
    enc4_output: torch.Tensor,
    target_size: int,
) -> np.ndarray:
    """
    Calcula un heatmap normalizado desde la salida de enc4.

    Promedia los canales del feature map para obtener un mapa 2D
    y lo redimensiona al tamaño de la imagen original.

    Args:
        enc4_output: Tensor de activaciones (1, 256, H, W).
        target_size: Tamaño final del heatmap (cuadrado).

    Returns:
        Array numpy normalizado en [0, 1], forma (target_size, target_size).
    """
    act     = enc4_output.squeeze().cpu().numpy()   # (256, H, W)
    heatmap = act.mean(axis=0)                       # (H, W)
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    hmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
    return np.array(
        hmap_pil.resize((target_size, target_size), PILImage.BILINEAR)
    ) / 255.0


def _plot_distance_bars(
    ax: plt.Axes,
    d_covid: float,
    d_nocovid: float,
) -> None:
    """
    Dibuja barras de distancia euclidiana a cada centroide.

    La barra más corta indica la clase predicha.
    Una diferencia grande entre barras indica alta confianza.

    Args:
        ax: Eje de matplotlib donde dibujar.
        d_covid: Distancia al centroide COVID.
        d_nocovid: Distancia al centroide No-COVID.
    """
    etiquetas = ["Dist.\nCentroide COVID", "Dist.\nCentroide No-COVID"]
    valores   = [d_covid, d_nocovid]
    colores   = ["#F44336", "#2196F3"]

    bars = ax.bar(etiquetas, valores, color=colores, alpha=0.8, width=0.5)
    ax.set_title(
        "Distancias al centroide\n(barra más corta = clase predicha)",
        fontsize=10,
    )
    ax.set_ylabel("Distancia euclidiana en Z")

    for bar, val in zip(bars, valores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )
    ax.grid(True, alpha=0.3, axis="y")


# ─────────────────────────────────────────────
# ZONAS DE CONFIANZA POR PARCHE
# ─────────────────────────────────────────────

def plot_confidence_zones(
    modelo: Autoencoder,
    dataset_test: Dataset,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    device: str,
    n_mostrar: int = 6,
) -> None:
    """
    Visualiza las zonas de la imagen en las que el modelo confía para decidir.

    Para cada imagen aplica oclusión por parches de 16×16: tapa cada parche
    y mide cuánto cambia la decisión. Las zonas que más cambian la decisión
    son las zonas críticas (verde). Las que no cambian nada son irrelevantes (rojo).

    Args:
        modelo: Autoencoder entrenado.
        dataset_test: Subset de test.
        centroide_covid: Centroide COVID en espacio latente.
        centroide_no_covid: Centroide No-COVID en espacio latente.
        device: 'cuda' o 'cpu'.
        n_mostrar: Número de imágenes del test a analizar.
    """
    modelo.eval()
    nombres = {0: "No-COVID", 1: "COVID"}

    for idx in range(min(n_mostrar, len(dataset_test))):
        img, label = dataset_test[idx]
        img_np     = img.squeeze().cpu().numpy()

        z_orig, d_c_orig, d_n_orig = _get_z_and_distances(
            modelo, img, centroide_covid, centroide_no_covid, device
        )
        pred, confianza = _predict_label(d_c_orig, d_n_orig)

        mapa = _compute_occlusion_map(
            modelo, img_np, centroide_covid, centroide_no_covid,
            d_c_orig, d_n_orig, device, patch_size=16,
        )

        _plot_confidence_panels(
            img_np, mapa, label, pred, confianza, nombres
        )


def _get_z_and_distances(
    modelo: Autoencoder,
    img: torch.Tensor,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, float, float]:
    """
    Obtiene el vector Z y las distancias a centroides para una imagen.

    Args:
        modelo: Autoencoder entrenado.
        img: Tensor imagen (1, H, W).
        centroide_covid: Centroide COVID.
        centroide_no_covid: Centroide No-COVID.
        device: 'cuda' o 'cpu'.

    Returns:
        Tupla (z_numpy, dist_covid, dist_nocovid).
    """
    img_t = img.unsqueeze(0).to(device)
    with torch.no_grad():
        _, z = modelo(img_t)
    z_np = z.cpu().numpy().squeeze()
    return (
        z_np,
        float(np.linalg.norm(z_np - centroide_covid)),
        float(np.linalg.norm(z_np - centroide_no_covid)),
    )


def _predict_label(d_covid: float, d_nocovid: float) -> Tuple[int, float]:
    """
    Retorna predicción y confianza desde distancias a centroides.

    Args:
        d_covid: Distancia al centroide COVID.
        d_nocovid: Distancia al centroide No-COVID.

    Returns:
        Tupla (prediccion, confianza_porcentaje).
    """
    total = d_covid + d_nocovid
    if d_covid < d_nocovid:
        return 1, (d_nocovid / total) * 100.0
    return 0, (d_covid / total) * 100.0


def _compute_occlusion_map(
    modelo: Autoencoder,
    img_np: np.ndarray,
    centroide_covid: np.ndarray,
    centroide_no_covid: np.ndarray,
    d_c_orig: float,
    d_n_orig: float,
    device: str,
    patch_size: int = 16,
) -> np.ndarray:
    """
    Genera un mapa de importancia por oclusión de parches.

    Para cada parche: tapa la región con ceros, re-infiere Z y mide
    cuánto cambian las distancias a los centroides. Mayor cambio = zona
    más importante para la decisión.

    Args:
        modelo: Autoencoder entrenado.
        img_np: Array numpy de la imagen (H, W).
        centroide_covid: Centroide COVID.
        centroide_no_covid: Centroide No-COVID.
        d_c_orig: Distancia original al centroide COVID.
        d_n_orig: Distancia original al centroide No-COVID.
        device: 'cuda' o 'cpu'.
        patch_size: Tamaño del parche cuadrado a ocluir.

    Returns:
        Mapa normalizado [0,1] de forma (H, W) redimensionado desde
        la cuadrícula de parches.
    """
    n_p  = img_np.shape[0] // patch_size
    mapa = np.zeros((n_p, n_p))

    for pi in range(n_p):
        for pj in range(n_p):
            img_mod = img_np.copy()
            img_mod[
                pi * patch_size:(pi + 1) * patch_size,
                pj * patch_size:(pj + 1) * patch_size,
            ] = 0.0

            img_t = (
                torch.tensor(img_mod, dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                _, z_mod = modelo(img_t)

            z_mod_np = z_mod.cpu().numpy().squeeze()
            d_c_mod  = float(np.linalg.norm(z_mod_np - centroide_covid))
            d_n_mod  = float(np.linalg.norm(z_mod_np - centroide_no_covid))
            mapa[pi, pj] = abs(d_c_mod - d_c_orig) + abs(d_n_mod - d_n_orig)

    if mapa.max() > 0:
        mapa /= mapa.max()

    h = img_np.shape[0]
    mapa_pil = PILImage.fromarray((mapa * 255).astype(np.uint8))
    return np.array(mapa_pil.resize((h, h), PILImage.BILINEAR)) / 255.0


def _plot_confidence_panels(
    img_np: np.ndarray,
    mapa: np.ndarray,
    label: int,
    pred: int,
    confianza: float,
    nombres: Dict[int, str],
) -> None:
    """
    Dibuja 3 paneles: original, mapa de confianza y zonas críticas.

    Paneles:
        1. Imagen original
        2. Mapa RdYlGn superpuesto (verde=confía, rojo=ignora)
        3. Solo zonas con confianza > 50%

    Args:
        img_np: Array numpy de la imagen (H, W).
        mapa: Mapa de importancia normalizado (H, W).
        label: Etiqueta real (0 o 1).
        pred: Predicción del modelo (0 o 1).
        confianza: Confianza en porcentaje.
        nombres: Diccionario {0: 'No-COVID', 1: 'COVID'}.
    """
    estado = "✅" if pred == label else "❌"
    color  = "#1565C0" if pred == label else "#B71C1C"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"{estado}  Real: {nombres[label]}  |  "
        f"Pred: {nombres[pred]}  |  "
        f"Confianza: {confianza:.1f}%",
        fontsize=12, color=color, fontweight="bold",
    )

    # Panel 1 — Original
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Imagen original", fontsize=11)
    axes[0].axis("off")

    # Panel 2 — Mapa de confianza
    axes[1].imshow(img_np, cmap="gray", alpha=0.5)
    conf_plot = axes[1].imshow(mapa, cmap="RdYlGn", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title(
        "VERDE = confía aquí\nROJO = no usa esta zona", fontsize=10
    )
    axes[1].axis("off")
    plt.colorbar(
        conf_plot, ax=axes[1], fraction=0.046,
        ticks=[0, 0.5, 1], label="No confía ← → Confía",
    )

    # Panel 3 — Zonas críticas (>50%)
    mascara = mapa.copy()
    mascara[mascara < 0.5] = 0.0

    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(mascara, cmap="Greens", alpha=0.7, vmin=0, vmax=1)
    axes[2].set_title(
        "Zonas críticas (>50%)\ndonde el modelo decide", fontsize=10
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()