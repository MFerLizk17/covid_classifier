"""
Dataset para cargar imágenes TAC preprocesadas desde disco.

Espera la siguiente estructura de directorios:
    data/processed/
    ├── covid/
    │   ├── imagen1.png
    │   └── ...
    └── non_covid/
        ├── imagen1.png
        └── ...
"""

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class TomografiaDataset(Dataset):
    """
    Carga imágenes TAC ya preprocesadas con sus etiquetas.

    Etiquetas:
        1 → COVID-19
        0 → Non-COVID-19

    Args:
        ruta_covid: Directorio con imágenes COVID-19.
        ruta_no_covid: Directorio con imágenes Non-COVID-19.
        img_size: Tamaño al que se redimensiona cada imagen (cuadrada).
    """

    def __init__(
        self,
        ruta_covid: Path,
        ruta_no_covid: Path,
        img_size: int = 128,
    ) -> None:
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        covid_imgs    = self._collect_images(ruta_covid,    label=1)
        no_covid_imgs = self._collect_images(ruta_no_covid, label=0)

        self.datos: List[Tuple[str, int]] = covid_imgs + no_covid_imgs

    def _collect_images(self, directory: Path, label: int) -> List[Tuple[str, int]]:
        """
        Recolecta todas las rutas de imágenes válidas en un directorio.

        Args:
            directory: Carpeta a escanear.
            label: Etiqueta numérica a asignar (1=COVID, 0=No-COVID).

        Returns:
            Lista de tuplas (ruta_str, label).
        """
        if not directory.exists():
            raise FileNotFoundError(
                f"Directorio no encontrado: {directory}\n"
                "Ejecuta primero: python main.py --mode preprocess"
            )
        return [
            (str(f), label)
            for f in sorted(directory.iterdir())
            if f.suffix.lower() in SUPPORTED_EXT
        ]

    def __len__(self) -> int:
        """Retorna el número total de imágenes en el dataset."""
        return len(self.datos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Carga y transforma una imagen por índice.

        Args:
            idx: Índice de la muestra.

        Returns:
            Tupla (tensor imagen [1, H, W], etiqueta).
        """
        ruta, label = self.datos[idx]
        img = Image.open(ruta).convert("RGB")
        return self.transform(img), label