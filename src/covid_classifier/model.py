"""
Arquitectura del Autoencoder convolucional para clasificación COVID/No-COVID.

Flujo:
    Imagen (1, 128, 128)
        → Encoder → Vector Z (LATENT_DIM,)
        → Decoder → Imagen reconstruida (1, 128, 128)

El Encoder entrenado se usa en inferencia para proyectar imágenes
al espacio latente y clasificar por distancia al centroide.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Bloque convolucional doble: Conv → BN → ReLU → Conv → BN → ReLU.

    Es la unidad básica tanto del Encoder como referencia estructural
    del Decoder. Usa padding=1 para mantener dimensiones espaciales
    dentro del bloque.

    Args:
        in_ch: Canales de entrada.
        out_ch: Canales de salida.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pasa el tensor por el bloque convolucional doble.

        Args:
            x: Tensor de entrada (B, in_ch, H, W).

        Returns:
            Tensor de salida (B, out_ch, H, W).
        """
        return self.block(x)


class Encoder(nn.Module):
    """
    Encoder CNN: Imagen (1, 128, 128) → Vector Z (LATENT_DIM,).

    Aplica 4 niveles de downsampling con MaxPool2d(2,2):
        128×128 → 64×64 → 32×32 → 16×16 → 8×8

    El feature map final (256, 8, 8) se aplana y proyecta al
    espacio latente mediante una capa lineal + BatchNorm1d.

    Args:
        latent_dim: Dimensión del vector Z de salida.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()

        self.enc1 = ConvBlock(1,   32)
        self.enc2 = ConvBlock(32,  64)
        self.enc3 = ConvBlock(64,  128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

        # Tras 4 poolings: 128 / 2^4 = 8 → feature map 256×8×8
        self.flatten   = nn.Flatten()
        self.fc_latent = nn.Linear(256 * 8 * 8, latent_dim)
        self.bn_latent = nn.BatchNorm1d(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica una imagen en su vector latente Z.

        Args:
            x: Tensor (B, 1, 128, 128).

        Returns:
            Vector latente Z de forma (B, latent_dim).
        """
        x = self.pool(self.enc1(x))   # → (B, 32,  64, 64)
        x = self.pool(self.enc2(x))   # → (B, 64,  32, 32)
        x = self.pool(self.enc3(x))   # → (B, 128, 16, 16)
        x = self.pool(self.enc4(x))   # → (B, 256,  8,  8)
        x = self.flatten(x)           # → (B, 256*8*8)
        z = self.bn_latent(self.fc_latent(x))  # → (B, latent_dim)
        return z


class Decoder(nn.Module):
    """
    Decoder CNN: Vector Z (LATENT_DIM,) → Imagen reconstruida (1, 128, 128).

    Invierte el Encoder mediante ConvTranspose2d con stride=2:
        8×8 → 16×16 → 32×32 → 64×64 → 128×128

    La activación Sigmoid en la salida garantiza valores en [0, 1],
    compatibles con la pérdida MSE frente a imágenes normalizadas.

    Args:
        latent_dim: Dimensión del vector Z de entrada.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        self.dec = nn.Sequential(
            # 8×8 → 16×16
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 64×64 → 128×128
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruye una imagen desde su vector latente.

        Args:
            z: Vector latente (B, latent_dim).

        Returns:
            Imagen reconstruida (B, 1, 128, 128).
        """
        x = self.fc(z)
        x = x.view(-1, 256, 8, 8)
        return self.dec(x)


class Autoencoder(nn.Module):
    """
    Autoencoder completo: Encoder + Decoder.

    Durante el entrenamiento retorna la imagen reconstruida y el
    vector Z para calcular la pérdida combinada.
    En inferencia solo se usa el Encoder para obtener Z y
    clasificar por distancia al centroide.

    Args:
        latent_dim: Dimensión del espacio latente compartido.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica y reconstruye un batch de imágenes.

        Args:
            x: Tensor de entrada (B, 1, 128, 128).

        Returns:
            Tupla (imagen_reconstruida, vector_z), ambos tensores.
        """
        z            = self.encoder(x)
        reconstruida = self.decoder(z)
        return reconstruida, z