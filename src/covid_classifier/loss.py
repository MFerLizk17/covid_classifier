"""
Función de pérdida contrastiva para separación de clases en el espacio latente.

La pérdida combinada del entrenamiento es:
    loss = MSE(reconstruida, original) + LAMBDA * ContrastiveLoss(Z, labels)

La ContrastiveLoss actúa como regularizador geométrico: obliga al Encoder
a organizar el espacio latente de forma que imágenes de la misma clase
queden agrupadas y clases distintas queden separadas por al menos MARGIN.
"""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Pérdida contrastiva por pares para separación de clases en espacio latente.

    Para cada par (i, j) dentro del batch:
        - Misma clase   → minimiza distancia²
        - Clase distinta → maximiza distancia hasta el margen:
                           penaliza si dist < MARGIN

    Fórmula por par:
        L(i,j) = y · d²  +  (1-y) · max(0, margin - d)²

    donde:
        d = ||z_i - z_j||₂  (distancia euclidiana)
        y = 1 si misma clase, 0 si distinta

    Args:
        margin: Distancia mínima deseada entre clases distintas.
                Con vectores normalizados por BatchNorm y LATENT_DIM=256,
                margin=2.0 es un objetivo razonable sin forzar separación
                excesiva.
    """

    def __init__(self, margin: float = 2.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula la pérdida contrastiva promedio sobre todos los pares del batch.

        Args:
            z: Vectores latentes del batch, forma (B, latent_dim).
            labels: Etiquetas del batch, forma (B,). Valores: 0 o 1.

        Returns:
            Escalar con la pérdida promedio sobre todos los pares.
        """
        batch_size = z.shape[0]
        loss  = torch.tensor(0.0, device=z.device)
        pares = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                distancia   = torch.norm(z[i] - z[j])
                misma_clase = (labels[i] == labels[j]).float()

                loss_par = (
                    misma_clase * distancia ** 2
                    + (1 - misma_clase)
                    * torch.clamp(self.margin - distancia, min=0.0) ** 2
                )
                loss  = loss + loss_par
                pares += 1

        return loss / pares if pares > 0 else loss