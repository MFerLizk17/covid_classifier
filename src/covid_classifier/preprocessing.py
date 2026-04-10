import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import kagglehub
from skimage.restoration import estimate_sigma

# ==============================
# 📥 DESCARGA DATASET
# ==============================

def download_dataset() -> str:
    """
    Descarga el dataset desde Kaggle usando kagglehub.
    """
    path = kagglehub.dataset_download(
        "mehradaria/covid19-lung-ct-scans"
    )
    print(f"Dataset descargado en: {path}")
    return path


# ==============================
# 🧠 FILTRO DE CALIDAD
# ==============================

class ImageQualityFilter:

    def compute_snr(self, image: np.ndarray) -> float:
        mean = np.mean(image)
        std = np.std(image)
        return float(mean / (std + 1e-8))

    def compute_blur(self, image: np.ndarray) -> float:
        lap = cv2.Laplacian(image, cv2.CV_64F)
        return float(lap.var())

    def compute_entropy(self, image: np.ndarray) -> float:
        hist = cv2.calcHist([image.astype(np.uint8)], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist + 1e-8
        return float(-np.sum(hist * np.log2(hist)))

    def detect_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        artifacts = {}

        # saturación (metal / outliers)
        artifacts["saturation"] = float(np.sum(image > 250) / image.size)

        # ruido estimado
        artifacts["noise_sigma"] = float(estimate_sigma(image, channel_axis=None))

        return artifacts

    def is_valid(self, image: np.ndarray, thresholds: Dict) -> Tuple[bool, Dict]:
        metrics = {}

        snr = self.compute_snr(image)
        blur = self.compute_blur(image)
        entropy = self.compute_entropy(image)
        artifacts = self.detect_artifacts(image)

        metrics.update({
            "snr": snr,
            "blur": blur,
            "entropy": entropy,
            **artifacts
        })

        valid = True
        reasons = []

        if snr < thresholds["snr"]:
            valid = False
            reasons.append("low_snr")

        if blur < thresholds["blur"]:
            valid = False
            reasons.append("blur")

        if not (thresholds["entropy_min"] <= entropy <= thresholds["entropy_max"]):
            valid = False
            reasons.append("entropy")

        if artifacts["saturation"] > thresholds["saturation"]:
            valid = False
            reasons.append("saturation")

        metrics["reason"] = "|".join(reasons)
        return valid, metrics


# ==============================
# 🧪 PREPROCESAMIENTO
# ==============================

class Preprocessor:

    def windowing(self, image: np.ndarray, wl: int = -600, ww: int = 1500) -> np.ndarray:
        """
        Simulación de windowing para imágenes PNG (aproximado)
        """
        min_val = wl - ww // 2
        max_val = wl + ww // 2

        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
        return image

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)

    def denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.bilateralFilter(image.astype(np.float32), 7, 50, 50)

    def clahe(self, image: np.ndarray) -> np.ndarray:
        image = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(image) / 255.0

    def resize(self, image: np.ndarray, size: int = 224) -> np.ndarray:
        h, w = image.shape
        scale = size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image_resized = cv2.resize(image, (new_w, new_h))

        pad_h = size - new_h
        pad_w = size - new_w

        image_padded = np.pad(
            image_resized,
            ((pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w // 2)),
            mode='constant'
        )

        return image_padded

    def process(self, image: np.ndarray) -> np.ndarray:
        image = self.windowing(image)
        image = self.denoise(image)
        image = self.clahe(image)
        image = self.normalize(image)
        image = self.resize(image)

        # expandir canal (H,W) → (H,W,1)
        image = np.expand_dims(image, axis=-1)

        return image


# ==============================
# 🚀 PIPELINE PRINCIPAL
# ==============================

def run_pipeline():
    dataset_path = download_dataset()

    output_dir = "/home/fernanda_lizcano/covid_classifier/data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    valid_dir = os.path.join(output_dir, "valid")
    reject_dir = os.path.join(output_dir, "rejected")

    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(reject_dir, exist_ok=True)

    log_data = []

    iqf = ImageQualityFilter()
    pre = Preprocessor()

    thresholds = {
        "snr": 8,
        "blur": 100,
        "entropy_min": 4,
        "entropy_max": 7,
        "saturation": 0.02
    }

    # buscar imágenes recursivamente
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            path = os.path.join(root, file)

            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

                valid, metrics = iqf.is_valid(img, thresholds)

                if valid:
                    processed = pre.process(img)
                    save_path = os.path.join(valid_dir, file)
                    cv2.imwrite(save_path, (processed * 255).astype(np.uint8))
                else:
                    save_path = os.path.join(reject_dir, file)
                    cv2.imwrite(save_path, img)

                log_data.append({
                    "file": file,
                    "valid": valid,
                    **metrics
                })

            except Exception as e:
                log_data.append({
                    "file": file,
                    "valid": False,
                    "error": str(e)
                })

    # guardar log
    df = pd.DataFrame(log_data)
    df.to_csv(os.path.join(output_dir, "log.csv"), index=False)

    print("Pipeline completado 🚀")


# ==============================
# ▶️ RUN
# ==============================

if __name__ == "__main__":
    run_pipeline()