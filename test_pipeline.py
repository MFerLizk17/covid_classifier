import cv2
import numpy as np
from pathlib import Path
from src.preprocesamiento import run_pipeline

clases     = ["COVID-19", "Non-COVID-19"]
raw_dir    = Path("./data/raw")
output_dir = Path("./data/test_procesadas")

for clase in clases:
    clase_dir = raw_dir / clase
    out_dir   = output_dir / clase
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [
        f for f in clase_dir.iterdir()
        if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    rng    = np.random.default_rng(seed=2000)
    sample = [images[i] for i in rng.choice(len(images), size=5, replace=False)]

    print(f"\n[{clase}] Procesando 5 imágenes de prueba...")
    ok          = 0
    desc_calidad = 0
    desc_mascara = 0

    for img_path in sample:
        result, motivo = run_pipeline(img_path)
        if result is not None:
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), (result * 255).astype("uint8"))
            print(f"  ✓ {img_path.name} → shape: {result.shape} dtype: {result.dtype}")
            ok += 1
        elif motivo == "calidad":
            print(f"  ✗ {img_path.name} descartada por SNR bajo")
            desc_calidad += 1
        elif motivo == "mascara":
            print(f"  ✗ {img_path.name} descartada por máscara vacía")
            desc_mascara += 1

    print(f"  Resultado: {ok} procesadas, {desc_calidad} por calidad, {desc_mascara} por máscara")

print("\nImágenes guardadas en data/test_procesadas/")