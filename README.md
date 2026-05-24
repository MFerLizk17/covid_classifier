# 🫁 Clasificador COVID-19 por TAC — Pipeline CT

Sistema de clasificación de tomografías computarizadas de tórax para detección de COVID-19, basado en un Autoencoder convolucional con pérdida contrastiva.

---

## 📋 Tabla de contenidos

- [Requisitos del sistema](#requisitos-del-sistema)
- [Instalación](#instalación)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Preparación del dataset](#preparación-del-dataset)
- [Uso del sistema](#uso-del-sistema)
- [Flujo completo recomendado](#flujo-completo-recomendado)
- [Modo predict — Interfaz CLI](#modo-predict--interfaz-cli)
- [Solución de errores comunes](#solución-de-errores-comunes)

---

## Requisitos del sistema

| Requisito | Versión mínima |
|---|---|
| Python | 3.11 |
| CUDA (opcional) | 11.8+ |
| cuDNN (opcional) | 8.x+ |
| RAM | 8 GB |
| Espacio en disco | 5 GB |

> El sistema detecta automáticamente si hay GPU disponible. Si no hay CUDA corre en CPU sin ningún cambio en el código.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/MFerLizk17/covid_classifier.git
cd covid_classifier
```

### 2. Crear entorno virtual

```bash
python -m venv venv
```

Activar el entorno:

- **Windows:**
```bash
venv\Scripts\activate
```

- **Linux / Mac:**
```bash
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Si estás en Linux y tienes conflictos con paquetes del sistema:

```bash
pip install -r requirements.txt --break-system-packages
```

### 4. Verificar instalación de PyTorch con CUDA

```bash
python -c "import torch; print(torch.__version__); print('CUDA disponible:', torch.cuda.is_available())"
```

Si CUDA no aparece disponible pero tienes GPU, instala PyTorch con soporte CUDA manualmente según tu versión:

```bash
# Para CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Estructura del proyecto

```
covid_classifier/
├── data/
│   ├── raw/                  ← imágenes originales sin procesar
│   │   ├── covid/
│   │   └── non_covid/
│   └── processed/            ← imágenes preprocesadas (generado automáticamente)
│       ├── covid/
│       └── non_covid/
├── src/
│   ├── __init__.py
│   ├── config.py             ← hiperparámetros centralizados
│   ├── preprocesamiento.py   ← pipeline CLAHE + NLM + segmentación pulmonar
│   ├── dataset.py            ← carga de imágenes para el modelo
│   ├── model.py              ← arquitectura Autoencoder (Encoder + Decoder)
│   ├── loss.py               ← ContrastiveLoss
│   ├── trainer.py            ← ciclo de entrenamiento y centroides
│   ├── evaluator.py          ← métricas y clasificación
│   └── visualizer.py        ← gráficas de pérdidas, PCA, heatmaps
├── tests/
│   ├── __init__.py
│   └── test_preprocessor.py
├── main.py                   ← punto de entrada principal
├── requirements.txt
└── README.md
```

---

## Preparación del dataset

Las imágenes crudas deben colocarse en las siguientes carpetas **antes de correr cualquier comando**:

```
data/raw/covid/        ← imágenes TAC de pacientes COVID-19
data/raw/non_covid/    ← imágenes TAC de pacientes No-COVID
```

Formatos soportados: `.png`, `.jpg`, `.jpeg`, `.bmp`

> Si las carpetas no existen, créalas manualmente antes de continuar.

---

## Uso del sistema

El sistema tiene cuatro modos de ejecución que se pasan como argumento:

```
python main.py --mode [preprocess | train | evaluate | predict]
```

### `--mode preprocess`
Preprocesa todas las imágenes crudas y las guarda en `data/processed/`.

```bash
python main.py --mode preprocess
```

Al finalizar imprime un reporte con total de imágenes procesadas, descartadas y tiempos.

---

### `--mode train`
Entrena el Autoencoder con las imágenes preprocesadas. Genera dos archivos:
- `autoencoder_covid.pth` — pesos del modelo
- `centroides_covid.pkl` — centroides de cada clase en el espacio latente

```bash
python main.py --mode train
```

> ⚠ Requiere haber corrido `--mode preprocess` primero.

---

### `--mode evaluate`
Carga el modelo entrenado y lo evalúa sobre el conjunto de prueba. Muestra métricas de clasificación, AUC-ROC y análisis del espacio latente.

```bash
python main.py --mode evaluate
```

> ⚠ Requiere haber corrido `--mode train` primero.

---

### `--mode predict`
Abre la interfaz CLI interactiva para clasificar una imagen individual.

```bash
python main.py --mode predict
```

También acepta una ruta directa:

```bash
python main.py --mode predict --image data/processed/covid/imagen.png
```

> ⚠ Requiere haber corrido `--mode train` primero.

Al usar el modo predict, la imagen cruda y nueva será procesada con el mismo pipeline del modo preprocess, pasa por una revisión de ruido y después una revisión de área de información para verificar que sea una imagen apta para que pase al modelo y haga la clasificación. 

---

## Flujo completo recomendado

Estos cuatro comandos deben correrse **en orden**, una sola vez:

```bash
# Paso 1 — preprocesar dataset
python main.py --mode preprocess

# Paso 2 — entrenar modelo
python main.py --mode train

# Paso 3 — evaluar rendimiento
python main.py --mode evaluate

# Paso 4 — clasificar imágenes nuevas
python main.py --mode predict
```

Una vez completados los pasos 1 y 2, solo se necesita el paso 4 para clasificar imágenes nuevas.

---

## Modo predict — Interfaz CLI

Al correr `--mode predict` sin `--image`, el sistema muestra un menú interactivo:

```
=======================================================
  CLASIFICADOR COVID-19 — SELECCIÓN DE IMAGEN
=======================================================

  Imágenes disponibles en data/processed/:

  [  1] covid/ct_001.png
  [  2] covid/ct_002.png
  [  3] non_covid/ct_100.png
  ...

  [  0] Ingresar ruta manual
-------------------------------------------------------

  Selecciona el número de la imagen (0 para ruta manual):
```

**Para usar una imagen externa al dataset**, selecciona `0` e ingresa la ruta completa:

```
Ruta: C:/Users/usuario/Desktop/tomografia.png
```

> En Windows usar `/` en lugar de `\` para evitar errores con espacios en la ruta.

El sistema muestra:
1. La imagen seleccionada → cierra la ventana para iniciar el análisis
2. Reporte en terminal con probabilidades COVID/No-COVID
3. Ventana con 4 paneles: imagen original, zona diagnóstica, reconstrucción y barras de probabilidad

> En linux 

El sistema guardará 3 tipo de imágenes en /data/: 
1. preview.png -> Permite ver la imagen que entrar al modelo para ser clasificada
2. rechazo_preprocesado.png -> En dado caso que no cumpla las dos condiciones (ruido y área de información) mostrará la información para tener más claridad del rechazo
3. resultado_clasificacion.png -> Si la imagen es aprobada, pasa por la predicción y dará información de porque esa decisión


---

## Solución de errores comunes

**`No se encontró el modelo en 'autoencoder_covid.pth'`**
```bash
# Solución: entrenar el modelo primero
python main.py --mode train
```

---

**`FileNotFoundError: data/processed/covid`**
```bash
# Solución: correr el preprocesamiento primero
python main.py --mode preprocess
```

---

**`Import "torch" could not be resolved` en VS Code**

No es un error, es una advertencia de Pylance. Solución:
1. `Ctrl + Shift + P`
2. Buscar **Python: Select Interpreter**
3. Seleccionar el entorno `venv` del proyecto

---

**CUDA disponible: False con GPU NVIDIA**

Reinstalar PyTorch con soporte CUDA según tu versión:
```bash
# Verificar versión de CUDA instalada
nvidia-smi

# Instalar PyTorch con CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Instalar PyTorch con CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

**`⚠ No se encontró el archivo` al ingresar ruta manual**

En Windows usar barras `/` en lugar de `\` y sin comillas:
```
# Correcto
C:/Users/usuario/Desktop/imagen.png

# Incorrecto
"C:\Users\usuario\Desktop\imagen.png"
```

---

**Entrenamiento muy lento en CPU**

Es normal. Para acelerar reducir `EPOCHS` en `src/config.py`:
```python
EPOCHS: int = 20   # default: 50
```

---

## Configuración avanzada

Todos los hiperparámetros están centralizados en `src/config.py`. Los más relevantes:

| Parámetro | Default | Descripción |
|---|---|---|
| `IMG_SIZE` | 128 | Tamaño de entrada al modelo |
| `LATENT_DIM` | 256 | Dimensión del espacio latente |
| `BATCH_SIZE` | 16 | Tamaño de batch |
| `EPOCHS` | 50 | Épocas de entrenamiento |
| `LR` | 1e-3 | Learning rate inicial |
| `LAMBDA` | 0.1 | Peso de la pérdida contrastiva |

Modificar cualquier valor en `config.py` afecta todo el sistema sin tocar otro archivo.
