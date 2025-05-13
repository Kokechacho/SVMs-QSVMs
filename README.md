# SVMs vs QSVMs: Experimental Comparison

Este repositorio contiene un conjunto de experimentos diseñados para comparar el rendimiento de máquinas de vectores de soporte clásicas (SVMs) con variantes cuánticas (QSVMs). Se han considerado múltiples kernels, datasets y métricas, replicando una metodología rigurosa basada en 10-fold cross-validation repetida varias veces.

---

## 📁 Estructura del proyecto

```
SVMS_EXP/
├── Analysis/              # Scripts para graficar y analizar resultados
│   ├── plots.py
├── Kernels/               # Definiciones de kernels clásicos y cuánticos
│   ├── all_kernels.py
│   └── custom.py
│   └── base.py
├── results/               # CSVs y gráficos generados automáticamente
│   ├── plots/
│   └── ...
├── Data/              # Carga de datasets y preprocesamiento
│   └── load_data.py
│   └── libsvm/
├── Model/              # Ejecución y entreno de las SVMs
│   └── train.py
│   └── evaluate.py
├── experiments.py         # Script principal para ejecutar todos los experimentos
├── config.py              # Configuración general (datasets, kernels, etc.)
├── requirements.txt       # Dependencias necesarias (aún por hacer)
└── README.md              # Este archivo
```

---

## ⚙️ Requisitos

- Python 3.9+
- pip

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

Las librerías necesarias incluyen:

- `scikit-learn`
- `matplotlib`
- `pandas`
- `numpy`

---

## 🚀 Cómo ejecutar los experimentos

Desde la raíz del proyecto:

```bash
python experiments.py
```

Esto realizará los siguientes pasos:

1. Cargar los datasets.
2. Ejecutar múltiples ensayos con particiones 10-fold diferentes.
3. Entrenar y evaluar SVMs y QSVMs.
4. Guardar métricas, tiempos de ejecución y vectores de soporte.
5. Generar gráficos y tablas comparativas.

Los resultados se almacenan automáticamente en la carpeta `results/`.

---

## 📊 Qué métricas se calculan

- **Accuracy**
- **F1-score**
- **Número de vectores de soporte (o su proporción con respecto al accuracy)**
- **Tiempo de entrenamiento**
- Diferencias de accuracy entre kernels

También se visualizan:

- **Gráficos de barras comparativos**
- **Tablas de diferencias de precisión**

---

## 🧪 Añadir un nuevo kernel

1. Abre `Kernels/classical_kernels.py` o `Kernels/quantum_kernels.py`.
2. Define tu función de kernel personalizada.
3. Añádela al diccionario de kernels en `config.py`.

---

## 📁 Añadir un nuevo dataset

1. Coloca el archivo en `Datasets/` o modifícalo en `load_data.py`.
2. Añade el nombre del dataset a la lista `DATASETS` en `config.py`.

---

## 🧠 Metodología

Cada experimento sigue esta estructura:

- Cada ensayo utiliza una partición distinta de 10-fold cross-validation.
- Se evalúan múltiples kernels sobre los mismos datos.
- Se calcula la media y desviación estándar de cada métrica. (accuracy, f-score...etc)

Este enfoque sigue la metodología descrita en el paper de referencia.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles. (No hay, habria que añadirla quizas)

---

## ✍️ Autor

Álvaro — [@Kokechacho](https://github.com/Kokechacho)  
Proyecto desarrollado en el marco de la Beca de Investigación para el proyecto " Quantum Support Vector Machines y optimización en Quantum Machine Learning (QML)" de la UAH.

---

## 📚 Referencias


