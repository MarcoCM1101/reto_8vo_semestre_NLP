# Detección de Contenido Relacionado con Anorexia usando Machine Learning

Este proyecto tiene como objetivo identificar publicaciones relacionadas con trastornos alimenticios, específicamente la anorexia, utilizando técnicas de **procesamiento de lenguaje natural (NLP)** y **aprendizaje automático**.

## 📌 Descripción

Se parte de una base de datos de textos (tweets), con los cuales se construye un modelo capaz de distinguir entre contenido “Control” y contenido asociado a “Anorexia”. Para ello se utilizan:

- Limpieza y normalización del texto.
- Vectorización mediante **TF-IDF**.
- Entrenamiento de un modelo **Random Forest** optimizado con búsqueda en malla (GridSearchCV).
- Visualización de resultados usando curvas ROC, matrices de confusión y análisis de n-gramas más relevantes.
- Evaluación en un conjunto de validación externo.

## ⚙️ Tecnologías utilizadas

- Python
- pandas, numpy
- sklearn
- matplotlib
- gensim, nltk
- joblib

## 📊 Resultados

### Conjunto de Prueba

- **Accuracy**: 0.90
- **AUC**: 0.96
- **Precision (Anorexia)**: 0.87
- **Recall (Anorexia)**: 0.94

### Conjunto de Validación Externa

- **Accuracy**: 0.84
- **AUC**: 0.93
- **Precision (Anorexia)**: 0.83
- **Recall (Anorexia)**: 0.90

### Visualizaciones

- **Curva ROC**: muestra alto rendimiento (AUC ≈ 0.96)
- **Matriz de confusión**: revela buen equilibrio entre clases.
- **Top n-gramas**: se identifican palabras clave como "anorexia", "proana", "thinspo", "gym", etc.

## 💡 Propuesta de mejora

En este proyecto se propone migrar a un enfoque basado en **embeddings** (Word2Vec, FastText, etc.) para capturar relaciones semánticas más profundas entre palabras, mejorando así la identificación contextual de mensajes relacionados con la anorexia.

## 📁 Estructura de archivos

```
├── data/
│   ├── data_train.csv
│   ├── data_test_fold1 - anorexia(in).csv
│   └── models/
│       ├── modelo_anorexia.pkl
│       └── vectorizador_tfidf.pkl
├── notebooks/
│   └── entrenamiento_modelo.ipynb
├── README.md
```

## 🚀 Cómo ejecutar

1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Ejecutar el script de entrenamiento o notebook.

3. Para validar sobre nuevos datos:
   ```python
   clf = joblib.load('modelo_anorexia.pkl')
   vectorizer = joblib.load('vectorizador_tfidf.pkl')
   ```

## ✍️ Autores

A01753729 Marco Antonio Caudillo Morales

A01747327 Jorge Daniel Rea Prado

A01753911 Oswaldo Daniel Hernández De Luna

A01754412 Adolfo Sebastian Gonzalez Mora
