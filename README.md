# 🎬 Project 1: Movie Genre Classification (Deep Learning)

A standalone **Deep Neural Network (DNN)** project for predicting movie genres (Action, Comedy, Drama, Thriller) based on budget, runtime, rating, release year, and language.

---

## 📌 Features & Architecture
- **Multi-Layer Perceptron (MLP)**: `Input -> Dense(256) -> Dense(128) -> Dense(64) -> Dense(4, Softmax)`
- **Preprocessing Pipeline**: `StandardScaler` for numeric features and `OneHotEncoder` for language feature, strictly fitted on training splits to prevent data leakage.
- **Model Persistence**: Serialized pipeline and neural network saved to `saved_models/genre_model.pkl`.
- **Visual Diagnostics**: Loss convergence and confusion matrix heatmaps exported to `visualizations/genre_evaluation.png`.
- **Interactive UI**: Complete Streamlit application (`app.py`) for live dataset exploration and real-time inference.

---

## 🚀 Quick Start

### 1. Train the Deep Neural Network
```bash
python train.py
```

### 2. Launch Interactive Streamlit App
```bash
streamlit run app.py
```

### 3. Run Automated Tests
```bash
pytest -v tests/test_genre.py
```
