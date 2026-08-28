# 🎬 Movie Genre Classification (Deep Learning)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-SVenkataPadmakar%2FDL__Project1-181717.svg?logo=github)](https://github.com/SVenkataPadmakar/DL_Project1)

An end-to-end, production-grade **Deep Neural Network (DNN)** project for multi-class movie genre classification (**Action**, **Comedy**, **Drama**, **Thriller**) based on budget, runtime, rating, release year, and language.

---

## 🌟 Highlights & Features
- **Deep Architecture**: Multi-Layer Perceptron (`256 -> 128 -> 64 -> Softmax(4)`) with ReLU activations and Adam optimizer.
- **Leakage-Free Preprocessing**: `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for categorical language feature, fitted strictly on training data.
- **Diagnostics & Visualizations**: Automatic confusion matrix heatmaps and loss convergence curves saved to [`visualizations/`](visualizations/).
- **Model Serialization**: Trained model and preprocessing pipeline saved to [`saved_models/genre_model.pkl`](saved_models/genre_model.pkl).
- **Interactive Web Studio**: Complete Streamlit application (`app.py`) for dataset exploration and real-time live inference.
- **Automated Tests**: Pytest suite in [`tests/`](tests/) validating data schema, model training, and predictions.

---

## 📁 Repository Structure

```
DL_Project1/
├── movie_genre.csv                   # Dataset
├── train.py                          # Deep Learning training pipeline
├── app.py                            # Interactive Streamlit Web Studio
├── model_utils.py                    # Preprocessing, plotting, and model persistence
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
├── saved_models/                     # Serialized model artifacts
│   └── genre_model.pkl
├── visualizations/                   # Evaluation charts
│   └── genre_evaluation.png
└── tests/                            # Automated test suite
    └── test_genre.py
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/SVenkataPadmakar/DL_Project1.git
cd DL_Project1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Deep Neural Network
```bash
python train.py
```

### 4. Launch the Interactive Web Application
```bash
streamlit run app.py
```

### 5. Run Automated Tests
```bash
pytest -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
