# 🎬 Movie Genre Classification (Artificial Neural Network - ANN)

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Repository](https://img.shields.io/badge/GitHub-SVenkataPadmakar%2FDL__Project1-181717.svg?logo=github)](https://github.com/SVenkataPadmakar/DL_Project1)

An end-to-end, production-grade **Artificial Neural Network (ANN)** deep learning project built in **PyTorch** for multi-class movie genre classification (**Action**, **Comedy**, **Drama**, **Thriller**) based on budget, runtime, rating, release year, and language.

---

## 🌟 Highlights & Features
- **Deep ANN Architecture**: Multi-Layer PyTorch Neural Network (`Dense(256) -> Dense(128) -> Dense(64) -> Softmax(4)`) with ReLU activations, Adam optimization, and CrossEntropyLoss backpropagation.
- **Leakage-Free Preprocessing**: `ColumnTransformer` with `StandardScaler` for continuous numeric features and `OneHotEncoder` for categorical language feature, fitted strictly on training data.
- **Diagnostics & Visualizations**: Automatic confusion matrix heatmaps and PyTorch epoch loss convergence curves saved to [`visualizations/`](visualizations/).
- **Model Serialization**: Trained PyTorch ANN model artifact and preprocessing pipeline saved to [`saved_models/genre_model.pkl`](saved_models/genre_model.pkl).
- **Interactive Web Studio**: Full-featured Streamlit application (`app.py`) for dataset exploration, interactive ANN training with hyperparameter tuning, and real-time live inference.
- **Automated Tests**: Pytest suite in [`tests/`](tests/) validating data schema, PyTorch ANN training, and predictions.

---

## 📁 Repository Structure

```
Project1_Movie_Genre_Classification/
├── movie_genre.csv                   # Dataset
├── train.py                          # PyTorch ANN training pipeline
├── app.py                            # Interactive Streamlit Web Studio
├── model_utils.py                    # PyTorch ANN classifier, preprocessing & plotting
├── requirements.txt                  # Python dependencies (includes torch)
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

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the PyTorch Artificial Neural Network (ANN)
```bash
python train.py
```

### 3. Launch the Interactive Web Application
```bash
streamlit run app.py
```

### 4. Run Automated Tests
```bash
pytest -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
