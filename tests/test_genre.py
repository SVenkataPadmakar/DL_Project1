import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = PROJECT_DIR.parent

for p in [str(PROJECT_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd

try:
    from Project1_Movie_Genre_Classification.train import train_model
    from Project1_Movie_Genre_Classification.model_utils import (
        load_model, VISUALIZATIONS_DIR, PyTorchANNClassifier
    )
except ImportError:
    from train import train_model
    from model_utils import (
        load_model, VISUALIZATIONS_DIR, PyTorchANNClassifier
    )


def test_genre_dataset_exists():
    p = PROJECT_DIR / "movie_genre.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert not df.empty
    assert "Genre" in df.columns


def test_genre_training_and_artifacts():
    acc, model = train_model()
    assert 0.0 <= acc <= 1.0
    assert isinstance(model, PyTorchANNClassifier)
    saved = load_model("genre_model.pkl")
    assert "model" in saved
    assert "preprocessor" in saved
    assert (VISUALIZATIONS_DIR / "genre_evaluation.png").exists()
