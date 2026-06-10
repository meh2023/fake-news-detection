from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "fake_news_model.joblib"


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python train.py` first."
        )
    return joblib.load(path)


def predict_news(text: str, model_path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    model = load_model(model_path)
    label = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)

    confidence = float(max(probabilities))
    prob_map = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": prob_map,
        "is_fake": label == "fake",
    }
