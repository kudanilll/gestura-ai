from pathlib import Path
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

def pick_model_path() -> Path:
    """
    Choose which model file to convert to TFLite, preferring the .keras format.
    """
    candidates = [
        MODELS_DIR / "gestura_bisindo_best.keras",
        MODELS_DIR / "gestura_bisindo_last.keras",
        MODELS_DIR / "gestura_bisindo_best.h5",
        MODELS_DIR / "gestura_bisindo_last.h5",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No model file found in models/ directory.\n"
        "Run training first: python -m gestura_ai.train"
    )

def convert_to_tflite():
    """
    Convert the chosen Keras model into a TFLite model.

    Output: models/gestura_bisindo_v1.tflite
    """
    model_path = pick_model_path()
    print(f"[Gestura] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    tflite_path = MODELS_DIR / "gestura_bisindo_v1.tflite"
    tflite_path.write_bytes(tflite_model)

    print(f"[Gestura] Saved TFLite model to: {tflite_path}")

if __name__ == "__main__":
    convert_to_tflite()
