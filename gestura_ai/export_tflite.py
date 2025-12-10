# gestura_ai/export_tflite.py
from pathlib import Path
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

def convert_to_tflite():
    """
    Convert the best or last Keras model (.h5) into a TFLite model.

    Output: models/gestura_bisindo_v1.tflite
    """
    best_model_path = MODELS_DIR / "gestura_bisindo_best.h5"
    last_model_path = MODELS_DIR / "gestura_bisindo_last.h5"

    if best_model_path.exists():
        model_path = best_model_path
    elif last_model_path.exists():
        model_path = last_model_path
    else:
        raise FileNotFoundError(
            f"No .h5 model found in {MODELS_DIR}. "
            "Run training first (python -m gestura_ai.train)."
        )

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
