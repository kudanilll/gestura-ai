from pathlib import Path
import json
import time

import cv2
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

def pick_model_path() -> Path:
    """
    Choose which model file to load, preferring the new .keras format.
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
        "Expected one of:\n"
        "  - gestura_bisindo_best.keras\n"
        "  - gestura_bisindo_last.keras\n"
        "  (or legacy .h5 files)\n"
        "Run training first: python -m gestura_ai.train"
    )

def load_model_and_labels():
    """
    Load the trained model (prefer .keras) and the label list from labels.json.
    """
    model_path = pick_model_path()
    print(f"[Gestura] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    labels_path = MODELS_DIR / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.json not found at {labels_path}. "
            "It should have been created by gestura_ai.train."
        )

    with labels_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    return model, class_names

def preprocess_frame(frame_bgr):
    """
    Preprocess a single BGR frame from OpenCV into a tensor suitable for
    MobileNetV2-based model:

    - Convert BGR -> RGB
    - Resize to 224x224
    - Convert to float32
    - Apply mobilenet_v2.preprocess_input
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))

    img = frame_resized.astype("float32")
    img_batch = np.expand_dims(img, axis=0)
    img_batch = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    return img_batch


def main():
    model, class_names = load_model_and_labels()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("[Gestura] Press 'q' to quit the webcam window.")

    last_pred = None
    last_pred_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Gestura] Failed to read frame from webcam.")
                break

            now = time.time()
            if now - last_pred_time > 0.2:  # ~5 FPS for inference
                input_tensor = preprocess_frame(frame)
                preds = model.predict(input_tensor, verbose=0)[0]

                pred_idx = int(np.argmax(preds))
                confidence = float(preds[pred_idx])
                label = class_names[pred_idx]

                last_pred = (label, confidence)
                last_pred_time = now

            if last_pred is not None:
                label, confidence = last_pred
                text = f"{label} ({confidence:.2f})"
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Gestura - Webcam Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
