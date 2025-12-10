from pathlib import Path
import json
import os

import tensorflow as tf

from gestura_ai.datasets import load_datasets
from gestura_ai.models import build_gestura_model

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def configure_tf():
    """
    Configure TensorFlow device usage.

    - If GESTURA_FORCE_CPU=1, hide all GPUs and use CPU only.
    - Otherwise, if GPUs are available, enable memory growth and use them.
    """
    force_cpu = os.getenv("GESTURA_FORCE_CPU", "0") == "1"
    if force_cpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            print("[Gestura] Forcing CPU usage (GESTURA_FORCE_CPU=1).")
        except Exception as exc:  # noqa: BLE001
            print(f"[Gestura] Failed to hide GPUs: {exc}")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[Gestura] No GPU found. Training will run on CPU.")
        return

    print(f"[Gestura] Found {len(gpus)} GPU(s): {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[Gestura] Enabled memory growth on all GPUs.")
    except RuntimeError as exc:
        # Memory growth must be set before GPUs are initialized.
        print(f"[Gestura] Failed to set memory growth: {exc}")

def main():
    configure_tf()

    # 1. Load datasets
    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)

    print(f"[Gestura] Detected {num_classes} classes: {class_names}")

    # 2. Build model
    model = build_gestura_model(num_classes)

    # 3. Callbacks (checkpointing + early stopping + LR scheduling)
    checkpoint_path = MODELS_DIR / "gestura_bisindo_best.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            verbose=1,
        ),
    ]

    # 4. Training
    history = model.fit(  # noqa: F841  (history kept for future analysis if needed)
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
    )

    # 5. Save final (last-epoch) model and labels
    last_model_path = MODELS_DIR / "gestura_bisindo_last.h5"
    model.save(last_model_path)
    print(f"[Gestura] Saved last model to: {last_model_path}")

    labels_path = MODELS_DIR / "labels.json"
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"[Gestura] Saved labels to: {labels_path}")

if __name__ == "__main__":
    main()
