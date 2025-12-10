from pathlib import Path
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 1337

# Root of the project: gestura/
ROOT_DIR = Path(__file__).resolve().parents[1]

# Images root: gestura/data/images/{train,val}/...
IMAGES_ROOT = ROOT_DIR / "data" / "images"

def _ensure_dir(path: Path, name: str) -> None:
    """
    Ensure that a required directory exists.
    This fails early with a helpful message instead of a cryptic stack trace.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{name} directory not found: {path}\n"
            "Expected structure:\n"
            "  data/\n"
            "    images/\n"
            "      train/  # class folders (A, B, ...)\n"
            "      val/    # class folders (A, B, ...)"
        )

def load_datasets():
    """
    Load train & validation datasets from:
      - data/images/train
      - data/images/val

    Each of those directories must contain one subfolder per class (A, B, ...).
    """
    train_dir = IMAGES_ROOT / "train"
    val_dir = IMAGES_ROOT / "val"

    _ensure_dir(train_dir, "Train")
    _ensure_dir(val_dir, "Validation")

    # image_dataset_from_directory will infer class names from subfolders
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        color_mode="rgb",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED,
        color_mode="rgb",
    )

    class_names = train_ds.class_names  # order of labels; must be saved for Flutter

    autotune = tf.data.AUTOTUNE

    def train_preprocess(image, label):
        """
        Basic data augmentation for training.
        We keep the pixel range in [0, 255] and let MobileNetV2's preprocess_input
        handle normalization later inside the model.
        """
        image = tf.cast(image, tf.float32)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, label

    def val_preprocess(image, label):
        """
        Validation preprocessing: only type casting, no augmentation.
        """
        image = tf.cast(image, tf.float32)
        return image, label

    train_ds = (
        train_ds
        .map(train_preprocess, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    val_ds = (
        val_ds
        .map(val_preprocess, num_parallel_calls=autotune)
        .prefetch(autotune)
    )

    return train_ds, val_ds, class_names
