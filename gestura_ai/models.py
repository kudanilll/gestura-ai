import tensorflow as tf

IMG_SIZE = (224, 224, 3)

def build_gestura_model(num_classes: int) -> tf.keras.Model:
    """
    Build a MobileNetV2-based classifier for BISINDO alphabet.
    We use ImageNet-pretrained weights and replace the top classifier.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE,
        include_top=False,
        weights="imagenet",
    )
    # First phase: keep the base frozen, only train the classification head.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE, name="input_image")

    # Pixels come in as [0, 255]; let preprocess_input normalize to [-1, 1].
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="gestura_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
