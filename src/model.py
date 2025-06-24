import tensorflow as tf
from keras import layers, Model

NUM_CLASSES = 5  # ajusta a tus enfermedades + saludable

def build_model(fine_tune_at=None):
    # 1. Base pre-entrenada
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3)
    )
    base.trainable = False

    # 2. Cabeza de clasificación
    x = layers.Input(shape=(224,224,3))
    y = base(x, training=False)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dropout(0.3)(y)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(y)

    model = Model(x, outputs)

    # 3. Fine-tuning opcional
    if fine_tune_at is not None:
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base.layers[fine_tune_at:]:
            layer.trainable = True

    return model
