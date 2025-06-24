import tensorflow as tf
from datetime import datetime
from src.data_loader import get_generators
from src.model import build_model

DATA_DIR = "data"

#Crear generadores
train_gen, val_gen, _ = get_generators(DATA_DIR)

#Instanciar el modelo (fase congelada)
model = build_model(fine_tune_at=None)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#Callbacks
log_dir = f"experiments/tensorboard/{datetime.now():%Y%m%d-%H%M%S}"
callbacks = [
  tf.keras.callbacks.TensorBoard(log_dir=log_dir),
  tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

#Entrenamiento inicial
history = model.fit(
  train_gen,
  validation_data=val_gen,
  epochs=10,
  callbacks=callbacks
)

#Fine-tuning
# Descongelar últimas N capas (ajusta N según experimentos)
model = build_model(fine_tune_at=-20)
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
ft_history = model.fit(
  train_gen,
  validation_data=val_gen,
  epochs=20,
  callbacks=callbacks
)

#Guardar modelo
model.save("models/cacao_classifier.h5", include_optimizer=False)
print("Modelo guardado en models/cacao_classifier.h5")
