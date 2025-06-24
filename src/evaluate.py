import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from src.data_loader import get_generators

# 1. Carga modelo y generadores
model = tf.keras.models.load_model("models/cacao_classifier.h5")
_, _, test_gen = get_generators("data")

# 2. Predicciones
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
labels = list(test_gen.class_indices.keys())

# 3. Reporte de métricas
print(classification_report(y_true, y_pred, target_names=labels))

# 4. Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, cmap='Blues')
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i,j], ha='center')
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.show()
