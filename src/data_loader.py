from keras.preprocessing.image import ImageDataGenerator
from src.data_loader import get_generators

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

def create_generators(base_dir):
    train_gen = train_datagen.flow_from_directory(
        f"{base_dir}/train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_gen = val_test_datagen.flow_from_directory(
        f"{base_dir}/val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_gen = val_test_datagen.flow_from_directory(
        f"{base_dir}/test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen, test_gen

train_gen, val_gen, test_gen = get_generators("data")
print("Número de imágenes de entrenamiento:", train_gen.samples)
print("Número de imágenes de validación:", val_gen.samples)
print("Número de imágenes de prueba:", test_gen.samples)