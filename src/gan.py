import tensorflow as tf
import os
from keras import layers
from data import Dataset

# Definición del generador
def build_generator(latent_dim):
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, input_dim=latent_dim),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh')
    ])
    return model

# Definición del discriminador
def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

#Definir la GAN (Generador + Discriminador)
def build_gan(generator, discriminator):
    discriminator.trainable = False  # El discriminador está congelado durante el entrenamiento del generador
    model = tf.keras.Sequential([generator, discriminator])
    return model

#Preprocesamiento de las imágenes
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (64, 64))  # Ajustamos a 64x64
    img = (img / 127.5) - 1.0  # Normalizamos a [-1, 1]
    return img

#Cargar las imágenes del dataset
def load_images_from_directory(directory):
    all_images = []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            img = load_and_preprocess_image(image_path)
            all_images.append(img)
    return Dataset.from_tensor_slices(all_images)

dataset = load_images_from_directory('data/train')  # Ajusta la ruta según tu proyecto

#
