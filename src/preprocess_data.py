import splitfolders

# Divide los datos originales en 70% train, 15% val y 15% test
splitfolders.ratio(
    "data",  # Carpeta de origen con las imágenes organizadas por clase
    output="data",  # Carpeta de salida donde se generarán las subcarpetas train, val y test
    seed=42,  # Semilla para garantizar que la división sea reproducible
    ratio=(0.7, 0.15, 0.15),  # Proporción de los datos
    group_prefix=None  # Si no hay prefijos de grupo para las imágenes
)

print("Datos divididos en train, val y test.")