# Importar los módulos necesarios
import os

# Ejecutar el script 1: Generar datos
print("Ejecutando 1_Generar_datos.py...")
os.system("python 1_Generar_datos.py")

# Ejecutar el script 2: Generar distancias
print("Ejecutando 2_Generar_distancias.py...")
os.system("python 2_Generar_distancias.py")

# Ejecutar el script 3: Generar clustering
print("Ejecutando 3_Generar_clustering.py...")
os.system("python 3_Generar_clustering.py")

print("Ejecución completa.")
