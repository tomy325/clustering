import os
import shutil
import subprocess
import datetime
import re

# === CONFIGURACIÓN INICIAL ===

# Pedir al usuario el número de clusters a usar en todos los métodos
num_clusters = input("Ingrese el número de clusters para todos los métodos de clustering: ").strip()

# Validar entrada
if not num_clusters.isdigit():
    raise ValueError("La cantidad de clusters debe ser un número entero.")
cluster_inputs = f"{num_clusters}\nclusterizado\n"


num_trials = input("Ingrese el número de ensayos por filtro (num_trials_per_filter): ").strip()
if not num_trials.isdigit():
    raise ValueError("El número de ensayos debe ser un número entero.")


# Método de clustering
valid_methods = ['ward', 'single', 'complete', 'average', 'centroid', 'median']
method = input(f"Ingrese el método de clustering jerárquico ({', '.join(valid_methods)}): ").strip().lower()
if method not in valid_methods:
    raise ValueError(f"Método inválido. Debe ser uno de: {', '.join(valid_methods)}")

# Guardar el método para uso posterior
with open("clustering_method.txt", "w") as f:
    f.write(method)

file_path = "1_Generar_datos.py"
with open(file_path, "r") as f:
    code = f.read()

# Reemplaza cualquier asignación a num_trials_per_filter por el nuevo valor
code = re.sub(r"num_trials_per_filter\s*=\s*\d+", f"num_trials_per_filter = {num_trials}", code)

# Guardar el archivo actualizado
with open(file_path, "w") as f:
    f.write(code)

# === CREAR CARPETA DE SALIDA CON FECHA Y HORA ACTUAL PARA EVITAR SOBRESCRIBIR LA CARPETA===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"resultados_pipeline_{timestamp}"
os.makedirs(output_folder)

# Guardar el nombre de la carpeta para usarlo en otros scripts
with open("output_path.txt", "w") as f:
    f.write(output_folder)

# === PASO 1: Generar datos ===
print("Ejecutando 1_Generar_datos.py...")
subprocess.run(["python", "1_Generar_datos.py"])

# Mover spike_trains.csv a la carpeta de salida
shutil.move("spike_trains.csv", os.path.join(output_folder, "spike_trains.csv"))

# === PASO 2: Calcular distancias ===
print("Ejecutando 2_Generar_distancias.py...")
shutil.copy(os.path.join(output_folder, "spike_trains.csv"), "spike_trains.csv")
subprocess.run(["python", "2_Generar_distancias.py"])
for fname in ["matriz_distancia.csv", "matriz_ISI.csv", "matriz_SPIKE.csv"]:
    shutil.move(fname, os.path.join(output_folder, fname))

# === PASO 3: Nelson-Aalen areas ===
print("Ejecutando 2.1_Nelson_areas.py...")
shutil.copy(os.path.join(output_folder, "spike_trains.csv"), "spike_trains.csv")
subprocess.run(["python", "2.1_Nelson_areas.py"])
for fname in ["areav1.csv", "areav2.csv"]:
    shutil.move(fname, os.path.join(output_folder, fname))

# === PASO 4: Clustering + Dendrogramas ===
print("Ejecutando 3_Generar_clustering.py...")

# Crear archivo de entrada automática para clustering
with open("cluster_input.txt", "w") as f:
    f.write(cluster_inputs)

# Copiar los archivos necesarios al directorio actual
shutil.copy(os.path.join(output_folder, "spike_trains.csv"), "spike_trains.csv")
for fname in ["matriz_distancia.csv", "matriz_ISI.csv", "matriz_SPIKE.csv", "areav1.csv", "areav2.csv"]:

    shutil.copy(os.path.join(output_folder, fname), fname)

# Ejecutar clustering
with open("cluster_input.txt", "r") as f:
    subprocess.run(["python", "3_Generar_clustering.py"], stdin=f)

# Mover archivo final clusterizado
shutil.move("clusterizado.csv", os.path.join(output_folder, "clusterizado.csv"))

# Mover imágenes de dendrogramas
for fname in ["dendro_mean_isi_spike.png", "dendro_isi.png", "dendro_spike.png", "dendro_area1.png", "dendro_area2.png"]:
    if os.path.exists(fname):
        shutil.move(fname, os.path.join(output_folder, fname))

# === PASO 5: Análisis de clusters ===
print("Ejecutando 4_Analizar_clusters.py...")

# Ejecutar análisis (el script debe leer output_path.txt)
subprocess.run(["python", "4_Analizar_clusters.py"])

# === LIMPIEZA  ===
for fname in ["cluster_input.txt", "spike_trains.csv", "matriz_distancia.csv", "matriz_ISI.csv", "matriz_SPIKE.csv", "areav1.csv", "areav2.csv"]:
    if os.path.exists(fname):
        os.remove(fname)


# === PASO 6: Crear tablas resumen por métrica y cluster ===
print("Generando tablas resumen por métrica y cluster...")

from tablas import generar_tabla_conteo

resumenes = {
    "mean_isi_spike.csv": "resumen_mean_isi_spike.csv",
    "isi.csv": "resumen_isi.csv",
    "spike.csv": "resumen_spike.csv",
    "area1.csv": "resumen_area1.csv",
    "area2.csv": "resumen_area2.csv",
}

for entrada, salida in resumenes.items():
    input_path = os.path.join(output_folder, entrada)
    output_path = os.path.join(output_folder, salida)
    if os.path.exists(input_path):
        tabla = generar_tabla_conteo(input_path, output_path)

from tablas import graficar_resumen_cluster

resumenes = {
    "mean_isi_spike.csv": "mean_isi_spike",
    "spike.csv": "spike",
    "isi.csv": "isi",
    "area1.csv": "area1",
    "area2.csv": "area2"
}

for archivo_csv, nombre_distancia in resumenes.items():
    path_csv = os.path.join(output_folder, f"resumen_{archivo_csv}")
    if os.path.exists(path_csv):
        graficar_resumen_cluster(path_csv, nombre_distancia, output_folder)

# === LIMPIEZA FINAL: eliminar archivos auxiliares de texto ===
text_files_to_remove = [
    "cluster_input.txt",
    "clustering_method.txt",
    "output_path.txt"
]

for txt_file in text_files_to_remove:
    if os.path.exists(txt_file):
        os.remove(txt_file)

print(f"\n🎉 Proceso completo. Revisa la carpeta: {output_folder}")