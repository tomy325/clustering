import os
import shutil
import subprocess
import datetime
import re

# === CONFIGURACIÓN INICIAL ===

# Pedir al usuario el número de clusters a usar en todos los métodos
num_clusters = input("Ingrese el número de clusters para todos los métodos de clustering: ").strip()

# === INPUT DE J (para Fourier optimizada) ===
J_valor = input("Ingrese el valor de J (y K) para la representación de Fourier optimizada: ").strip()
if not J_valor.isdigit():
    raise ValueError("J debe ser un número entero.")

# Validar entrada
if not num_clusters.isdigit():
    raise ValueError("La cantidad de clusters debe ser un número entero.")
cluster_inputs = f"{num_clusters}\nclusterizado\n"

file_path = "1_Generar_datos.py"


num_trials = input("Ingrese el número de ensayos por filtro (num_trials_per_filter): ").strip()
if not num_trials.isdigit():
    raise ValueError("El número de ensayos debe ser un número entero.")

# Pedir varianza (como porcentaje, por ejemplo 10 para 10%)
variabilidad_input = input("Ingrese el porcentaje de variabilidad para l y v (ejemplo: 10 para 10%): ").strip()
try:
    variabilidad_float = float(variabilidad_input)
except ValueError:
    raise ValueError("La variabilidad debe ser un número (puede ser decimal).")


# === INPUT: ¿Desea guardar archivos CSV intermedios? ===
guardar_csv = input("¿Desea conservar los archivos CSV generados? (s/n): ").strip().lower()
guardar_csv = guardar_csv == "s"  # True si 's', False si 'n'


# Guardar en el archivo 1_Generar_datos.py
with open(file_path, "r") as f:
    code = f.read()

# Reemplaza o agrega la línea de variabilidad
if "VARIABILIDAD_PORCENTAJE" in code:
    code = re.sub(r"VARIABILIDAD_PORCENTAJE\s*=\s*[\d.]+", f"VARIABILIDAD_PORCENTAJE = {variabilidad_float}", code)
else:
    code = re.sub(r"(# Parámetros para los ensayos.*?\n)", r"\1VARIABILIDAD_PORCENTAJE = " + str(variabilidad_float) + "\n", code, flags=re.DOTALL)

# Guardar de nuevo
with open(file_path, "w") as f:
    f.write(code)



# === PEDIR MÉTODOS DE CLUSTERING Y GUARDARLOS ===
valid_methods = ['ward', 'single', 'complete', 'average', 'centroid', 'median']
method_input = input(f"Ingrese uno o más métodos de clustering jerárquico separados por coma ({', '.join(valid_methods)}): ").strip().lower()

methods = [m.strip() for m in method_input.split(',') if m.strip() in valid_methods]
if not methods:
    raise ValueError(f"No se ingresaron métodos válidos. Deben ser alguno(s) de: {', '.join(valid_methods)}")

# Guardar métodos en clustering_methods.txt
with open("clustering_methods.txt", "w") as f:
    for method in methods:
        f.write(method + "\n")

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

# === PASO 2.2: Fourier distancia optimizada ===
print("Ejecutando 2.2_Fourier_dist_optimizada.py...")
shutil.copy(os.path.join(output_folder, "spike_trains.csv"), "spike_trains.csv")
with open("fourier_config.txt", "w") as f:
    f.write(J_valor)

subprocess.run(["python", "2.2_Fourier_dist_optimizada.py"])

# Verificar que el archivo se haya generado correctamente
fourier_file = "fourier_opt_matriz.csv"
if os.path.exists(fourier_file):
    shutil.move(fourier_file, os.path.join(output_folder, fourier_file))
    print(f"✅ {fourier_file} movido a {output_folder}")
else:
    raise FileNotFoundError(f"❌ No se encontró el archivo {fourier_file}. Revisa 2.2_Fourier_dist_optimizada.py.")

# === PASO 4: Clustering + Dendrogramas ===
print("Ejecutando 3_Generar_clustering.py...")

# Crear archivo de entrada automática para clustering
with open("cluster_input.txt", "w") as f:
    f.write(cluster_inputs)

# Copiar los archivos necesarios al directorio actual
shutil.copy(os.path.join(output_folder, "spike_trains.csv"), "spike_trains.csv")
for fname in ["matriz_distancia.csv", "matriz_ISI.csv", "matriz_SPIKE.csv", "areav1.csv", "areav2.csv", "fourier_opt_matriz.csv"]:
    shutil.copy(os.path.join(output_folder, fname), fname)

# Ejecutar clustering
with open("cluster_input.txt", "r") as f:
    subprocess.run(["python", "3_Generar_clustering.py"], stdin=f)



# Mover imágenes de dendrogramas
for fname in ["dendro_mean_isi_spike.png", "dendro_isi.png", "dendro_spike.png", "dendro_area1.png", "dendro_area2.png", "dendro_fourier_opt.png"]:
    if os.path.exists(fname):
        shutil.move(fname, os.path.join(output_folder, fname))

# === PASO 5: Análisis de clusters ===
print("Ejecutando 4_Analizar_clusters.py...")

# Ejecutar análisis (el script debe leer output_path.txt)
subprocess.run(["python", "4_Analizar_clusters.py"])

# === LIMPIEZA  ===
for fname in ["cluster_input.txt", "spike_trains.csv", "matriz_distancia.csv", "matriz_ISI.csv", "matriz_SPIKE.csv", "areav1.csv", "areav2.csv","fourier_opt_matriz.csv"]:
    if os.path.exists(fname):
        os.remove(fname)


# === MOVER ARCHIVOS CLUSTERIZADOS Y DENDROGRAMAS POR MÉTODO ===
for method in methods:
    cluster_file = f"clusterizado_{method}.csv"
    if os.path.exists(cluster_file):
        shutil.move(cluster_file, os.path.join(output_folder, cluster_file))
    
    for prefix in ["mean_isi_spike", "isi", "spike", "area1", "area2", "fourier_opt"]:
        dendro_file = f"dendro_{prefix}_{method}.png"
        if os.path.exists(dendro_file):
            shutil.move(dendro_file, os.path.join(output_folder, dendro_file))



# === LIMPIEZA FINAL: eliminar archivos auxiliares d8e texto ===
text_files_to_remove = [
    "cluster_input.txt",
    "clustering_methods.txt",  
    "output_path.txt",
    "fourier_config.txt"
]

for txt_file in text_files_to_remove:
    if os.path.exists(txt_file):
        os.remove(txt_file)


# === LIMPIEZA FINAL ===
if not guardar_csv:
    print("🧹 Eliminando todos los archivos .csv intermedios en la carpeta de salida...")
    for archivo in os.listdir(output_folder):
        if archivo.endswith(".csv"):
            os.remove(os.path.join(output_folder, archivo))
    print("✅ Archivos .csv eliminados.")
else:
    print("📁 Archivos .csv intermedios conservados.")




print(f"\n🎉 Proceso completo. Revisa la carpeta: {output_folder}")

