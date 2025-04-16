import pandas as pd
import os
import time  # Para medir el tiempo de ejecución


start_time = time.time()

# Leer la ruta de salida desde el archivo generado
with open("output_path.txt") as f:
    output_folder = f.read().strip()

ruta_archivo = os.path.join(output_folder, "clusterizado.csv")
# === Cargar datos ===
df = pd.read_csv(ruta_archivo)

# === Crear carpeta si no existe ===
os.makedirs(output_folder, exist_ok=True)

# === Guardar archivos por cluster, mostrando SOLO la métrica correspondiente ===

# 1. Mean ISI-SPIKE
df[['filter', 'clusters_mean_isi_spike']].to_csv(os.path.join(output_folder, 'mean_isi_spike.csv'), index=False)

# 2. ISI
df[['filter', 'clusters_isi']].to_csv(os.path.join(output_folder, 'isi.csv'), index=False)

# 3. SPIKE
df[['filter', 'clusters_spike']].to_csv(os.path.join(output_folder, 'spike.csv'), index=False)

# 4. Área sin suavizado (NA)
df[['filter', 'clusters_NA']].to_csv(os.path.join(output_folder, 'area1.csv'), index=False)

# 5. Área con suavizado (NA suavizado)
df[['filter', 'clusters_NA_suavizado']].to_csv(os.path.join(output_folder, 'area2.csv'), index=False)

print(f"✅ Archivos separados correctamente por métrica y cluster en: {output_folder}")


# Medir el tiempo final
end_time = time.time()

# Mostrar el tiempo de ejecución
execution_time = end_time - start_time
print(f"El código tomó {execution_time:.2f} segundos en ejecutarse.")
