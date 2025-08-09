# === main_pipeline.py MODIFICADO ===
import os
import shutil
import subprocess
import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time




# === INPUT ===
n_repeticiones = input("¿Cuántas veces deseas ejecutar el pipeline completo?: ").strip()
if not n_repeticiones.isdigit() or int(n_repeticiones) <= 0:
    raise ValueError("El número de repeticiones debe ser un entero positivo.")
n_repeticiones = int(n_repeticiones)

num_clusters = input("Ingrese el número de clusters para todos los métodos de clustering: ").strip()
if not num_clusters.isdigit():
    raise ValueError("La cantidad de clusters debe ser un número entero.")
cluster_inputs = f"{num_clusters}\nclusterizado\n"

J_valor = input("Ingrese el valor de J (y K) para la representación de Fourier optimizada: ").strip()
if not J_valor.isdigit():
    raise ValueError("J debe ser un número entero.")

J_haar = input("Ingrese el valor de J para la representación Haar: ").strip()
if not J_haar.isdigit():
    raise ValueError("J debe ser un número entero.")

num_trials = input("Ingrese el número de ensayos por filtro (num_trials_per_filter): ").strip()
if not num_trials.isdigit():
    raise ValueError("El número de ensayos debe ser un número entero.")

variabilidad_input = input("Ingrese el porcentaje de variabilidad para l y v (ejemplo: 10 para 10%): ").strip()
try:
    variabilidad_float = float(variabilidad_input)
except ValueError:
    raise ValueError("La variabilidad debe ser un número (puede ser decimal).")

guardar_csv = input("¿Desea conservar los archivos CSV generados? (s/n): ").strip().lower()
guardar_csv = guardar_csv == "s"

valid_methods = ['ward', 'single', 'complete', 'average', 'centroid', 'median', 'spectral']

method_input = input(f"Ingrese uno o más métodos de clustering jerárquico separados por coma ({', '.join(valid_methods)}): ").strip().lower()
methods = [m.strip() for m in method_input.split(',') if m.strip() in valid_methods]
if not methods:
    raise ValueError("No se ingresaron métodos válidos válidos.")

# Guardar métodos en clustering_methods.txt
with open("clustering_methods.txt", "w") as f:
    for method in methods:
        f.write(method + "\n")

# Carpeta raíz para todo el experimento
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
raiz_experimento = f"experimento_grupo_{timestamp}"
os.makedirs(raiz_experimento, exist_ok=True)

# === LOOP DE REPETICIONES ===
for i in range(1, n_repeticiones + 1):
    tiempos_distancias = {}  # Dict para registrar tiempos por método

    print(f"\n Ejecutando repetición {i} de {n_repeticiones}...\n")
    run_folder = os.path.join(raiz_experimento, f"run_{i:02d}")
    os.makedirs(run_folder, exist_ok=True)
    output_folder = run_folder

    with open("output_path.txt", "w") as f:
        f.write(output_folder)
    with open("haar_config.txt", "w") as f:
        f.write(J_haar)
    with open("fourier_config.txt", "w") as f:
        f.write(J_valor)

    with open("1_Generar_datos.py", "r") as f:
        code = f.read()
    code = re.sub(r"VARIABILIDAD_PORCENTAJE\s*=\s*[\d.]+", f"VARIABILIDAD_PORCENTAJE = {variabilidad_float}", code)
    code = re.sub(r"num_trials_per_filter\s*=\s*\d+", f"num_trials_per_filter = {num_trials}", code)
    with open("1_Generar_datos.py", "w") as f:
        f.write(code)

    subprocess.run(["python", "1_Generar_datos.py"])
    shutil.move("spike_trains.csv", os.path.join(output_folder, "spike_trains.csv"))

    shutil.copy(os.path.join(output_folder, "spike_trains.csv"), "spike_trains.csv")
    start_time = time.time()
    subprocess.run(["python", "2_Generar_distancias.py"])
    tiempos_distancias["ISI/SPIKE"] = round(time.time() - start_time, 2)
    for f in ["matriz_distancia.csv", "matriz_ISI.csv", "matriz_SPIKE.csv"]:
        shutil.move(f, os.path.join(output_folder, f))



    start_time = time.time()
    subprocess.run(["python", "2.2_Fourier_dist_optimizada.py"])
    tiempos_distancias["Fourier"] = round(time.time() - start_time, 2)
    shutil.move("fourier_opt_matriz.csv", os.path.join(output_folder, "fourier_opt_matriz.csv"))


    start_time = time.time()
    subprocess.run(["python", "2.3_Wavelet_dist.py"])
    tiempos_distancias["Wavelet"] = round(time.time() - start_time, 2)
    shutil.move("wavalet_matriz.csv", os.path.join(output_folder, "wavalet_matriz.csv"))

    start_time = time.time()
    subprocess.run(["python", "2.5_Wavelet_multi_dist.py"])
    tiempos_distancias["Haar Multi"] = round(time.time() - start_time, 2)
    shutil.move("wavelet_multi_matriz.csv", os.path.join(output_folder, "wavelet_multi_matriz.csv"))


    with open("cluster_input.txt", "w") as f:
        f.write(cluster_inputs)
    shutil.copy(os.path.join(output_folder, "spike_trains.csv"), "spike_trains.csv")
    for f in ["matriz_distancia.csv", "matriz_ISI.csv", "matriz_SPIKE.csv", "fourier_opt_matriz.csv", "wavalet_matriz.csv","wavelet_multi_matriz.csv"]:
        shutil.copy(os.path.join(output_folder, f), f)
    with open("cluster_input.txt", "r") as f:
        subprocess.run(["python", "3_Generar_clustering.py"], stdin=f)

    subprocess.run(["python", "4_Analizar_clusters.py"])

    for f in os.listdir():
        if f.startswith("clusterizado_") or f.startswith("dendro_") or f.endswith(".csv") or f.endswith(".png"):
            if os.path.exists(f):
                shutil.move(f, os.path.join(output_folder, f))
    
    pd.DataFrame([tiempos_distancias]).to_csv(os.path.join(output_folder, "tiempos_distancias.csv"), index=False)



# === GRAFICO FINAL ===
datos = []
for run_folder in sorted(os.listdir(raiz_experimento)):
    run_path = os.path.join(raiz_experimento, run_folder)
    if not os.path.isdir(run_path):
        continue
    for fname in os.listdir(run_path):
        if fname.startswith("evaluacion_clusters_") and fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(run_path, fname))
            df["repeticion"] = run_folder
            datos.append(df)

if datos:
    df_combined = pd.concat(datos, ignore_index=True)
    df_combined["tipo_distancia"] = df_combined["método"].str.extract(r"clusters_(.*)")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_combined, x="tipo_distancia", y="Score promedio", palette="Set2")
    sns.stripplot(data=df_combined, x="tipo_distancia", y="Score promedio", color='black', alpha=0.3, jitter=True)
    plt.title("Distribución del Score Promedio por Tipo de Distancia")
    plt.ylabel("Score Promedio")
    plt.xlabel("Tipo de Distancia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    boxplot_path = os.path.join(raiz_experimento, "boxplot_score_promedio.png")
    plt.savefig(boxplot_path)
    print(f"📊 Boxplot generado en: {boxplot_path}")
else:
    print("⚠️ No se encontraron archivos de evaluación para graficar.")


# === GRÁFICO DE TIEMPOS DE DISTANCIA ===
tiempos_acumulados = []
for run_folder in sorted(os.listdir(raiz_experimento)):
    run_path = os.path.join(raiz_experimento, run_folder)
    archivo_tiempo = os.path.join(run_path, "tiempos_distancias.csv")
    if os.path.exists(archivo_tiempo):
        df = pd.read_csv(archivo_tiempo)
        df["repeticion"] = run_folder
        tiempos_acumulados.append(df)

if tiempos_acumulados:
    df_tiempos = pd.concat(tiempos_acumulados, ignore_index=True).melt(id_vars="repeticion", var_name="Métrica", value_name="Tiempo (s)")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_tiempos, x="Métrica", y="Tiempo (s)", palette="Blues")
    plt.title("Tiempos de ejecución por tipo de métrica de distancia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path_grafico = os.path.join(raiz_experimento, "tiempos_matrices_distancia.png")
    plt.savefig(path_grafico)
    print(f"⏱️ Gráfico de tiempos guardado en: {path_grafico}")


# === LIMPIEZA FINAL DE ARCHIVOS CSV SI NO SE DEBEN GUARDAR ===
if not guardar_csv:
    print("🧹 Eliminando archivos .csv intermedios en cada carpeta de repetición...")
    for run_folder in sorted(os.listdir(raiz_experimento)):
        run_path = os.path.join(raiz_experimento, run_folder)
        if not os.path.isdir(run_path):
            continue
        for archivo in os.listdir(run_path):
            if archivo.endswith(".csv"):
                os.remove(os.path.join(run_path, archivo))
    print("✅ Archivos CSV eliminados correctamente.")


# === LIMPIEZA FINAL DE ARCHIVOS .txt AUXILIARES ===
textos_auxiliares = [
    "output_path.txt",
    "haar_config.txt",
    "fourier_config.txt",
    "clustering_methods.txt",
    "cluster_input.txt"
]

for fname in textos_auxiliares:
    if os.path.exists(fname):
        os.remove(fname)
print("Archivos .txt auxiliares eliminados.")

print("\n✅ Todas las repeticiones fueron ejecutadas correctamente. Resultados en:", raiz_experimento)



# === RESUMEN DE MÉTRICAS COMO IMAGEN ===
def save_dataframe_as_table(df, filename, title="Resumen de Métricas"):
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(df) + 1))  # Ajustar alto según filas
    ax.axis('off')
    tabla = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center',
                     colLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.2)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# === RESUMEN CSV E IMAGEN ===
scores = []
tiempos = []
for run_folder in sorted(os.listdir(raiz_experimento)):
    run_path = os.path.join(raiz_experimento, run_folder)
    if not os.path.isdir(run_path):
        continue
    for archivo in os.listdir(run_path):
        if archivo.startswith("evaluacion_clusters_") and archivo.endswith(".csv"):
            df = pd.read_csv(os.path.join(run_path, archivo))
            df["repeticion"] = run_folder
            scores.append(df)
    tiempo_path = os.path.join(run_path, "tiempos_distancias.csv")
    if os.path.exists(tiempo_path):
        df = pd.read_csv(tiempo_path)
        df["repeticion"] = run_folder
        tiempos.append(df)



if scores and tiempos:
    df_scores = pd.concat(scores, ignore_index=True)
    df_scores["tipo_distancia"] = df_scores["método"].str.extract(r"clusters_(.*)")

        # Normalizar nombres en df_scores ANTES del groupby
    mapeo = {
        'mean_isi_spike_ward': 'ISI/SPIKE',
        'isi_ward': 'ISI/SPIKE',
        'spike_ward': 'ISI/SPIKE',
        'fourier_opt_ward': 'Fourier',
        'wavelet_ward': 'Wavelet',
        'wavelet_multi_ward': 'Haar Multi'
    }
    df_scores["tipo_distancia"] = df_scores["tipo_distancia"].map(mapeo)

    df_tiempos = pd.concat(tiempos, ignore_index=True)
    df_tiempos = df_tiempos.melt(id_vars="repeticion", var_name="tipo_distancia", value_name="tiempo")

    resumen_score = df_scores.groupby("tipo_distancia")["Score promedio"].agg(["mean", "var"]).reset_index()
    resumen_tiempo = df_tiempos.groupby("tipo_distancia")["tiempo"].agg(["mean", "var"]).reset_index()

    
    resumen_final = pd.merge(resumen_score, resumen_tiempo, on="tipo_distancia", suffixes=("_score", "_tiempo"))
    resumen_final.columns = ["Métrica", "Score promedio (media)", "Score promedio (varianza)",
                             "Tiempo (media s)", "Tiempo (varianza s²)"]

    resumen_csv = os.path.join(raiz_experimento, "resumen_metricas.csv")
    resumen_png = os.path.join(raiz_experimento, "resumen_metricas.png")


    if not resumen_final.empty:
        resumen_final = resumen_final.round(2)
        resumen_final.to_csv(resumen_csv, index=False)

        save_dataframe_as_table(resumen_final, resumen_png)
        print(f"📋 Tabla resumen guardada como imagen en: {resumen_png}")
    else:
        print("⚠️ No se pudo generar la tabla resumen: el DataFrame está vacío.")

else:
    print("⚠️ No hay suficientes datos para generar el resumen.")


from fpdf import FPDF
from PIL import Image

# === GENERAR DOCUMENTO RESUMEN PDF ===
class PDFResumen(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Resumen de Clustering y Distancias', ln=True, align='C')
        self.ln(5)

    def add_image(self, title, path, w=180):
        if os.path.exists(path):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, title, ln=True)
            self.ln(2)
            self.image(path, w=w)
            self.ln(10)

pdf = PDFResumen()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Incluir heatmaps de las primeras 5 ejecuciones
for i in range(1, min(6, n_repeticiones + 1)):
    run_path = os.path.join(raiz_experimento, f"run_{i:02d}")
    heatmap_path1 = os.path.join(run_path, "heatmaps_cluster_vs_filter_ward.png")
    heatmap_path2 = os.path.join(run_path, "heatmaps_distancias_comparativos.png")
    pdf.add_image(f"Heatmap Cluster vs Filtro - Run {i:02d}", heatmap_path1)
    pdf.add_image(f"Heatmap Distancias Comparadas - Run {i:02d}", heatmap_path2)

# Boxplot Score promedio
boxplot_path = os.path.join(raiz_experimento, "boxplot_score_promedio.png")
pdf.add_image("Distribución del Score Promedio por Tipo de Distancia", boxplot_path)

# Gráfico de tiempos por métrica
tiempos_path = os.path.join(raiz_experimento, "tiempos_matrices_distancia.png")
pdf.add_image("Tiempos de ejecución por tipo de métrica de distancia", tiempos_path)

# Tabla resumen de métricas
tabla_resumen_path = os.path.join(raiz_experimento, "resumen_metricas.png")
pdf.add_image("Resumen de métricas (score y tiempo)", tabla_resumen_path)

# Guardar PDF
resumen_pdf = os.path.join(raiz_experimento, "resumen_experimento.pdf")
pdf.output(resumen_pdf)
print(f"📄 Documento resumen generado en: {resumen_pdf}")

   