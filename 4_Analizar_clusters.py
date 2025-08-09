import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score, completeness_score
from sklearn.preprocessing import LabelEncoder

start_time = time.time()

# === RUTA DE SALIDA ===
with open("output_path.txt") as f:
    output_folder = f.read().strip()

os.makedirs(output_folder, exist_ok=True)

# === ARCHIVOS CLUSTERIZADOS A ANALIZAR ===
archivos = [f for f in os.listdir() if f.startswith("clusterizado_") and f.endswith(".csv")]

# === FUNCIÓN PARA EVALUAR Y GRAFICAR ===
def analizar_clusterizado(ruta_cluster):
    nombre_metodo = ruta_cluster.split("_")[-1].replace(".csv", "")
    df = pd.read_csv(ruta_cluster)

    columnas = {
        "mean_isi_spike": f"clusters_mean_isi_spike_{nombre_metodo}",
        "isi": f"clusters_isi_{nombre_metodo}",
        "spike": f"clusters_spike_{nombre_metodo}",
        #"area1": f"clusters_NA_{nombre_metodo}",
        #"area2": f"clusters_NA_suavizado_{nombre_metodo}",
        "fourier_opt": f"clusters_fourier_opt_{nombre_metodo}",
        "wavelet": f"clusters_wavelet_{nombre_metodo}",
        "wavelet_multi": f"clusters_wavelet_multi_{nombre_metodo}"

    }

    # Guardar archivos individuales
    for nombre, col in columnas.items():
        df[["filter", col]].to_csv(os.path.join(output_folder, f"{nombre}_{nombre_metodo}.csv"), index=False)

    # Codificar etiquetas verdaderas
    true_labels = LabelEncoder().fit_transform(df["filter"])

    # Evaluar clustering
    resultados = []
    for metodo in columnas.values():
        pred = df[metodo]
        ari = adjusted_rand_score(true_labels, pred)
        ami = adjusted_mutual_info_score(true_labels, pred)
        fmi = fowlkes_mallows_score(true_labels, pred)
        completeness = completeness_score(true_labels, pred)
        score_prom = round((ari + ami + fmi + completeness) / 4, 4)

        resultados.append({
            "método": metodo,
            "ARI": round(ari, 4),
            "AMI": round(ami, 4),
            "FMI": round(fmi, 4),
            "Completeness": round(completeness, 4),
            "Score promedio": score_prom    
        })

    # Guardar evaluación
    eval_df = pd.DataFrame(resultados)
    eval_path = os.path.join(output_folder, f"evaluacion_clusters_{nombre_metodo}.csv")
    eval_df.to_csv(eval_path, index=False)

    # === GRÁFICO: Score promedio ===
    plt.figure(figsize=(10, 5))
    plt.bar(eval_df["método"], eval_df["Score promedio"], color="slateblue")
    plt.title(f"Score promedio por método - {nombre_metodo}")
    plt.ylabel("Score promedio")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"comparacion_score_promedio_{nombre_metodo}.png"))
    plt.close()

    # === GRÁFICO: Comparación de métricas ===
    metricas = ["ARI", "AMI", "FMI", "Completeness"]
    metodos = eval_df["método"]
    valores = [eval_df[metrica].values for metrica in metricas]
    colores = ["#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b"]

    x = range(len(metodos))
    width = 0.2
    plt.figure(figsize=(12, 6))
    for i, (metrica, color) in enumerate(zip(metricas, colores)):
        plt.bar([xi + i * width for xi in x], valores[i], width=width, label=metrica, color=color)
    plt.xticks([xi + width * 1.5 for xi in x], metodos, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.ylabel("Valor")
    plt.title(f"Comparación de métricas de clustering – {nombre_metodo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"metricas_comparativas_{nombre_metodo}.png"))
    plt.close()

    # === HEATMAPS CLUSTERS ===
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for ax, (nombre, col) in zip(axs.flat, columnas.items()):
        contingency = pd.crosstab(df["filter"], df[col])
        sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
        ax.set_title(f"{nombre}")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Filtro")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"heatmaps_cluster_vs_filter_{nombre_metodo}.png"), dpi=300)
    plt.close()

    # === TABLA Y GRÁFICO DE CATEGORÍAS ===
    for nombre, col in columnas.items():
        path_csv = os.path.join(output_folder, f"{nombre}_{nombre_metodo}.csv")
        tabla_out = os.path.join(output_folder, f"tabla_conteo_{nombre}_{nombre_metodo}.csv")
        tabla = generar_tabla_conteo(path_csv, tabla_out)
        graficar_resumen_cluster(tabla_out, f"{nombre}_{nombre_metodo}", output_folder)

# === FUNCIONES AUXILIARES ===

def generar_tabla_conteo(path_csv, output_path=None):
    df = pd.read_csv(path_csv)
    if 'filter' not in df.columns:
        raise ValueError("El archivo debe contener una columna llamada 'filter'.")
    cluster_col = next((col for col in df.columns if col.startswith("clusters_")), None)
    if cluster_col is None:
        raise ValueError("No se encontró una columna de cluster que comience con 'clusters_'.")
    partes = df['filter'].str.split('_', expand=True)
    partes.columns = ['type', 'speed', 'duration']
    df = df.join(partes)

    filtros_completos = sorted(df['filter'].unique())
    tipos = ['ON', 'OF']
    velocidades = ['fast', 'slow']
    duraciones = ['sustained', 'transient']
    categorias = filtros_completos + tipos + velocidades + duraciones
    tabla = pd.DataFrame(index=categorias)

    for cluster in sorted(df[cluster_col].unique()):
        cluster_df = df[df[cluster_col] == cluster]
        conteo = {
            cat: cluster_df[
                (cluster_df['filter'] == cat) |
                (cluster_df['type'] == cat) |
                (cluster_df['speed'] == cat) |
                (cluster_df['duration'] == cat)
            ].shape[0]
            for cat in categorias
        }
        tabla[f"Cluster {cluster}"] = pd.Series(conteo)

    if output_path:
        tabla.to_csv(output_path)
    return tabla

def graficar_resumen_cluster(path_csv, nombre_distancia, output_folder):
    df = pd.read_csv(path_csv, index_col=0)
    filtros_completos = df.index[df.index.str.count('_') == 2]
    df_filtros = df.loc[filtros_completos]
    categorias = ['ON', 'OF', 'fast', 'slow', 'sustained', 'transient']
    df_categorias = df.loc[df.index.isin(categorias)]

    plt.figure(figsize=(10, 6))
    df_categorias.T.plot(kind='bar', stacked=False)
    plt.title(f"Categorías generales por cluster – {nombre_distancia}")
    plt.ylabel("Cantidad de elementos")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"barras_categoria_{nombre_distancia}.png"))
    plt.close()

# === LOOP PRINCIPAL DE ANÁLISIS ===
for archivo in archivos:
    print(f"📊 Analizando {archivo}...")
    analizar_clusterizado(archivo)

# === HEATMAP DE MATRICES DE DISTANCIA COMPARADAS ===
mats = {
    "Promedio ISI + SPIKE": "matriz_distancia.csv",
    "ISI Distance": "matriz_ISI.csv",
    "SPIKE Distance": "matriz_SPIKE.csv",
    #"Área sin suavizar": "areav1.csv",
    #"Área suavizada": "areav2.csv",
    "Fourier Optimizada": "fourier_opt_matriz.csv",
    "Wavelet Distance": "wavalet_matriz.csv",
    "Wavelet Multi Distance": "wavelet_multi_matriz.csv"
}

fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for ax, (titulo, archivo) in zip(axs.flat, mats.items()):
    m = pd.read_csv(os.path.join(output_folder, archivo))
    sns.heatmap(m, ax=ax, cmap="viridis", square=True, cbar=True, xticklabels=False, yticklabels=False)
    ax.set_title(titulo, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "heatmaps_distancias_comparativos.png"), dpi=300)
plt.close()

print("✅ Evaluación y gráficos guardados en:", output_folder)
print(f"⏱️ Tiempo de ejecución: {time.time() - start_time:.2f} segundos")
