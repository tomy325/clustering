import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generar_tabla_conteo(path_csv, output_path=None):
    df = pd.read_csv(path_csv)

    if 'filter' not in df.columns:
        raise ValueError("El archivo debe contener una columna llamada 'filter'.")

    cluster_col = next((col for col in df.columns if col.startswith("clusters_")), None)
    if cluster_col is None:
        raise ValueError("No se encontró una columna de cluster que comience con 'clusters_'.")

    # Separar 'filter' en type, speed y duration
    partes = df['filter'].str.split('_', expand=True)
    partes.columns = ['type', 'speed', 'duration']
    df = df.join(partes)

    # Categorías por grupo
    filtros_completos = sorted(df['filter'].unique())
    tipos = ['ON', 'OF']
    velocidades = ['fast', 'slow']
    duraciones = ['sustained', 'transient']

    # Construir índice ordenado manualmente
    categorias = filtros_completos + tipos + velocidades + duraciones
    tabla = pd.DataFrame(index=categorias)

    # Contar ocurrencias por cluster
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

    # Guardar CSV si se indica
    if output_path:
        tabla.to_csv(output_path)

    return tabla

def graficar_resumen_cluster(path_csv, nombre_distancia, output_folder):
    """
    Genera y guarda 3 gráficos a partir de una tabla resumen de conteo por categoría y cluster:
    - Heatmap de filtros completos
    - Barras agrupadas por categoría general
    - Barras agrupadas por filtro completo

    Parámetros:
    - path_csv: ruta al archivo .csv con resumen
    - nombre_distancia: str que aparecerá en el título de los gráficos
    - output_folder: carpeta donde guardar las imágenes
    """
    df = pd.read_csv(path_csv, index_col=0)

    filtros_completos = df.index[df.index.str.count('_') == 2]
    df_filtros = df.loc[filtros_completos]

    categorias = ['ON', 'OF', 'fast', 'slow', 'sustained', 'transient']
    df_categorias = df.loc[df.index.isin(categorias)]

    # === Gráfico 1: Heatmap ===
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_filtros, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Heatmap por filtro completo – {nombre_distancia}")
    plt.ylabel("Filtro")
    plt.xlabel("Cluster")
    plt.tight_layout()
    heatmap_path = os.path.join(output_folder, f"heatmap_{nombre_distancia}.png")
    plt.savefig(heatmap_path)
    plt.close()

    # === Gráfico 2: Barras por categoría general ===
    plt.figure(figsize=(10, 6))
    df_categorias.T.plot(kind='bar', stacked=False)
    plt.title(f"Categorías generales por cluster – {nombre_distancia}")
    plt.ylabel("Cantidad de elementos")
    plt.xlabel("Cluster")
    plt.tight_layout()
    barras_cat_path = os.path.join(output_folder, f"barras_categoria_{nombre_distancia}.png")
    plt.savefig(barras_cat_path)
    plt.close()



    return {
        "heatmap": heatmap_path,
        "barras_categoria": barras_cat_path,
    }

'''    # === Gráfico 3: Barras por filtros completos ===
    plt.figure(figsize=(12, 6))
    df_filtros.T.plot(kind='bar', stacked=False)
    plt.title(f"Filtros completos por cluster – {nombre_distancia}")
    plt.ylabel("Cantidad de elementos")
    plt.xlabel("Cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    barras_filtros_path = os.path.join(output_folder, f"barras_filtros_{nombre_distancia}.png")
    plt.savefig(barras_filtros_path)
    plt.close()'''