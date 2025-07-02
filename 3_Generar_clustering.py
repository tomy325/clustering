# === 3_Generar_clustering.py actualizado ===
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform

# === FUNCIONES ===
def dendograma(distance_matrix, method='ward', save_path=None):
    condensed_distance_matrix = squareform(distance_matrix)
    linked = linkage(condensed_distance_matrix, method)
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=None, distance_sort='descending')
    plt.title(f'Hierarchical Clustering Dendrogram ({method})')
    plt.xlabel('Spike Train Index')
    plt.ylabel('Distance')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def hierarchical(distance_matrix, num_clusters, method='ward'):
    condensed_distance_matrix = squareform(distance_matrix)
    linked = linkage(condensed_distance_matrix, method)
    clusters = fcluster(linked, t=num_clusters, criterion='maxclust')
    return clusters

# === LECTURA DE MÉTODOS DESDE ARCHIVO AUXILIAR ===
with open("clustering_methods.txt") as f:
    methods = [line.strip() for line in f if line.strip()]

# === LECTURA DE ARCHIVOS ===
mean_isi_spike = pd.read_csv('matriz_distancia.csv')
isi = pd.read_csv('matriz_ISI.csv')
spike = pd.read_csv('matriz_SPIKE.csv')
a1 = pd.read_csv('areav1.csv')
a2 = pd.read_csv('areav2.csv')
df = pd.read_csv('spike_trains.csv')
fourier_opt = pd.read_csv("fourier_opt_matriz.csv")


# === INPUT DE NÚMERO DE CLUSTERS ===
num_clusters = int(input())

for method in methods:
    print(f"Procesando método de clustering: {method}")
    
    dendograma(mean_isi_spike, method=method, save_path=f"dendro_mean_isi_spike_{method}.png")
    clusters1 = hierarchical(mean_isi_spike, num_clusters, method=method)

    dendograma(isi, method=method, save_path=f"dendro_isi_{method}.png")
    clusters2 = hierarchical(isi, num_clusters, method=method)

    dendograma(spike, method=method, save_path=f"dendro_spike_{method}.png")
    clusters3 = hierarchical(spike, num_clusters, method=method)

    dendograma(a1, method=method, save_path=f"dendro_area1_{method}.png")
    clusters4 = hierarchical(a1, num_clusters, method=method)

    dendograma(a2, method=method, save_path=f"dendro_area2_{method}.png")
    clusters5 = hierarchical(a2, num_clusters, method=method)

    dendograma(fourier_opt, method=method, save_path=f"dendro_fourier_opt_{method}.png")
    clusters6 = hierarchical(fourier_opt, num_clusters, method=method)

    # Guardar resultados con nombre específico por método
    file_name = f"clusterizado_{method}.csv"
    df_temp = df.copy()
    df_temp[f'clusters_mean_isi_spike_{method}'] = clusters1
    df_temp[f'clusters_isi_{method}'] = clusters2
    df_temp[f'clusters_spike_{method}'] = clusters3
    df_temp[f'clusters_NA_{method}'] = clusters4
    df_temp[f'clusters_NA_suavizado_{method}'] = clusters5
    df_temp[f'clusters_fourier_opt_{method}'] = clusters6
    df_temp.to_csv(file_name, index=False)



