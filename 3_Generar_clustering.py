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

# === LECTURA DE MÉTODO DESDE ARCHIVO AUXILIAR ===
with open("clustering_method.txt") as f:
    method = f.read().strip()

# === LECTURA DE ARCHIVOS ===
mean_isi_spike = pd.read_csv('matriz_distancia.csv')
isi = pd.read_csv('matriz_ISI.csv')
spike = pd.read_csv('matriz_SPIKE.csv')
a1 = pd.read_csv('areav1.csv')
a2 = pd.read_csv('areav2.csv')
df = pd.read_csv('spike_trains.csv')

# === INPUT DE NÚMERO DE CLUSTERS ===
num_clusters = int(input())

# === CLUSTERING ===
dendograma(mean_isi_spike, method=method, save_path="dendro_mean_isi_spike.png")
clusters1 = hierarchical(mean_isi_spike, num_clusters, method=method)

dendograma(isi, method=method, save_path="dendro_isi.png")
clusters2 = hierarchical(isi, num_clusters, method=method)

dendograma(spike, method=method, save_path="dendro_spike.png")
clusters3 = hierarchical(spike, num_clusters, method=method)

dendograma(a1, method=method, save_path="dendro_area1.png")
clusters4 = hierarchical(a1, num_clusters, method=method)

dendograma(a2, method=method, save_path="dendro_area2.png")
clusters5 = hierarchical(a2, num_clusters, method=method)

# === GUARDADO ===
file_name1 = input("")
file_name1 = f"{file_name1}.csv" if not file_name1.endswith(".csv") else file_name1

df['clusters_mean_isi_spike'] = clusters1
df['clusters_isi'] = clusters2
df['clusters_spike'] = clusters3
df['clusters_NA'] = clusters4
df['clusters_NA_suavizado'] = clusters5

df.to_csv(file_name1, index=False)
