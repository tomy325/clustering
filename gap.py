import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def calculate_dispersion(distance_matrix, labels):
    """
    Calcula la dispersión intra-cluster como la suma de las distancias dentro de cada cluster.
    """
    dispersion = 0
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = np.where(labels == label)[0]
        cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
        dispersion += cluster_distances.sum() / (2 * len(cluster_points))  # Evitar doble conteo
    return dispersion

def eval_gap_stat(distance_matrix, max_clusters=10, n_random=10):
    """
    Calcula la estadística Gap para una matriz de distancia.
    """
    # Convertir la matriz de distancia a forma condensada
    condensed_matrix = squareform(distance_matrix)
    linked = linkage(condensed_matrix, method='ward')

    # Inicializar resultados
    gaps = []
    dispersions_original = []
    dispersions_random = []

    # Iterar sobre diferentes números de clusters
    for k in range(1, max_clusters + 1):
        # Clustering en los datos originales
        labels_original = fcluster(linked, t=k, criterion='maxclust')
        dispersion_original = calculate_dispersion(distance_matrix, labels_original)
        dispersions_original.append(dispersion_original)

        # Clustering en datos aleatorizados
        random_dispersions = []
        for _ in range(n_random):
            random_matrix = np.random.permutation(distance_matrix)  # Mezcla la matriz
            random_matrix = (random_matrix + random_matrix.T) / 2  # Asegurar simetría
            np.fill_diagonal(random_matrix, 0)  # Asegurar que la diagonal sea 0
            condensed_random = squareform(random_matrix)
            linked_random = linkage(condensed_random, method='ward')
            labels_random = fcluster(linked_random, t=k, criterion='maxclust')
            dispersion_random = calculate_dispersion(random_matrix, labels_random)
            random_dispersions.append(dispersion_random)

        # Calcular la estadística Gap
        mean_random_dispersion = np.mean(random_dispersions)
        dispersions_random.append(mean_random_dispersion)
        gaps.append(np.log(mean_random_dispersion) - np.log(dispersion_original))

    return gaps, dispersions_original, dispersions_random


def plot_gap_stat(gaps, max_clusters):
    """
    Genera un gráfico de la estadística Gap.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), gaps, marker='o', label='Gap Statistic')
    plt.title('Gap Statistic vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Gap Statistic')
    plt.grid(True)
    plt.legend()
    plt.show()

distancia_isi=pd.read_csv('ISI.csv')
distancia_spike=pd.read_csv('SPIKE.csv')

# Cargar matrices de distancia
distance_matrix_isi = distancia_isi.values  # Asegúrate de que sea un numpy array
distance_matrix_spike = distancia_spike.values  # Asegúrate de que sea un numpy array

# Calcular estadística Gap para ISI y SPIKE
gaps_isi, disp_ori_isi, disp_rand_isi = eval_gap_stat(distance_matrix_isi, max_clusters=10, n_random=10)
gaps_spike, disp_ori_spike, disp_rand_spike = eval_gap_stat(distance_matrix_spike, max_clusters=10, n_random=10)

# Graficar resultados
plot_gap_stat(gaps_isi, max_clusters=10)
plot_gap_stat(gaps_spike, max_clusters=10)
