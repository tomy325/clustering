import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform
import pandas as pd

def silhouette(matriz_distancia, max_clusters=10):
    sil_scores = []
    
    # Iterar sobre diferentes números de clusters
    for n_clusters in range(2, max_clusters + 1):
        # Aplicar KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=32)
        cluster_labels = kmeans.fit_predict(matriz_distancia)
        
        # Calcular el índice de Silhouette
        sil_score = silhouette_score(matriz_distancia, cluster_labels, metric='precomputed')
        sil_scores.append(sil_score)

    # Graficar el índice de Silhouette para diferentes números de clusters
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, max_clusters + 1), sil_scores, marker='o')
    plt.title('Índice de Silhouette para diferentes clusters')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()


def codo(matriz_distancia, max_clusters=10):
    inertias = []
    
    # Iterar sobre diferentes números de clusters
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=32)
        kmeans.fit(matriz_distancia)
        
        # Guardar la inercia
        inertias.append(kmeans.inertia_)

    # Graficar el gráfico de codo
    plt.figure(figsize=(6, 4))
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.title('Gráfico de Codo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.grid(True)
    plt.show()

def hierarchical(distance_matrix, num_clusters=8,  method='ward'):
    """
    Perform hierarchical clustering on the distance matrix and plot the dendrogram.
    
    Parameters:
    distance_matrix : array
        Matriz de distancias que se utilizará para el clustering.
    num_clusters : int
        Número deseado de clusters para el corte.
    
    Returns:
    clusters : array
        Array con los clusters asignados a cada tren de picos.
    """
    # Convertir la matriz de distancias completa en una forma condensada
    condensed_distance_matrix = squareform(distance_matrix)
    
    # Perform clustering
    linked = linkage(condensed_distance_matrix, method)
    
    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=np.arange(distance_matrix.shape[0]), distance_sort='descending')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Spike Train Index')
    plt.ylabel('Distance')
    plt.show()
    
    # Asignar clusters utilizando el número de clusters deseado
    clusters = fcluster(linked, t=num_clusters, criterion='maxclust')
    
    return clusters


distancia=pd.read_csv('matriz_distancia.csv')
silhouette(distancia)
codo(distancia)



clusters1=hierarchical(distancia ,8, 'single')
clusters2=hierarchical(distancia ,8, 'complete') 
clusters3=hierarchical(distancia ,8, 'average')
clusters4=hierarchical(distancia ,8, 'ward')
