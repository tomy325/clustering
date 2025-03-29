import matplotlib.pyplot as plt
#from sklearn.metrics import silhouette_score
#from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform
import pandas as pd
from sklearn.metrics import confusion_matrix


def dendograma(distance_matrix,  method='ward'):
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
    
    return 

def hierarchical(distance_matrix, num_clusters,  method='ward'):
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
    
    # Asignar clusters utilizando el número de clusters deseado
    clusters = fcluster(linked, t=num_clusters, criterion='maxclust')

    return clusters






#cargar matrices de distancia
mean_isi_spike=pd.read_csv('matriz_distancia.csv')
isi=pd.read_csv('ISI.csv')
spike=pd.read_csv('SPIKE.csv')


#cargar spike trains
df=pd.read_csv('spike_trains.csv')








dendograma(mean_isi_spike,  method='ward')
num1=int(input('Ingrese el número de clusters para mean_isi_spike: '))
clusters1=hierarchical(mean_isi_spike, num1,  method='ward')

dendograma(isi,  method='ward')
num2=int(input('Ingrese el número de clusters para isi: '))
clusters2=hierarchical(isi, num2,  method='ward')

dendograma(spike,  method='ward')
num3=int(input('Ingrese el número de clusters para spike: '))
clusters3=hierarchical(spike, num3,  method='ward')



# Pedir al usuario el nombre del archivo
file_name1 = input("Ingrese el nombre del archivo clusterizado: ")

# Asegurar que tenga la extensión .csv
file_name1 = f"{file_name1}.csv" if not file_name1.endswith(".csv") else file_name1

# Agregar la columna 'clusters_d1' al DataFrame 'df'
df['clusters_mean_isi_spike'] = clusters1
df['clusters_isi'] = clusters2
df['clusters_spike'] = clusters3
df.to_csv(file_name1, index=False)






'''
# Pedir al usuario el nombre del archivo
file_name1 = input("Ingrese el nombre del archivo mean-isi-spike (sin extensión): ")

# Asegurar que tenga la extensión .csv
file_name1 = f"{file_name1}.csv" if not file_name1.endswith(".csv") else file_name1


# Pedir al usuario el nombre del archivo
file_name2 = input("Ingrese el nombre del archivo isi (sin extensión): ")

# Asegurar que tenga la extensión .csv
file_name2 = f"{file_name2}.csv" if not file_name2.endswith(".csv") else file_name2

# Pedir al usuario el nombre del archivo
file_name3 = input("Ingrese el nombre del archivo spike (sin extensión): ")

# Asegurar que tenga la extensión .csv
file_name3 = f"{file_name3}.csv" if not file_name3.endswith(".csv") else file_name3
'''