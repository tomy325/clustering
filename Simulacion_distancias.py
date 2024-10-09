import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.cluster.hierarchy import fcluster

# Ajustar parametros para r(x)
r_min = 0.5
r_max = 100
c = 4

# Parámetros para los ensayos (Spikes)
num_trials_per_filter = 5  # 50 ensayos por filtro
dt = 0.001  # Intervalo de tiempo (1 ms)

# Definir las combinaciones posibles de parámetros en un diccionario
filters_params = {
    'ON_fast_sustained': {'p': 1, 'l': 0.4, 'v': 1.2},
    'OF_fast_sustained': {'p': -1, 'l': 0.4, 'v': 1.2},
    'ON_slow_sustained': {'p': 1, 'l': 1, 'v': 1.2},
    'OF_slow_sustained': {'p': -1, 'l': 1, 'v': 1.2},
    'ON_fast_transient': {'p': 1, 'l': 0.4, 'v': 0.65},
    'OF_fast_transient': {'p': -1, 'l': 0.4, 'v': 0.65},
    'ON_slow_transient': {'p': 1, 'l': 1, 'v': 0.65},
    'OF_slow_transient': {'p': -1, 'l': 1, 'v': 0.65}
}

# Definir funciones
# Función r(x)
def r_function(t, min=0.5, max=100, c=4): 
    r = ((2 * max - min) / (1 + np.exp(-c * (t - 1)))) + min
    return r

# Función de intensidad/hazard/riesgo/tasa
def filter_result(t):
    h = np.sin(2 * t) * np.exp(-t**2 / 4)
    return h

# Filtro gaussiano
def gauss(p, t, mu, sigma, v):
    pdf = norm.pdf(t, mu, sigma/2)
    kate = p * pdf * np.sin(2 * np.pi * (t / sigma)**v)
    return kate

# Función estímulo
def estimulo(t):
    if t <= 1.5:
        return -1
    elif t > 1.5 and t <= 3.5:
        return 1
    elif t > 3.5 and t <= 5.5:
        return -1
    elif t > 5.5 and t <= 7.5:
        return 0
    elif t > 7.5 and t <= 12.5:
        return np.sin(np.pi * (t - 7.5)**2)
    elif t > 12.5 and t <= 14.5:
        return 0
    elif t > 14.5 and t <= 19.5:
        return 0.2 * (t - 14.5) * np.sin(3 * np.pi * (t - 14.5))
    elif t > 19.5 and t <= 21.5:
        return 0

# Convolución lineal
def linear_response(f, g):
    N = len(f)
    response = np.zeros(N)
    for n in range(N):
        for k in range(N):
            if n - k >= 0:
                response[n] += f[k] * g[n - k]
    return response


import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

def get_spike_times(spike_train):
    """
    Dado un arreglo binario (0s y 1s), devuelve los índices (posiciones) donde hay picos (1s).
    """
    return np.where(spike_train == 1)[0]  # Devuelve los índices donde hay "1s" (spikes)

def isi_distance(spike_train_1, spike_train_2):
    """
    Calcula la distancia ISI entre dos trenes de picos (spike trains) representados como arreglos binarios.
    
    Parameters:
    spike_train_1 : array-like
        Primer tren de picos (arreglo de 0s y 1s).
    spike_train_2 : array-like
        Segundo tren de picos (arreglo de 0s y 1s).
        
    Returns:
    float
        La distancia ISI entre los dos trenes de picos.
    """
    # Obtener los tiempos (índices) donde ocurren los picos (1s)
    spike_times_1 = get_spike_times(spike_train_1)
    spike_times_2 = get_spike_times(spike_train_2)
    
    # Calcular los intervalos inter-picos (ISI)
    ISI_1 = np.diff(spike_times_1)  # Diferencias entre tiempos consecutivos de picos
    ISI_2 = np.diff(spike_times_2)

    # Asegurarnos de que ambos trenes de picos tienen suficientes ISIs para comparar
    num_intervals = min(len(ISI_1), len(ISI_2))

    
    ##################################

    # Acumular la distancia ISI
    isi_distance_accumulator = 0

    for isi_x, isi_y in zip(ISI_1[:num_intervals], ISI_2[:num_intervals]):
        if isi_x <= isi_y:
            isi_distance_accumulator += (isi_x / isi_y) - 1
        else:
            isi_distance_accumulator += -((isi_y / isi_x) - 1)
    
    # Promediar la distancia ISI
    distance = np.abs(isi_distance_accumulator) 
    return distance


def spike_distance(spike_train_1, spike_train_2):
    """
    Calcula la distancia SPIKE entre dos trenes de picos (spike trains) representados como arreglos binarios (0s y 1s).
    
    Parameters:
    spike_train_1 : array-like
        Primer tren de picos (arreglo de 0s y 1s).
    spike_train_2 : array-like
        Segundo tren de picos (arreglo de 0s y 1s).
    t : float
        Tiempo en el que calcular la distancia SPIKE.
        
    Returns:
    float
        La distancia SPIKE entre los dos trenes de picos en el tiempo t.
    """
    # Obtener los tiempos de picos (índices donde ocurren los "1s")
    spike_times_1 = get_spike_times(spike_train_1)
    spike_times_2 = get_spike_times(spike_train_2)

    spike_times_1 = np.insert(spike_times_1, 0, 0)    
    spike_times_1 = np.append(spike_times_1, 1000)    

    spike_times_2 = np.insert(spike_times_2, 0, 0)   
    spike_times_2 = np.append(spike_times_2, 1000)

    DS=0

    for t in range(0,1000):
        # Encontrar el pico anterior y el siguiente para el tren 1
        previous_spike_1 = spike_times_1[spike_times_1 <= t].max() 
        following_spike_1 = spike_times_1[spike_times_1 > t].min() 
    
        # Encontrar el pico anterior y el siguiente para el tren 2
        previous_spike_2 = spike_times_2[spike_times_2 <= t].max() 
        following_spike_2 = spike_times_2[spike_times_2 > t].min()  
    

    
        # Diferencias de tiempos entre picos anteriores y siguientes
        delta_P = previous_spike_1 - previous_spike_2
        delta_F = following_spike_1 - following_spike_2

        # Promedio de tiempo hasta el anterior y el siguiente spike
        avg_previous = (t - previous_spike_1 + t - previous_spike_2) / 2
        avg_following = (following_spike_1 - t + following_spike_2 - t) / 2

        # Calcular la distancia SPIKE
        DS += (abs(delta_P) * avg_following + abs(delta_F) * avg_previous) / (avg_previous + avg_following)**2
    DS= DS/1000
    return DS

def compute_distance_matrix(spike_trains):
    """
    Computa la matriz de distancias pareadas usando las distancias ISI y SPIKE.
    
    Parameters:
    spike_trains : list of arrays
        Lista de trenes de picos (spike trains) representados como arreglos binarios (0s y 1s).
        
    Returns:
    distance_matrix : array
        Matriz de distancias combinando ISI y SPIKE distances.
    """
    num_trains = len(spike_trains)
    distance_matrix = np.zeros((num_trains, num_trains))
    
    for i in range(num_trains):
        for j in range(i + 1, num_trains):  # Solo calculamos la parte superior de la matriz, ya que es simétrica
            
            # Calcular las distancias ISI y SPIKE
            isi_dist = isi_distance(spike_trains[i], spike_trains[j])
            spike_dist = spike_distance(spike_trains[i], spike_trains[j])
            

            distance_matrix[i, j] = (isi_dist + spike_dist) / 2
            
            # Copiar el valor a la parte simétrica inferior de la matriz
            distance_matrix[j, i] = distance_matrix[i, j]
    
    return distance_matrix


def hierarchical_clustering(distance_matrix, num_clusters=8):
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
    # Perform clustering
    linked = linkage(distance_matrix, method='ward')
    
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


# Crear un vector de tiempo limitado entre 0 y 21.5
t = np.linspace(0, 21.5, 1000)

# Matriz para almacenar todos los spikes (8 tipos de filtros * 50 ensayos = 400 filas)
all_spike_trains = np.zeros((num_trials_per_filter * len(filters_params), len(t)))

# Etiquetas para saber a qué filtro pertenecía cada ensayo (se revolverá luego)
labels = []

# Iterar sobre cada filtro en el diccionario
trial_index = 0
for filter_name, params in filters_params.items():
    p = params['p']
    l = params['l']
    v = params['v']

    # Evaluar las funciones gauss y estimulo en ese dominio
    gauss_values = gauss(p, t, 0, l, v)
    estimulo_values = np.array([estimulo(i) for i in t])

    # Realizar la convolución manual
    response = linear_response(gauss_values, estimulo_values)

    # Normalizar la respuesta entre -1 y 1
    response_min = np.min(response)
    response_max = np.max(response)
    normalized_response = (response - response_min) / (response_max - response_min) * 2 - 1

    # Calcular el rate (tasa de disparo) basado en la convolución normalizada
    rate = np.array([r_function(j) for j in normalized_response])

    # Generar los trenes de spikes para 50 ensayos
    for trial in range(num_trials_per_filter):
        spike_trains = np.random.rand(len(t)) < rate * dt
        all_spike_trains[trial_index] = spike_trains
        labels.append(filter_name)  # Agregar la etiqueta correspondiente a este ensayo
        trial_index += 1

# Revolver los datos y sus etiquetas
shuffled_indices = np.random.permutation(all_spike_trains.shape[0])
all_spike_trains = all_spike_trains[shuffled_indices]
shuffled_labels = np.array(labels)[shuffled_indices]


matriz=compute_distance_matrix(all_spike_trains)
# Llamar a la función de clustering jerárquico y obtener los clusters
clusters = hierarchical_clustering(matriz, num_clusters=8)

# Contar el número de elementos en cada cluster
unique_clusters, counts = np.unique(clusters, return_counts=True)

# Mostrar el conteo de datos en cada cluster
for cluster_id, count in zip(unique_clusters, counts):
    print(f"Cluster {cluster_id}: {count} elementos")
