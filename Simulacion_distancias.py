import numpy as np
import pandas as pd

# Leer los datos del CSV
data = pd.read_csv("spike_trains.csv")
# Extraer solo los datos de los trenes de picos (omitimos la columna de etiquetas)
spike_trains = data.drop(columns=["Filter"]).values

# Obtener el número de trenes de picos
num_trains = spike_trains.shape[0]

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
            isi_distance_accumulator += np.abs((isi_x / isi_y) - 1)
        else:
            isi_distance_accumulator += np.abs(-((isi_y / isi_x) - 1))
    
    # Promediar la distancia ISI
    return isi_distance_accumulator


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
    N=len(spike_train_1)
    # Obtener los tiempos de picos (índices donde ocurren los "1s")
    spike_times_1 = get_spike_times(spike_train_1)
    spike_times_2 = get_spike_times(spike_train_2)

    spike_times_1 = np.insert(spike_times_1, 0, 0)    
    spike_times_1 = np.append(spike_times_1, N)    

    spike_times_2 = np.insert(spike_times_2, 0, 0)   
    spike_times_2 = np.append(spike_times_2, N)

    DS=0

    for t in range(0,N):
        # Encontrar el pico anterior y el siguiente para el tren 1
        previous_spike_1 = spike_times_1[spike_times_1 <= t].max() 
        following_spike_1 = spike_times_1[spike_times_1 > t].min() 
    
        # Encontrar el pico anterior y el siguiente para el tren 2
        previous_spike_2 = spike_times_2[spike_times_2 <= t].max() 
        following_spike_2 = spike_times_2[spike_times_2 > t].min()


        denominador=(following_spike_1-previous_spike_1+following_spike_2-previous_spike_2)/2  
    

    
        # Diferencias de tiempos entre picos anteriores y siguientes
        delta_P = previous_spike_1 - previous_spike_2
        delta_F = following_spike_1 - following_spike_2

        # Promedio de tiempo hasta el anterior y el siguiente spike
        avg_previous = (t - previous_spike_1 + t - previous_spike_2) / 2
        avg_following = (following_spike_1 - t + following_spike_2 - t) / 2

        # Calcular la distancia SPIKE
        DS += (abs(delta_P) * avg_following + abs(delta_F) * avg_previous) / denominador**2
    return DS




# Inicializar la matriz de distancia
distance_matrix = np.zeros((num_trains, num_trains))


print(get_spike_times(spike_trains[1]))
"""
# Calcular la matriz de distancia
for i in range(num_trains):
    for j in range(i + 1, num_trains):  # Solo calculamos la mitad superior
        isi_dist = isi_distance(spike_trains[i], spike_trains[j])
        spike_dist = spike_distance(spike_trains[i], spike_trains[j])
        
        # Promedio de las dos distancias
        average_distance = (isi_dist + spike_dist) / 2
        distance_matrix[i, j] = average_distance
        distance_matrix[j, i] = average_distance  # Simetría en la matriz

# Guardar la matriz de distancia en un CSV
distance_df = pd.DataFrame(distance_matrix)
distance_df.to_csv("distance_matrix.csv", index=False)
"""