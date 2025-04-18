import numpy as np
import pandas as pd
import time  # Para medir el tiempo de ejecución


start_time = time.time()

# Leer los datos del CSV, omitiendo la columna de nombre del filtro
spike_trains = pd.read_csv("spike_trains.csv").drop(columns=["filter"])

# Obtener el número de trenes de picos
num_trains = spike_trains.shape[0]




def isi_distance(spike_train_1, spike_train_2):
    """
    Calcula la distancia ISI entre dos trenes de picos (spike trains) representados como 
    arreglos que contienen los tiempos de las ocurrencias de los spikes.
    
    Parameters:
    spike_train_1 : 
        Primer tren de picos (puede contener valores NA).
    spike_train_2 :
        Segundo tren de picos (puede contener valores NA).
        
    Returns:
    float
        La distancia ISI entre los dos trenes de picos.
    """

    

    
    # Calcular los intervalos inter-picos (ISI)
    ISI_1 = np.diff(spike_train_1)  # Diferencias entre tiempos consecutivos de picos
    ISI_2 = np.diff(spike_train_2)

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
    Calcula la distancia SPIKE entre dos trenes de picos (spike trains) .
    
    Parameters:
    spike_train_1 : 
        Primer tren de picos 
    spike_train_2 : 
        Segundo tren de picos
    t : float
        Tiempo en el que calcular la distancia SPIKE.
        
    Returns:
    float
        La distancia SPIKE entre los dos trenes de picos en el tiempo t.
    """

        # Agregar 0 al inicio si no está presente
    if spike_train_1[0] != 0:
        spike_train_1 = np.insert(spike_train_1, 0, 0)
    if spike_train_2[0] != 0:
        spike_train_2 = np.insert(spike_train_2, 0, 0)

    # Agregar 21.5 al final si no está presente
    if spike_train_1[-1] != 21.5:
        spike_train_1 = np.append(spike_train_1, 21.5)
    if spike_train_2[-1] != 21.5:
        spike_train_2 = np.append(spike_train_2, 21.5)


    DS=0

    for t in np.linspace(0.000002, 21.488, 1000):
        # Encontrar el pico anterior y el siguiente para el tren 1
        previous_spike_1 = spike_train_1[spike_train_1 <= t].max() 
        following_spike_1 = spike_train_1[spike_train_1 > t].min() 
    
        # Encontrar el pico anterior y el siguiente para el tren 2
        previous_spike_2 = spike_train_2[spike_train_2 <= t].max() 
        following_spike_2 = spike_train_2[spike_train_2 > t].min()


        denominador=(following_spike_1-previous_spike_1+following_spike_2-previous_spike_2)/2  
    

    
        # Diferencias de tiempos entre picos anteriores y siguientes
        delta_P = previous_spike_1 - previous_spike_2
        delta_F = following_spike_1 - following_spike_2

        # Promedio de tiempo hasta el anterior y el siguiente spike
        inv_avg_previous = 1/((t - previous_spike_1 + t - previous_spike_2) / 2)
        inv_avg_following = 1/((following_spike_1 - t + following_spike_2 - t) / 2)

        # Calcular la distancia SPIKE
        DS += (abs(delta_P) * inv_avg_following + abs(delta_F) * inv_avg_previous) / denominador**2
    return DS




# Inicializar la matriz de distancia
distance_matrix = np.zeros((num_trains, num_trains))
ISI_matrix = np.zeros((num_trains, num_trains))
SPIKE_matrix = np.zeros((num_trains, num_trains))



for i in range(num_trains):
    for j in range(i + 1, num_trains):  # Solo calculamos la mitad superior
        isi_dist = isi_distance(spike_trains.iloc[i].dropna().to_numpy(), spike_trains.iloc[j].dropna().to_numpy())
        spike_dist = spike_distance(spike_trains.iloc[i].dropna().to_numpy(), spike_trains.iloc[j].dropna().to_numpy())
        
        # Promedio de las dos distancias
        average_distance = (isi_dist + spike_dist) / 2

        ISI_matrix[i, j] = isi_dist
        ISI_matrix[j, i] = isi_dist  # Simetría en la matriz

            
        SPIKE_matrix[i, j] = spike_dist
        SPIKE_matrix[j, i] = spike_dist  # Simetría en la matriz

        distance_matrix[i, j] = average_distance
        distance_matrix[j, i] = average_distance  # Simetría en la matriz



# Guardar la matriz de distancia en un CSV
distance_df = pd.DataFrame(distance_matrix)
ISI_df=pd.DataFrame(ISI_matrix)
SPIKE_df=pd.DataFrame(SPIKE_matrix)

distance_df.to_csv("matriz_distancia.csv", index=False)
ISI_df.to_csv("matriz_ISI.csv", index=False)
SPIKE_df.to_csv("matriz_SPIKE.csv", index=False)

# Medir el tiempo final
end_time = time.time()

# Mostrar el tiempo de ejecución
execution_time = end_time - start_time
print(f"El código tomó {execution_time:.2f} segundos en ejecutarse.")
