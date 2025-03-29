import numpy as np
import pandas as pd
import pyspike as spk


spike_trains = pd.read_csv("spike_trains.csv").drop(columns=["filter"])
num_trains = spike_trains.shape[0]

def isi_distance(spike_train_1, spike_train_2):
    """
    Calcula la distancia ISI entre dos trenes de picos (spike trains) representados como 
    arreglos que contienen los tiempos de las ocurrencias de los spikes.
    
    Parameters:
    spike_train_1 : array-like
        Primer tren de picos (puede contener valores NA).
    spike_train_2 : array-like
        Segundo tren de picos (puede contener valores NA).
        
    Returns:
    float
        La distancia ISI entre los dos trenes de picos.
    """

    train_1 = spike_train_1.dropna().values
    train_2 = spike_train_2.dropna().values
    
    
    # Crear objetos SpikeTrain
    st1 = spk.SpikeTrain(train_1, edges=(0, 21.5))
    st2 = spk.SpikeTrain(train_2, edges=(0, 21.5))


    
    # Calcular los intervalos inter-picos (ISI)
    ISI_1 = np.diff(spike_train_1)  # Diferencias entre tiempos consecutivos de picos
    ISI_2 = np.diff(spike_train_2)
    # Calcular la ISI Distance

    isi_prof = spk.isi_profile(st1, st2)
    distance = isi_prof.avrg()

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
    return isi_distance_accumulator, distance



my_ISI_matrix = np.zeros((num_trains, num_trains))
py_matrix= np.zeros((num_trains, num_trains)) 





for i in range(num_trains):
    for j in range(i + 1, num_trains):  # Solo calculamos la mitad superior
        isi_dist , pydist = isi_distance(spike_trains.iloc[i].dropna().to_numpy(), spike_trains.iloc[j].dropna().to_numpy())
        my_ISI_matrix[i, j] = isi_dist
        my_ISI_matrix[j, i] = isi_dist  # Simetría en la matriz
        py_matrix[i, j] = pydist
        py_matrix[j, i] = pydist  # Simetría en la matriz        






ISI_df=pd.DataFrame(my_ISI_matrix)
ISI_df.to_csv("my_ISI.csv", index=False)

py_df=pd.DataFrame(py_matrix)
py_df.to_csv("py_ISI.csv", index=False)


