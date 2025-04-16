import numpy as np
import pandas as pd
import time  # Para medir el tiempo de ejecución


start_time = time.time()

# Leer los datos del CSV, omitiendo la columna de nombre del filtro
spike_trains = pd.read_csv("spike_trains.csv").drop(columns=["filter"])
t=np.linspace(0,21.5,1000)
# Obtener el número de trenes de picos
num_trains = spike_trains.shape[0]

# Nelson-Aalen: esta funcion recibe un spike train
# (arreglo con los timepos de ocurrencia de los spikes)
#  y calcula el estimador de Nelson
def nelson(event_times):
    n = len(event_times)
    eventos = np.arange(1, n+1)
    return event_times, eventos

def suavizado(t, x_conocidos, y_conocidos):
    return np.interp(t, x_conocidos, y_conocidos)

def area1(spiketrain):
    spiketrain = np.array(spiketrain)
    x, estimador = nelson(spiketrain)
    area = 0.0
    for i in range(len(spiketrain) - 1):
        intervalo = spiketrain[i+1] - spiketrain[i]
        area += intervalo * estimador[i]
    # Añadir el último tramo desde el último spike hasta el final (21.5)
    area += (21.5 - spiketrain[-1]) * estimador[-1]
    return area

def area2(spiketrain, t):
    spiketrain = np.array(spiketrain)
    x, estimador = nelson(spiketrain)
    interpolado = suavizado(t, x, estimador)
    area = 0.0
    for i in range(len(t) - 1):
        intervalo = t[i+1] - t[i]
        area += intervalo * interpolado[i]
    # Último tramo (opcional, ya cubierto normalmente)
    area += (21.5 - t[-1]) * interpolado[-1]
    return area


def diff_area(spiketrain1,spiketrain2,t):
    diffv1=abs(area1(spiketrain1) - area1(spiketrain2))
    diffv2=abs(area2(spiketrain1,t) - area2(spiketrain2,t))
    return  diffv1, diffv2


# Inicializar la matriz de distancia
area1_matrix = np.zeros((num_trains, num_trains))
area2_matrix = np.zeros((num_trains, num_trains))


for i in range(num_trains):
    for j in range(i + 1, num_trains):  # Solo calculamos la mitad superior
        a1,a2= diff_area(spike_trains.iloc[i].dropna(), spike_trains.iloc[j].dropna(),t)

        area1_matrix[i, j] = a1
        area1_matrix[j, i] = a1 

        area2_matrix[i, j] = a2
        area2_matrix[j, i] = a2 



# Guardar la matriz de distancia en un CSV
areav1 = pd.DataFrame(area1_matrix)
areav2=pd.DataFrame(area2_matrix)


areav1.to_csv("areav1.csv", index=False)
areav2.to_csv("areav2.csv", index=False)


# Medir el tiempo final
end_time = time.time()

# Mostrar el tiempo de ejecución
execution_time = end_time - start_time
print(f"El código tomó {execution_time:.2f} segundos en ejecutarse.")