import numpy as np
import pandas as pd

# Leer J desde archivo de configuración
with open("haar_config.txt", "r") as f:
    J = int(f.read().strip())

# Leer spike trains
df = pd.read_csv("spike_trains.csv")
neurons = df.drop(columns=["filter", "l", "v"], errors='ignore').values
num_trains = len(neurons)



# Función base Haar φ(x)
def haar_phi(x):
    return np.where((0 <= x) & (x < 0.5), 1,
           np.where((0.5 <= x) & (x < 1), -1, 0))

# Calcular Φ_{n,jk} para un spike train
def calcular_phi_njk(spike_train, J):
    spike_train = spike_train[~np.isnan(spike_train)]/21.5
    phi_njk = []
    for j in range(J + 1):
        for k in range(2 ** j):
            valores = 2**j * spike_train - k
            suma = np.sum(haar_phi(valores))
            phi_njk.append(suma)
    return np.array(phi_njk)

# Calcular todos los vectores Φ_n
all_phi = [calcular_phi_njk(neurons[i], J) for i in range(num_trains)]

# Calcular la matriz de distancia euclidiana entre Φ_n y Φ_m
dist_matrix = np.zeros((num_trains, num_trains))
for i in range(num_trains):
    for j in range(i + 1, num_trains):
        delta = all_phi[i] - all_phi[j]
        dist = np.sum(delta ** 2)
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

# Guardar la matriz
df_matrix = pd.DataFrame(dist_matrix)
df_matrix.to_csv("wavelet_multi_matriz.csv", index=False)
print("✅ Matriz de distancia Haar multiresolución guardada como 'wavelet_multi_matriz.csv'")
