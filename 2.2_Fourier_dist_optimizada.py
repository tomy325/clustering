import numpy as np
import pandas as pd
from itertools import combinations
import os

# ======================= LEER CONFIGURACIÓN =======================
with open("fourier_config.txt", "r") as f:
    N = int(f.read().strip())  # J = K

# ======================= CARGA DE DATOS =======================
df = pd.read_csv("spike_trains.csv") 
neurons = df.drop(columns=["filter", "l", "v"]).values

num_neurons = len(neurons)

# ======================= FUNCIONES AUXILIARES =======================
def compute_phi_psi(spike_train, N):
    spike_train = spike_train[~np.isnan(spike_train)]/21.5
    phi = np.array([np.sum(np.sin(np.pi * spike_train * j)) for j in range(1, N + 1)])
    psi = np.array([np.sum(np.cos(np.pi * spike_train * k)) for k in range(1, N + 1)])

    # Normalizar por norma L2
    phi_norm = np.linalg.norm(phi)
    psi_norm = np.linalg.norm(psi)

    if phi_norm > 0:
        phi = phi / phi_norm
    if psi_norm > 0:
        psi = psi / psi_norm

    return phi, psi

# ======================= MATRIZ DE DISTANCIAS =======================
phi_psi = [compute_phi_psi(neurons[n], N) for n in range(num_neurons)]
dist_matrix = np.zeros((num_neurons, num_neurons))

for i, j in combinations(range(num_neurons), 2):
    phi_i, psi_i = phi_psi[i]
    phi_j, psi_j = phi_psi[j]
    delta_phi = phi_i - phi_j
    delta_psi = psi_i - psi_j
    distance = np.sum(delta_phi ** 2) + np.sum(delta_psi ** 2)
    dist_matrix[i, j] = distance
    dist_matrix[j, i] = distance

# ======================= GUARDAR RESULTADOS =======================
df_matrix = pd.DataFrame(dist_matrix)
df_matrix.to_csv("fourier_opt_matriz.csv", index=False)
print("✅ Matriz de distancia Fourier optimizada guardada como 'fourier_opt_matriz.csv.csv'")
