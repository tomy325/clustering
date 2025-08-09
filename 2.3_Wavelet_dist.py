import numpy as np
import pywt
import pandas as pd
from itertools import combinations
import os
import matplotlib.pyplot as plt
import seaborn as sns


# === LEER CONFIGURACIÓN DE J DESDE ARCHIVO ===
with open("haar_config.txt", "r") as f:
    J = int(f.read().strip())


# === CARGA DE DATOS ===
df = pd.read_csv("spike_trains.csv")
neurons = df.drop(columns='filter', errors='ignore').values
num_trains = len(neurons)

# === FUNCIONES ===

def haar_phi(x):
    x = np.asarray(x)  # Asegura que x sea un array
    return np.where((0 <= x) & (x < 0.5), 1,
           np.where((0.5 <= x) & (x < 1), -1, 0))


def calcular_Phi(spike_train, j, K):
    """
    Calcula Phi_{n,k}^{(j)} para un spike_train dado.
    spike_train: array de tiempos de spike
    j: nivel de escala
    """
    Phi = np.zeros(K)
    for k in range(K):
        valores = 2**j * spike_train - k
        Phi[k] = np.sum(haar_phi(valores))
    return Phi


def distancia_maxima_haar(spike_train_n, spike_train_m, j):
    """
    Calcula la distancia máxima d_max entre dos neuronas usando representación Haar optimizada.
    """
    spike_train_n = spike_train_n[~np.isnan(spike_train_n)] / 21.5
    spike_train_m = spike_train_m[~np.isnan(spike_train_m)] / 21.5

    K = 2**j
    Phi_n = calcular_Phi(spike_train_n, j, K)
    Phi_m = calcular_Phi(spike_train_m, j, K)

    delta_Phi = Phi_n - Phi_m
    d_max = np.sum(delta_Phi ** 2)
    return d_max

def matriz_distancia_maxima(neuronas, j):
    """
    Calcula la matriz simétrica de distancias máximas entre spike trains.
    
    neuronas: ndarray de forma (n_neuronas, t_spikes), con valores NaN donde no hay spike.
    j: nivel de escala para Haar.
    """
    n = len(neuronas)
    matriz = np.zeros((n, n))

    for i in range(n):
        for k in range(i + 1, n):
            dist = distancia_maxima_haar(neuronas[i], neuronas[k], j)
            matriz[i, k] = dist
            matriz[k, i] = dist  # simétrica

    return matriz



dist_matrix=matriz_distancia_maxima(neurons, J)

# === GUARDAR MATRIZ ===
df_dist = pd.DataFrame(dist_matrix)
df_dist.to_csv("wavalet_matriz.csv", index=False)
print(f"✅ Matriz de distancia wavelet guardada como 'wavalet_matriz.csv'")

